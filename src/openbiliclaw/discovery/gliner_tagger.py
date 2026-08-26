"""Local zero-shot NER entity tags for the pool-entry tagging stage.

Wraps ``gliner-community/gliner_large-v2.5`` (Apache-2.0, multilingual,
DeBERTa-v3-large backbone) behind a lazily-loaded, thread-offloaded tagger so
candidate claims can carry entity mentions (game / anime / person / org …)
without any extra LLM calls. The heavy ``gliner`` dependency stays optional:
everything here imports cleanly without it and callers fail open.

Pure helpers (text building, entity normalization, JSON payload) are
unit-testable without torch; only :class:`GlinerEntityTagger` touches the
model.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

# Community mirror of urchade/gliner_large_license-v2.5. huggingface_hub
# honors ``HF_ENDPOINT``, so hosts behind the official endpoint can point at
# a mirror (e.g. https://hf-mirror.com) without code changes.
GLINER_DEFAULT_MODEL_ID = "gliner-community/gliner_large-v2.5"

# Domain labels for cross-platform discovery candidates. GLiNER takes these
# at predict time — zero-shot, no retraining. The v2.5 config caps forward
# passes at 30 label types; config normalization enforces the same ceiling.
DEFAULT_GLINER_LABELS: tuple[str, ...] = (
    "游戏",
    "动漫",
    "影视",
    "音乐",
    "人物",
    "组织",
    "产品",
    "地点",
)

GLINER_MAX_LABELS = 30

# Word-splitter strategies accepted by ``GlinerEntityTagger``. ``auto`` picks
# per text: CJK-dominant inputs use ``jieba`` (Chinese needs word-level
# splitting; the checkpoint's whitespace splitter treats a whole sentence as
# one token), everything else keeps the trained-in ``whitespace`` behavior so
# Latin multi-word entities stay coherent.
GLINER_SPLITTER_CHOICES: tuple[str, ...] = (
    "auto",
    "whitespace",
    "jieba",
    "hanlp",
    "universal",
)
DEFAULT_GLINER_WORD_SPLITTER = "auto"
# Texts whose CJK+kana+hangul character share reaches this fraction are routed
# through the CJK splitter under ``auto``.
CJK_RATIO_THRESHOLD = 0.2

_DEFAULT_THRESHOLD = 0.5
_DEFAULT_MAX_CHARS = 512
_DEFAULT_MAX_ENTITIES = 16
_ENTITY_TEXT_MAX_CHARS = 48

# Schema-defying placeholder spans (same spirit as tag_channel's set).
_ENTITY_PLACEHOLDERS = frozenset(
    {
        "",
        "-",
        "—",
        "–",
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "unknown",
        "undefined",
        "待定",
        "未知",
        "无",
        "空",
    }
)


class GlinerModelError(RuntimeError):
    """Raised when the local GLiNER model cannot be loaded or run."""


def build_gliner_input_text(
    title: str,
    description: str = "",
    body_text: str = "",
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Join title/description/body into one bounded NER input string."""

    parts = [str(part or "") for part in (title, description, body_text)]
    collapsed = " ".join(" ".join(part.split()) for part in parts if part.strip())
    limit = max(1, int(max_chars))
    return collapsed[:limit]


def normalize_gliner_entities(
    raw_entities: Iterable[Any],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    max_entities: int = _DEFAULT_MAX_ENTITIES,
) -> list[dict[str, Any]]:
    """Filter, dedupe and rank one text's raw GLiNER entities.

    Keeps compact ``{"text", "label", "score"}`` rows with score at or above
    ``threshold``. Duplicate ``(label, text)`` pairs keep their highest score;
    output is score-descending then lexicographic for stability.
    """

    try:
        floor = min(1.0, max(0.0, float(threshold)))
    except (TypeError, ValueError):
        floor = _DEFAULT_THRESHOLD
    cap = max(1, int(max_entities))
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        raw_text = raw.get("text")
        if not isinstance(raw_text, str):
            continue
        label = raw.get("label")
        if not isinstance(label, str):
            continue
        text = " ".join(raw_text.split())
        label = label.strip()
        if not text or len(text) > _ENTITY_TEXT_MAX_CHARS:
            continue
        if text.lower() in _ENTITY_PLACEHOLDERS or text in _ENTITY_PLACEHOLDERS:
            continue
        if not label:
            continue
        try:
            raw_score = raw.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                continue
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score != score or score < floor:  # NaN-safe threshold gate
            continue
        score = min(1.0, max(0.0, score))
        key = (label, text)
        existing = best.get(key)
        if existing is None or score > float(existing["score"]):
            best[key] = {
                "text": text,
                "label": label,
                "score": round(score, 4),
            }
    ranked = sorted(
        best.values(),
        key=lambda row: (-float(row["score"]), str(row["label"]), str(row["text"])),
    )
    return ranked[:cap]


def entities_payload_to_json(entities: Sequence[dict[str, Any]]) -> str:
    """Serialize normalized entities to the stored JSON payload ("" when empty)."""

    if not entities:
        # Distinguish "tagged, nothing found" from "never tagged": an empty
        # list still persists so claim retries don't re-run inference.
        return "[]"
    return json.dumps(list(entities), ensure_ascii=False, separators=(",", ":"))


def cjk_ratio(text: str) -> float:
    """Share of CJK (han / kana / hangul) characters among all non-space chars."""

    chars = [ch for ch in str(text or "") if not ch.isspace()]
    if not chars:
        return 0.0
    cjk = sum(
        1
        for ch in chars
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\u3040" <= ch <= "\u30ff"
        or "\xac00" <= ch <= "\ud7af"
    )
    return cjk / len(chars)


def split_by_script(
    texts: Sequence[str],
    *,
    threshold: float = CJK_RATIO_THRESHOLD,
) -> list[bool]:
    """Per-text flag: True when the text should use the CJK word splitter."""

    return [cjk_ratio(text) >= threshold for text in texts]


class GlinerEntityTagger:
    """Lazily-loaded GLiNER wrapper producing per-text entity lists.

    ``predict_fn`` is a test seam replacing the real model call. Without it,
    the first :meth:`predict_texts` loads ``gliner.GLiNER`` inside a worker
    thread; model load and inference never block the event loop.
    """

    def __init__(
        self,
        *,
        model_id: str = GLINER_DEFAULT_MODEL_ID,
        labels: Sequence[str] = DEFAULT_GLINER_LABELS,
        threshold: float = _DEFAULT_THRESHOLD,
        max_chars: int = _DEFAULT_MAX_CHARS,
        max_entities: int = _DEFAULT_MAX_ENTITIES,
        batch_size: int = 8,
        word_splitter: str = DEFAULT_GLINER_WORD_SPLITTER,
        predict_fn: (
            Callable[[list[str], list[str]], list[list[dict[str, Any]]]]
            | Callable[[list[str], list[str]], Awaitable[list[list[dict[str, Any]]]]]
            | None
        ) = None,
    ) -> None:
        self.model_id = str(model_id or "").strip() or GLINER_DEFAULT_MODEL_ID
        cleaned_labels: list[str] = []
        seen_labels: set[str] = set()
        for label in labels:
            cleaned = str(label or "").strip()
            if cleaned and cleaned not in seen_labels:
                seen_labels.add(cleaned)
                cleaned_labels.append(cleaned)
        self.labels = tuple(cleaned_labels or DEFAULT_GLINER_LABELS)
        try:
            self.threshold = min(1.0, max(0.0, float(threshold)))
        except (TypeError, ValueError):
            self.threshold = _DEFAULT_THRESHOLD
        self.max_chars = max(1, int(max_chars))
        self.max_entities = max(1, int(max_entities))
        self.batch_size = max(1, int(batch_size))
        normalized_splitter = str(word_splitter or DEFAULT_GLINER_WORD_SPLITTER).strip().lower()
        self.word_splitter = (
            normalized_splitter
            if normalized_splitter in GLINER_SPLITTER_CHOICES
            else DEFAULT_GLINER_WORD_SPLITTER
        )
        self._predict_fn = predict_fn
        self._model: Any = None
        self._load_error = ""
        self._splitter_warned = False
        self._lock = asyncio.Lock()

    @staticmethod
    def library_available() -> bool:
        """Whether the optional ``gliner`` package is importable."""

        return find_spec("gliner") is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error:
            raise GlinerModelError(self._load_error)
        try:
            from gliner import GLiNER
        except ImportError as exc:
            self._load_error = f"gliner package unavailable: {exc}"
            logger.warning("%s", self._load_error)
            raise GlinerModelError(self._load_error) from exc
        try:
            self._model = GLiNER.from_pretrained(self.model_id, load_tokenizer=True)
        except Exception as exc:
            self._load_error = f"failed to load GLiNER model {self.model_id!r}: {exc}"
            logger.warning("%s", self._load_error)
            raise GlinerModelError(self._load_error) from exc
        return self._model

    def _set_model_splitter(self, use_cjk: bool) -> None:
        """Point the loaded model's word splitter at the requested script.

        The model ships with a whitespace splitter (matches its training). For
        CJK texts we swap in ``jieba`` so Chinese sentences become word tokens
        instead of one giant span; Latin texts keep the stock behavior.
        """

        processor = getattr(self._model, "data_processor", None)
        if processor is None or not hasattr(processor, "words_splitter"):
            if not self._splitter_warned:
                self._splitter_warned = True
                logger.debug("loaded GLiNER model exposes no words_splitter; keeping default")
            return
        current = getattr(processor, "words_splitter", None)
        current_type = type(getattr(current, "splitter", current)).__name__
        if not use_cjk:
            if current_type == "WhitespaceTokenSplitter":
                return
            from gliner.data_processing import WordsSplitter

            processor.words_splitter = WordsSplitter("whitespace")
            return
        from gliner.data_processing import WordsSplitter

        try:
            processor.words_splitter = WordsSplitter(
                self.word_splitter if self.word_splitter != "auto" else "jieba"
            )
        except Exception as exc:
            if not self._splitter_warned:
                self._splitter_warned = True
                logger.warning(
                    "CJK word splitter unavailable (%s); falling back to whitespace "
                    "(Chinese entity recall will suffer)",
                    exc,
                )
            processor.words_splitter = WordsSplitter("whitespace")

    def _run_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        model = self._load_model()
        rows: list[list[dict[str, Any]]] = [[] for _ in texts]
        groups: dict[bool, list[int]] = {False: [], True: []}
        if self.word_splitter == "auto":
            # Route per text: CJK-dominant inputs go through jieba so Chinese
            # sentences become word tokens; Latin texts keep the stock
            # whitespace behavior (multi-word entities stay coherent).
            for index, flag in enumerate(split_by_script(texts)):
                groups[flag].append(index)
        else:
            groups[self.word_splitter != "whitespace"] = list(range(len(texts)))
        for use_cjk, indices in groups.items():
            if not indices:
                continue
            if self._predict_fn is None:
                self._set_model_splitter(use_cjk)
            subset = [texts[i] for i in indices]
            raw: Any
            try:
                raw = model.batch_predict_entities(
                    subset,
                    list(self.labels),
                    threshold=self.threshold,
                    batch_size=self.batch_size,
                )
            except TypeError:
                # Older gliner releases without batching fall back to per-text
                # prediction rather than failing the whole claim.
                raw = [
                    model.predict_entities(text, list(self.labels), threshold=self.threshold)
                    for text in subset
                ]
            raw_rows = cast("list[list[dict[str, Any]]]", raw)
            for local_index, original_index in enumerate(indices):
                rows[original_index] = raw_rows[local_index]
        return rows

    async def predict_texts(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Return normalized entities for each input text, order-preserving."""

        if not texts:
            return []
        if self._predict_fn is not None:
            result = self._predict_fn(texts, list(self.labels))
            if inspect.isawaitable(result):
                result = await result
            raw_rows = list(result)
        else:
            async with self._lock:
                raw_rows = await asyncio.to_thread(self._run_batch, list(texts))
        if len(raw_rows) != len(texts):
            raise GlinerModelError(
                f"GLiNER returned {len(raw_rows)} entity lists for {len(texts)} texts"
            )
        return [
            normalize_gliner_entities(
                row,
                threshold=self.threshold,
                max_entities=self.max_entities,
            )
            for row in raw_rows
        ]

    async def tag_contents(self, contents: list[Any]) -> int:
        """Fill ``gliner_*`` fields on items missing them; return tagged count.

        Items already carrying a payload (including the ``"[]"``
        found-nothing marker) are skipped. Zero-entity texts are recorded so
        repeated claim retries do not re-run inference.
        """

        pending = [
            content
            for content in contents
            if not str(getattr(content, "gliner_entities_json", "") or "").strip()
        ]
        if not pending:
            return 0
        texts = [
            build_gliner_input_text(
                str(getattr(content, "title", "") or ""),
                str(getattr(content, "description", "") or ""),
                str(getattr(content, "body_text", "") or ""),
                max_chars=self.max_chars,
            )
            for content in pending
        ]
        predicted = await self.predict_texts(texts)
        tagged = 0
        for content, entities in zip(pending, predicted, strict=True):
            content.gliner_entities_json = entities_payload_to_json(entities)
            content.gliner_model = self.model_id
            tagged += 1
        return tagged
