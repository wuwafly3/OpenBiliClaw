"""GLiNER local entity tagging: pure helpers, tagger, engine, DB, pipeline."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from openbiliclaw.discovery.candidate_pipeline import DiscoveryCandidatePipeline
from openbiliclaw.discovery.candidate_pool import (
    discovered_content_to_candidate_write,
    row_to_discovered_content,
)
from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent
from openbiliclaw.discovery.gliner_tagger import (
    DEFAULT_GLINER_LABELS,
    GLINER_DEFAULT_MODEL_ID,
    GlinerEntityTagger,
    GlinerModelError,
    build_gliner_input_text,
    entities_payload_to_json,
    normalize_gliner_entities,
)
from openbiliclaw.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path


def _make_item(**overrides: Any) -> DiscoveredContent:
    defaults: dict[str, Any] = {
        "title": "原神3.0前瞻直播总结",
        "description": "新地图须弥城实机演示",
        "source_platform": "bilibili",
        "source_strategy": "search",
        "content_id": "BV1xx411c7mD",
        "bvid": "BV1xx411c7mD",
    }
    defaults.update(overrides)
    return DiscoveredContent(**defaults)


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_build_gliner_input_text_joins_and_collapses() -> None:
    text = build_gliner_input_text("  原神  前瞻 \n 直播 ", " 须弥\n实机 ", "")
    assert text == "原神 前瞻 直播 须弥 实机"


def test_build_gliner_input_text_truncates_to_max_chars() -> None:
    text = build_gliner_input_text("a" * 40, max_chars=16)
    assert len(text) == 16
    assert text == "a" * 16


def test_normalize_gliner_entities_filters_threshold_and_placeholders() -> None:
    raw = [
        {"text": "原神", "label": "游戏", "score": 0.91},
        {"text": "原神", "label": "游戏", "score": 0.55},  # dedupe keeps highest
        {"text": "-", "label": "游戏", "score": 0.9},  # placeholder span
        {"text": "", "label": "游戏", "score": 0.9},  # empty span
        {"text": "米哈游", "label": "组织", "score": 0.35},  # below threshold
        {"text": "无", "label": "人物", "score": 0.8},  # placeholder token
        {"text": 42, "label": "游戏", "score": 0.9},  # non-string text
        {"text": "散兵", "label": ""},  # missing score/label
        "not-a-dict",
    ]
    rows = normalize_gliner_entities(raw, threshold=0.5)
    assert rows == [{"text": "原神", "label": "游戏", "score": 0.91}]


def test_normalize_gliner_entities_sorts_and_caps() -> None:
    raw = [{"text": f"实体{i}", "label": "游戏", "score": 1 - i / 10} for i in range(6)]
    rows = normalize_gliner_entities(raw, threshold=0.1, max_entities=3)
    assert [row["text"] for row in rows] == ["实体0", "实体1", "实体2"]


def test_entities_payload_to_json_round_trip() -> None:
    assert entities_payload_to_json([]) == "[]"
    payload = entities_payload_to_json([{"text": "原神", "label": "游戏", "score": 0.91}])
    assert payload == '[{"text":"原神","label":"游戏","score":0.91}]'
    assert json.loads(payload)[0]["label"] == "游戏"


# ── GlinerEntityTagger ──────────────────────────────────────────────────────


def test_tagger_constructor_cleans_labels() -> None:
    tagger = GlinerEntityTagger(labels=[" 游戏 ", "", "游戏", "动漫"])
    assert tagger.labels == ("游戏", "动漫")
    assert GlinerEntityTagger(labels=[]).labels == DEFAULT_GLINER_LABELS
    assert GlinerEntityTagger(model_id=" ").model_id == GLINER_DEFAULT_MODEL_ID


@pytest.mark.asyncio
async def test_tag_contents_with_sync_predict_seam() -> None:
    calls: list[list[str]] = []

    def predict_fn(texts: list[str], labels: list[str]) -> list[list[dict[str, Any]]]:
        calls.append(list(texts))
        assert "游戏" in labels
        return [
            [{"text": "原神", "label": "游戏", "score": 0.9}],
            [],
        ]

    tagger = GlinerEntityTagger(predict_fn=predict_fn)
    first, second = _make_item(), _make_item(content_id="BV2", bvid="BV2")
    tagged = await tagger.tag_contents([first, second])

    assert tagged == 2
    assert json.loads(first.gliner_entities_json) == [
        {"text": "原神", "label": "游戏", "score": 0.9}
    ]
    # Zero-entity texts persist the found-nothing marker so claim retries
    # never re-run inference.
    assert second.gliner_entities_json == "[]"
    assert first.gliner_model == GLINER_DEFAULT_MODEL_ID
    assert len(calls) == 1

    # Already-tagged items are skipped entirely.
    third = _make_item(gliner_entities_json="[]")
    assert await tagger.tag_contents([first, third]) == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tag_contents_accepts_async_predict_seam() -> None:
    async def predict_fn(texts: list[str], labels: list[str]) -> list[list[dict[str, Any]]]:
        return [[] for _ in texts]

    tagger = GlinerEntityTagger(predict_fn=predict_fn)
    item = _make_item()
    assert await tagger.tag_contents([item]) == 1
    assert item.gliner_entities_json == "[]"


@pytest.mark.asyncio
async def test_predict_texts_rejects_length_mismatch() -> None:
    def predict_fn(texts: list[str], labels: list[str]) -> list[list[dict[str, Any]]]:
        return []

    tagger = GlinerEntityTagger(predict_fn=predict_fn)
    with pytest.raises(GlinerModelError):
        await tagger.predict_texts(["one", "two"])


# ── Engine integration ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_disabled_makes_zero_calls() -> None:
    engine = ContentDiscoveryEngine(gliner_tag_enabled=False)
    item = _make_item()
    assert await engine.apply_gliner_entity_tags([item]) == 0
    assert item.gliner_entities_json == ""
    assert engine._gliner_tagger is None


@pytest.mark.asyncio
async def test_engine_without_library_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GlinerEntityTagger, "library_available", staticmethod(lambda: False))
    engine = ContentDiscoveryEngine(gliner_tag_enabled=True)
    item = _make_item()

    assert await engine.apply_gliner_entity_tags([item]) == 0
    assert item.gliner_entities_json == ""
    assert engine._gliner_unavailable_logged is True
    # Second call stays quiet and still fails open.
    assert await engine.apply_gliner_entity_tags([item]) == 0
    assert engine._gliner_tagger is None


@pytest.mark.asyncio
async def test_engine_applies_injected_tagger(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real package is optional; pretend it is present for lazy construction.
    monkeypatch.setattr(GlinerEntityTagger, "library_available", staticmethod(lambda: True))
    engine = ContentDiscoveryEngine(gliner_tag_enabled=True)
    engine._gliner_tagger = GlinerEntityTagger(
        labels=["游戏"],
        predict_fn=lambda texts, labels: [
            [{"text": "原神", "label": "游戏", "score": 0.9}] for _ in texts
        ],
    )
    item = _make_item()
    assert await engine.apply_gliner_entity_tags([item]) == 1
    assert json.loads(item.gliner_entities_json)[0]["text"] == "原神"
    # Engine settings flow into the lazily built tagger.
    configured = ContentDiscoveryEngine(
        gliner_tag_enabled=True,
        gliner_model_id="custom/model",
        gliner_labels=["游戏"],
        gliner_threshold=0.7,
        gliner_max_chars=256,
    )
    tagger = configured._get_gliner_tagger()
    assert tagger is not None
    assert tagger.model_id == "custom/model"
    assert tagger.labels == ("游戏",)
    assert tagger.threshold == 0.7
    assert tagger.max_chars == 256


# ── Database round trip ─────────────────────────────────────────────────────


def test_database_persists_gliner_tags(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.initialize()
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(discovery_candidates)")}
    assert {"gliner_entities_json", "gliner_model", "gliner_at"} <= columns

    item = _make_item()
    inserted = db.enqueue_discovery_candidates([discovered_content_to_candidate_write(item)])
    assert inserted == 1
    claimed = db.claim_discovery_candidates_for_eval(limit=5, claim_token="tok")
    candidate_id = int(claimed[0]["id"])

    updated = db.update_discovery_candidate_gliner_tags(
        [
            {
                "candidate_id": candidate_id,
                "gliner_entities_json": '[{"text":"原神","label":"游戏","score":0.9}]',
                "gliner_model": GLINER_DEFAULT_MODEL_ID,
            }
        ]
    )
    assert updated == 1
    # Empty payloads are rejected so never-tagged rows stay distinguishable.
    assert db.update_discovery_candidate_gliner_tags([{"candidate_id": candidate_id}]) == 0

    stored = db.get_discovery_candidates_by_ids([candidate_id])[0]
    restored = row_to_discovered_content(dict(stored))
    assert restored.gliner_model == GLINER_DEFAULT_MODEL_ID
    assert json.loads(restored.gliner_entities_json)[0]["text"] == "原神"
    raw_at = (
        sqlite3.connect(tmp_path / "test.db")
        .execute("SELECT gliner_at FROM discovery_candidates WHERE id = ?", (candidate_id,))
        .fetchone()[0]
    )
    assert raw_at  # timestamp stamped even for future empty-list updates


# ── Pipeline fail-open integration ──────────────────────────────────────────


class _GlinerCapableEngine:
    """Minimal engine exposing only the GLiNER seam the pipeline touches."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.tagged: list[DiscoveredContent] = []

    async def apply_gliner_entity_tags(self, contents: list[DiscoveredContent]) -> int:
        if self.fail:
            raise RuntimeError("model exploded")
        self.tagged.extend(contents)
        for content in contents:
            content.gliner_entities_json = '[{"text":"原神","label":"游戏","score":0.9}]'
            content.gliner_model = GLINER_DEFAULT_MODEL_ID
        return len(contents)


def _enqueue_one(db: Database, **overrides: Any) -> None:
    db.enqueue_discovery_candidates(
        [discovered_content_to_candidate_write(_make_item(**overrides))]
    )


@pytest.mark.asyncio
async def test_pipeline_persists_gliner_rows(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.initialize()
    _enqueue_one(db)
    engine = _GlinerCapableEngine()
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    claim = pipeline.claim_batch(limit=5)
    assert claim is not None

    await pipeline._maybe_gliner_tag_claim(claim)

    assert len(engine.tagged) == 1
    stored = db.get_discovery_candidates_by_ids([int(claim.rows[0]["id"])])[0]
    assert "原神" in str(stored["gliner_entities_json"])
    assert stored["gliner_model"] == GLINER_DEFAULT_MODEL_ID


@pytest.mark.asyncio
async def test_pipeline_gliner_fail_open_keeps_eval_running(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.initialize()
    _enqueue_one(db)
    engine = _GlinerCapableEngine(fail=True)
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    claim = pipeline.claim_batch(limit=5)
    assert claim is not None

    await pipeline._maybe_gliner_tag_claim(claim)

    stored = db.get_discovery_candidates_by_ids([int(claim.rows[0]["id"])])[0]
    assert str(stored["gliner_entities_json"] or "") == ""


@pytest.mark.asyncio
async def test_pipeline_skips_when_capability_absent(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.initialize()
    _enqueue_one(db)
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=object())  # type: ignore[arg-type]
    claim = pipeline.claim_batch(limit=5)
    assert claim is not None

    await pipeline._maybe_gliner_tag_claim(claim)

    stored = db.get_discovery_candidates_by_ids([int(claim.rows[0]["id"])])[0]
    assert str(stored["gliner_entities_json"] or "") == ""


# ── Word-splitter routing (Chinese support) ─────────────────────────────────


def test_cjk_ratio_and_script_split() -> None:
    from openbiliclaw.discovery.gliner_tagger import cjk_ratio, split_by_script

    assert cjk_ratio("") == 0.0
    assert cjk_ratio("hello world") == 0.0
    assert cjk_ratio("战双帕弥什cos跳舞") == pytest.approx(0.7)
    assert cjk_ratio("战双帕弥什") == 1.0
    flags = split_by_script(
        [
            "战双帕弥什3.0版本直播汇总",
            "The Lore of Punishing: Gray Raven Chapter 22",
            "H100, H200, B200 — GPU vendors",
            "【原神】5.0前瞻特别节目",
        ]
    )
    assert flags == [True, False, False, True]


def test_tagger_normalizes_word_splitter() -> None:
    from openbiliclaw.discovery.gliner_tagger import DEFAULT_GLINER_WORD_SPLITTER

    assert GlinerEntityTagger().word_splitter == "auto"
    assert GlinerEntityTagger(word_splitter=" JIEBA ").word_splitter == "jieba"
    assert (
        GlinerEntityTagger(word_splitter="nonsense").word_splitter == DEFAULT_GLINER_WORD_SPLITTER
    )


def test_engine_normalizes_gliner_word_splitter() -> None:
    engine = ContentDiscoveryEngine(gliner_word_splitter="hanlp")
    assert engine.gliner_word_splitter == "hanlp"
    fallback = ContentDiscoveryEngine(gliner_word_splitter="bogus")
    assert fallback.gliner_word_splitter == "auto"
