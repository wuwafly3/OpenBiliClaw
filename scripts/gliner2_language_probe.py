"""Language-capability probe for the GLiNER2.5 checkpoints.

Compares Chinese vs English zero-shot capability of ``gliner2`` boundary
checkpoints on real teacher-labeled candidates:

* classification — ``classify_text`` over the closed style/temporal vocab,
  scored by exact match against stored teacher tags (majority baselines
  reported alongside);
* entity extraction — yield, self-grounding rate (span text occurs in the
  source) and mean confidence as quality proxies (no gold spans exist).

Texts route to the char splitter when CJK-dominant and whitespace otherwise,
matching the production routing rule. Read-only; no LLM calls.

Usage::

    uv run --extra dev python scripts/gliner2_language_probe.py \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db --limit 400
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_LIMIT = 400
DEFAULT_SEED = 2026
CJK_THRESHOLD = 0.2
MULTI_MODEL = "fastino/gliner2.5-multi-v1"
ENGLISH_MODEL = "fastino/gliner2.5-base-v1"

from openbiliclaw.discovery.gliner_tagger import cjk_ratio  # noqa: E402
from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.discovery.style_keys import (  # noqa: E402
    STYLE_KEY_DEFINITIONS,
    normalize_style_key,
)
from openbiliclaw.discovery.temporal import normalize_temporal_class  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402

STYLE_KEYS = [key for key, _ in STYLE_KEY_DEFINITIONS]
STYLE_DESC = {k: f"{k}: {d}" for k, d in STYLE_KEY_DEFINITIONS}
TEMPORAL_DESC = {
    "breaking": "breaking: 刚发生/首发的突发内容，时效以小时计",
    "current": "current: 近期热点或当下流行，时效以天到周计",
    "evergreen": "evergreen: 长期有效的教程/科普/作品内容，价值不依赖时间",
    "historical": "historical: 对已闭合事件或过去年代的回顾、考据、档案",
    "versioned": "versioned: 指涉可识别且仍在迭代的具体对象，更新后价值衰减",
}


def load_teacher_rows(database: Database) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, title, description, tags,
               topic_group, style_key, temporal_class, franchise_key
        FROM discovery_candidates
        WHERE score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY evaluated_at DESC, id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    return [dict(row) for row in cursor.fetchall()]


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin across source_platform, mirroring the other GLiNER probes."""

    import random

    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return list(rows)
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        platform = str(row.get("source_platform") or "unknown").strip() or "unknown"
        by_platform[platform].append(row)
    rng = random.Random(seed)
    queues = []
    for platform in sorted(by_platform):
        bucket = list(by_platform[platform])
        rng.shuffle(bucket)
        queues.append(bucket)
    picked: list[dict[str, Any]] = []
    while queues and len(picked) < limit:
        remaining = []
        for bucket in queues:
            if not bucket:
                continue
            picked.append(bucket.pop())
            if bucket:
                remaining.append(bucket)
            if len(picked) >= limit:
                break
        queues = remaining
    return picked


def input_text(row: dict[str, Any], max_chars: int = 400) -> str:
    return " ".join(f"{row.get('title', '')} {row.get('description', '')}".split())[:max_chars]


def classify_split(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    use_descriptions: bool,
) -> dict[str, dict[str, float]]:
    """Classify every row (script-routed splitter); return per-lang exact rates."""

    schema_labels = (
        {"style": STYLE_DESC, "temporal": TEMPORAL_DESC}
        if use_descriptions
        else {"style": dict.fromkeys(STYLE_KEYS), "temporal": dict.fromkeys(TEMPORAL_DESC)}
    )
    buckets: dict[bool, list[int]] = {False: [], True: []}
    for index, row in enumerate(rows):
        flag = cjk_ratio(input_text(row)) >= CJK_THRESHOLD
        buckets[flag].append(index)

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for use_char, indices in buckets.items():
        if not indices:
            continue
        model.set_word_splitter("char" if use_char else "whitespace")
        schema = (
            model.create_schema()
            .classification("style", schema_labels["style"], cls_threshold=0.0)
            .classification("temporal", schema_labels["temporal"], cls_threshold=0.0)
        )
        for i in indices:
            row = rows[i]
            lang = "zh" if use_char else "en"
            gold_s = normalize_style_key(row.get("style_key"))
            gold_t = normalize_temporal_class(row.get("temporal_class"))
            result = model.extract(input_text(row), schema) or {}
            pred_s = str(result.get("style") or "")
            pred_t = str(result.get("temporal") or "")
            if gold_s:
                stats[lang]["s_tot"] += 1
                stats[lang]["s_hit"] += int(pred_s == gold_s)
            if gold_t and gold_t != "unknown":
                stats[lang]["t_tot"] += 1
                stats[lang]["t_hit"] += int(pred_t == gold_t)
    return {
        lang: {
            "n": len(rows),
            "style_exact": stats[lang]["s_hit"],
            "style_tot": stats[lang]["s_tot"],
            "temporal_exact": stats[lang]["t_hit"],
            "temporal_tot": stats[lang]["t_tot"],
        }
        for lang in ("zh", "en")
    }


def ner_split(model: Any, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Entity extraction quality proxies per language (no gold spans)."""

    labels = ["游戏", "动漫", "影视", "音乐", "人物", "组织", "产品", "地点"]
    buckets: dict[bool, list[int]] = {False: [], True: []}
    for index, row in enumerate(rows):
        flag = cjk_ratio(input_text(row)) >= CJK_THRESHOLD
        buckets[flag].append(index)

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_sum: dict[str, float] = defaultdict(float)
    for use_char, indices in buckets.items():
        if not indices:
            continue
        model.set_word_splitter("char" if use_char else "whitespace")
        for i in indices:
            row = rows[i]
            text = input_text(row)
            lang = "zh" if use_char else "en"
            try:
                result = model.extract_entities(text, labels, include_confidence=True)
            except Exception:
                continue
            entities = (result or {}).get("entities") or {}
            spans = [e for group in entities.values() for e in group]
            stats[lang]["n"] += 1
            if spans:
                stats[lang]["with_entities"] += 1
            stats[lang]["total"] += len(spans)
            for span in spans:
                span_text = str(span.get("text") or "").strip()
                confidence_sum[lang] += float(span.get("confidence") or 0.0)
                if span_text and span_text.casefold() in text.casefold():
                    stats[lang]["grounded"] += 1
    out: dict[str, dict[str, float]] = {}
    for lang in ("zh", "en"):
        total = int(stats[lang]["total"])
        out[lang] = {
            "rows": stats[lang]["n"],
            "with_entities": stats[lang]["with_entities"],
            "spans": total,
            "grounded": stats[lang]["grounded"],
            "mean_conf": (confidence_sum[lang] / total) if total else 0.0,
        }
    return out


def _pct(hits: int, total: int) -> str:
    return f"{hits / total:.1%}" if total else "n/a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--models",
        default=MULTI_MODEL,
        help=f"Comma-separated gliner2 checkpoints (default {MULTI_MODEL}; "
        f"add {ENGLISH_MODEL} for the English-only reference)",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)

    db_path = args.db.expanduser().resolve()
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 1
    database = Database(db_path)
    database.initialize()
    try:
        pool = load_teacher_rows(database)
        sample = stratified_sample(pool, limit=max(1, int(args.limit)), seed=int(args.seed))
        zh_n = sum(1 for r in sample if cjk_ratio(input_text(r)) >= CJK_THRESHOLD)
        print("=== GLiNER2.5 language-capability probe ===")
        print(f"  db              {db_path}")
        print(f"  sample          {len(sample)} (limit={args.limit} seed={args.seed})")
        print(f"  split           zh={zh_n} en={len(sample) - zh_n} (CJK>= {CJK_THRESHOLD})")
        if not sample:
            return 2

        for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
            from gliner2 import AutoExtractor

            t0 = time.monotonic()
            model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
            load_s = time.monotonic() - t0
            print(f"\n── {model_id}  (load {load_s:.0f}s)")

            for use_desc, tag in ((False, "codenames"), (True, "codenames+desc")):
                t0 = time.monotonic()
                rates = classify_split(model, sample, use_descriptions=use_desc)
                el = time.monotonic() - t0
                print(f"  classification [{tag}]  ({el / max(1, len(sample)) * 1000:.0f} ms/item)")
                for lang in ("zh", "en"):
                    s = rates[lang]
                    print(
                        f"    {lang}  style {_pct(int(s['style_exact']), int(s['style_tot']))}"
                        f"  temporal {_pct(int(s['temporal_exact']), int(s['temporal_tot']))}"
                        f"  (n={int(s['style_tot'])}/{int(s['temporal_tot'])})"
                    )

            t0 = time.monotonic()
            ner = ner_split(model, sample)
            el = time.monotonic() - t0
            print(f"  entity extraction  ({el / max(1, len(sample)) * 1000:.0f} ms/item)")
            for lang in ("zh", "en"):
                b = ner[lang]
                rows_n = int(b["rows"])
                spans = int(b["spans"])
                print(
                    f"    {lang}  yield {_pct(int(b['with_entities']), rows_n)}"
                    f"  spans/item {spans / rows_n:.2f}"
                    f"  grounded {_pct(int(b['grounded']), spans)}"
                    f"  conf {float(b['mean_conf']):.2f}"
                    f"  (rows={rows_n})"
                )
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
