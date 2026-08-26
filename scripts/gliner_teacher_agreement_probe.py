"""Live measurement probe for the local GLiNER entity-tagging stage.

Samples teacher-allowlist rows from the real candidate pool, runs
``gliner-community/gliner_large-v2.5`` locally (no LLM calls), and reports
entity yield plus agreement against the stored teacher tags
(``topic_group`` / ``franchise_key``).

Usage::

    uv run --extra dev python scripts/gliner_teacher_agreement_probe.py \\
        --config E:/otherproject/OpenBiliClaw/config.toml \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --limit 100

Read-only by default; pass ``--persist`` to write ``gliner_*`` columns
(teacher / cheap-channel columns are never touched either way). Requires the
optional ``[gliner]`` extra; model weights download on first use and honor
``HF_ENDPOINT`` (set ``https://hf-mirror.com`` behind a blocked endpoint).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_LIMIT = 100
DEFAULT_BATCH_SIZE = 8

from openbiliclaw.discovery.gliner_tagger import (  # noqa: E402
    DEFAULT_GLINER_LABELS,
    GLINER_DEFAULT_MODEL_ID,
    GlinerEntityTagger,
)
from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin sample across ``source_platform`` so thin platforms appear."""

    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return list(rows)
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        platform = str(row.get("source_platform") or "unknown").strip() or "unknown"
        by_platform[platform].append(row)
    rng = random.Random(seed)
    queues: list[list[dict[str, Any]]] = []
    for platform in sorted(by_platform):
        bucket = list(by_platform[platform])
        rng.shuffle(bucket)
        queues.append(bucket)
    picked: list[dict[str, Any]] = []
    while queues and len(picked) < limit:
        remaining: list[list[dict[str, Any]]] = []
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


def load_teacher_rows(database: Database) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, content_type, title,
               body_text, bvid, content_id, description, tags,
               topic_group, style_key, temporal_class, franchise_key,
               teacher_model
        FROM discovery_candidates
        WHERE score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY evaluated_at DESC, id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    return [dict(row) for row in cursor.fetchall()]


def _norm(value: object) -> str:
    return "".join(str(value or "").casefold().split())


def _contains_any(haystacks: list[str], needle_norm: str) -> bool:
    if not needle_norm:
        return False
    return any(needle_norm in _norm(item) for item in haystacks)


def compare_row(row: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    """One row's agreement signals between GLiNER entities and teacher tags."""

    title = str(row.get("title") or "")
    description = str(row.get("description") or "")
    topic_group = str(row.get("topic_group") or "").strip()
    franchise_key = str(row.get("franchise_key") or "").strip()
    raw_tags = row.get("tags")
    if isinstance(raw_tags, str):
        try:
            decoded = json.loads(raw_tags)
            platform_tags = [str(item) for item in decoded] if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            platform_tags = []
    elif isinstance(raw_tags, list):
        platform_tags = [str(item) for item in raw_tags]
    else:
        platform_tags = []

    topic_hit = False
    franchise_hit = False
    grounded_hits = 0
    for entity in entities:
        text_norm = _norm(entity.get("text"))
        if not text_norm:
            continue
        if topic_group and (
            _contains_any([topic_group], text_norm) or _norm(topic_group) in text_norm
        ):
            topic_hit = True
        if franchise_key and (
            _contains_any([franchise_key], text_norm) or _norm(franchise_key) in text_norm
        ):
            franchise_hit = True
        if _contains_any([title, description, *platform_tags], text_norm):
            grounded_hits += 1
    # Bidirectional coverage: a multi-entity topic may be covered jointly even
    # when no single span equals it (e.g. topic "深度学习" vs spans "深度" +
    # "学习" is rare, but "机器学习 神经网络" style joins do happen).
    if topic_group and not topic_hit:
        topic_norm = _norm(topic_group)
        joined = _norm(" ".join(str(e.get("text") or "") for e in entities))
        topic_hit = bool(topic_norm) and topic_norm in joined
    return {
        "id": int(row.get("id") or 0),
        "platform": str(row.get("source_platform") or "unknown"),
        "title": title[:60],
        "topic_group": topic_group,
        "n_entities": len(entities),
        "topic_hit": topic_hit,
        "has_franchise": bool(franchise_key),
        "franchise_hit": franchise_hit,
        "grounded_hits": grounded_hits,
        "entities": entities,
    }


def aggregate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(comparisons)
    with_entities = sum(1 for c in comparisons if c["n_entities"] > 0)
    topic_rows = [c for c in comparisons if c["topic_group"]]
    topic_hits = sum(1 for c in topic_rows if c["topic_hit"])
    franchise_rows = [c for c in comparisons if c["has_franchise"]]
    franchise_hits = sum(1 for c in franchise_rows if c["franchise_hit"])
    label_counter: Counter[str] = Counter()
    for c in comparisons:
        for e in c["entities"]:
            label_counter[str(e.get("label") or "?")] += 1
    by_platform: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "with_entities": 0, "topic_rows": 0, "topic_hits": 0}
    )
    for c in comparisons:
        bucket = by_platform[c["platform"]]
        bucket["n"] += 1
        if c["n_entities"] > 0:
            bucket["with_entities"] += 1
        if c["topic_group"]:
            bucket["topic_rows"] += 1
            if c["topic_hit"]:
                bucket["topic_hits"] += 1
    return {
        "n": n,
        "with_entities": with_entities,
        "avg_entities": (sum(c["n_entities"] for c in comparisons) / n) if n else 0.0,
        "topic_rows": len(topic_rows),
        "topic_hits": topic_hits,
        "franchise_rows": len(franchise_rows),
        "franchise_hits": franchise_hits,
        "label_distribution": dict(label_counter.most_common()),
        "by_platform": {k: dict(v) for k, v in sorted(by_platform.items())},
    }


def _rate(hits: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{hits / total:.1%} ({hits}/{total})"


def print_report(stats: dict[str, Any], elapsed: float, tagged_n: int) -> None:
    n = int(stats["n"])
    print("=== entity yield ===")
    print(f"  items with >=1 entity {_rate(int(stats['with_entities']), n)}")
    print(f"  avg entities/item     {stats['avg_entities']:.2f}")
    print(
        f"  wall time             {elapsed:.1f}s"
        f"  ({(elapsed / max(1, tagged_n) * 1000):.0f} ms/item)"
    )
    print("=== agreement vs teacher ===")
    print(
        f"  topic_group hit       {_rate(int(stats['topic_hits']), int(stats['topic_rows']))}"
        "  (entity covers the teacher topic, bidirectional containment)"
    )
    print(
        f"  franchise_key hit     "
        f"{_rate(int(stats['franchise_hits']), int(stats['franchise_rows']))}"
        "  (on rows where teacher set a franchise)"
    )
    labels = stats.get("label_distribution") or {}
    if labels:
        top = ", ".join(f"{label}×{count}" for label, count in list(labels.items())[:10])
        print(f"  label distribution    {top}")
    by_platform = stats.get("by_platform") or {}
    if by_platform:
        print("  by platform:")
        for platform, bucket in by_platform.items():
            print(
                f"    {platform:12s} n={bucket['n']:3d}"
                f"  entities={_rate(int(bucket['with_entities']), int(bucket['n']))}"
                f"  topic={_rate(int(bucket['topic_hits']), int(bucket['topic_rows']))}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-id", default=GLINER_DEFAULT_MODEL_ID)
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels; empty keeps the 8 domain defaults.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write gliner_* columns for the sampled rows (default read-only).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the full per-item comparison dump.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace, database: Database) -> int:
    pool = load_teacher_rows(database)
    sample = stratified_sample(pool, limit=max(1, int(args.limit)), seed=int(args.seed))
    print("=== GLiNER teacher-agreement probe ===")
    print(f"  db                    {args.db}")
    print(f"  model                 {args.model_id}")
    print(f"  threshold/max_chars   {args.threshold} / {args.max_chars}")
    print(f"  teacher allowlist     {len(pool)}")
    print(f"  sample                {len(sample)} (limit={args.limit} seed={args.seed})")
    platform_counts = Counter(str(row.get("source_platform") or "?") for row in sample)
    print(f"  sample platforms      {dict(platform_counts)}")
    if not sample:
        print("  no teacher-labeled rows to sample")
        return 2

    labels = [item.strip() for item in str(args.labels).split(",") if item.strip()] or list(
        DEFAULT_GLINER_LABELS
    )
    print(f"  labels                {labels}")

    tagger = GlinerEntityTagger(
        model_id=str(args.model_id),
        labels=labels,
        threshold=float(args.threshold),
        max_chars=int(args.max_chars),
        batch_size=max(1, int(args.batch_size)),
    )
    contents = [
        SimpleNamespace(
            title=str(row.get("title") or ""),
            description=str(row.get("description") or ""),
            body_text=str(row.get("body_text") or ""),
            gliner_entities_json="",
            gliner_model="",
        )
        for row in sample
    ]
    print("=== running local inference (no LLM calls) ===")
    started = time.monotonic()
    tagged_n = await tagger.tag_contents(contents)
    elapsed = time.monotonic() - started

    comparisons: list[dict[str, Any]] = []
    persist_rows: list[dict[str, Any]] = []
    for row, content in zip(sample, contents, strict=True):
        payload = str(getattr(content, "gliner_entities_json", "") or "")
        try:
            entities = json.loads(payload) if payload else []
        except json.JSONDecodeError:
            entities = []
        comparisons.append(compare_row(row, entities))
        if args.persist and payload:
            persist_rows.append(
                {
                    "candidate_id": int(row.get("id") or 0),
                    "gliner_entities_json": payload,
                    "gliner_model": str(getattr(content, "gliner_model", "") or ""),
                }
            )

    written = 0
    if args.persist and persist_rows:
        written = database.update_discovery_candidate_gliner_tags(persist_rows)

    stats = aggregate(comparisons)
    print_report(stats, elapsed, tagged_n)
    print("=== persistence ===")
    print(
        f"  persisted             {written}"
        f"{' (--persist)' if args.persist else '  (read-only run; pass --persist to write)'}"
    )

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(
                {"stats": stats, "items": comparisons},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  json dump             {args.json_out}")
    return 0 if tagged_n else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    config_path = args.config.expanduser().resolve()
    db_path = args.db.expanduser().resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}")
        return 1
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 1
    os.environ.setdefault("OPENBILICLAW_PROJECT_ROOT", str(LIVE_ROOT.resolve()))

    database = Database(db_path)
    database.initialize()
    try:
        return asyncio.run(_run(args, database))
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
