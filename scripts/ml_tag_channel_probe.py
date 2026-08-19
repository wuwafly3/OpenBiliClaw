"""Live measurement probe for the Wave 1 cheap tags-only LLM channel.

Selects teacher-allowlist rows missing ``tag_channel_source``, calls
``discovery.tag_batch``, persists only ``tag_channel_*``, and prints unit
cost versus recent ``discovery.evaluate_batch`` plus teacher agreement.

Usage::

    uv run --extra dev python scripts/ml_tag_channel_probe.py \\
        --config E:/otherproject/OpenBiliClaw/config.toml \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --limit 90

Does not change ``[discovery].tag_channel_mode``. Agreement is diagnostic
for S1.8, not a merge gate.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_LIMIT = 90
DEFAULT_BATCH_SIZE = 45
DEFAULT_MAX_TOKENS = 1024
EVAL_CALLER = "discovery.evaluate_batch"
TAG_CALLER = "discovery.tag_batch"
# Same default as ContentDiscoveryEngine text eval; used only to turn
# per-call eval usage into an approximate per-candidate baseline.
EVAL_CANDIDATES_PER_CALL_ESTIMATE = 45

from openbiliclaw.config import (  # noqa: E402
    llm_concurrency_from_config,
    load_config,
)
from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent  # noqa: E402
from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.discovery.style_keys import normalize_style_key  # noqa: E402
from openbiliclaw.discovery.tag_channel import TAG_CHANNEL_SOURCE_LLM  # noqa: E402
from openbiliclaw.discovery.temporal import normalize_temporal_class  # noqa: E402
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService, module_overrides_from_config  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.memory.manager import MemoryManager  # noqa: E402
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


def agreement_stats(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
) -> dict[str, Any]:
    """Exact-match rates of cheap tags against teacher tags."""

    n = len(pairs)
    style_hits = 0
    temporal_hits = 0
    topic_hits = 0
    tagged = 0
    by_platform: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "style": 0, "temporal": 0, "topic": 0}
    )
    for teacher, cheap in pairs:
        if cheap.get("source") != TAG_CHANNEL_SOURCE_LLM:
            continue
        tagged += 1
        platform = teacher.get("platform") or "unknown"
        bucket = by_platform[platform]
        bucket["n"] += 1
        if cheap.get("style_key") == teacher.get("style_key"):
            style_hits += 1
            bucket["style"] += 1
        if cheap.get("temporal_class") == teacher.get("temporal_class"):
            temporal_hits += 1
            bucket["temporal"] += 1
        if cheap.get("topic_group") == teacher.get("topic_group"):
            topic_hits += 1
            bucket["topic"] += 1
    return {
        "compared": tagged,
        "requested": n,
        "style_exact": style_hits,
        "temporal_exact": temporal_hits,
        "topic_exact": topic_hits,
        "by_platform": dict(by_platform),
    }


def _rate(hits: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{hits / total:.3f} ({hits}/{total})"


def load_untagged_teacher_rows(database: Database) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, content_type, title,
               body_text, bvid, content_id, description, published_at, duration,
               topic_group, style_key, temporal_class, teacher_model,
               tag_channel_source
        FROM discovery_candidates
        WHERE score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
          AND COALESCE(tag_channel_source, '') = ''
        ORDER BY evaluated_at DESC, id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    return [dict(row) for row in cursor.fetchall()]


def row_to_content(row: dict[str, Any]) -> DiscoveredContent:
    return DiscoveredContent(
        bvid=str(row.get("bvid") or row.get("content_id") or ""),
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        body_text=str(row.get("body_text") or ""),
        published_at=str(row.get("published_at") or ""),
        duration=int(row.get("duration") or 0),
        source_platform=str(row.get("source_platform") or "bilibili"),
        content_type=str(row.get("content_type") or "video"),
        content_id=str(row.get("content_id") or row.get("bvid") or ""),
        source_strategy=str(row.get("source_strategy") or ""),
    )


def eval_baseline(database: Database, *, days: int = 7) -> dict[str, float | int]:
    cursor = database.conn.execute(
        """
        SELECT COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(estimated_cost_cny), 0) AS cost_cny
        FROM llm_usage
        WHERE caller = ?
          AND timestamp >= datetime('now', '-' || ? || ' day', 'localtime')
        """,
        (EVAL_CALLER, max(1, int(days))),
    )
    row = cursor.fetchone()
    if row is None:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_input_tokens": 0,
            "cost_cny": 0.0,
        }
    return {
        "calls": int(row["calls"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "cached_input_tokens": int(row["cached_input_tokens"] or 0),
        "cost_cny": float(row["cost_cny"] or 0.0),
    }


def _route_label(config: object) -> str:
    overrides = module_overrides_from_config(config)
    chosen = overrides.get("evaluation") or overrides.get("discovery")
    default_provider = str(getattr(getattr(config, "llm", None), "default_provider", "") or "")
    if chosen is None:
        return f"default_provider={default_provider} (no evaluation/discovery override)"
    if chosen.custom_chain:
        chain = ",".join(chosen.chain) or "(empty)"
        return f"bucket=evaluation instance_chain={chain}"
    provider = str(chosen.provider or default_provider)
    model = str(chosen.model or "")
    return f"bucket=evaluation provider={provider} model={model}"


def _print_agreement(stats: dict[str, Any]) -> None:
    compared = int(stats["compared"])
    print(f"  compared (tagged)     {compared} / requested {stats['requested']}")
    print(f"  style exact           {_rate(int(stats['style_exact']), compared)}")
    print(f"  temporal exact        {_rate(int(stats['temporal_exact']), compared)}")
    print(f"  topic exact           {_rate(int(stats['topic_exact']), compared)}")
    print("  note: topic_group is open vocabulary; exact match understates agreement")
    by_platform = stats.get("by_platform") or {}
    if by_platform:
        print("  by platform:")
        for platform in sorted(by_platform):
            bucket = by_platform[platform]
            n = int(bucket["n"])
            print(
                f"    {platform:12s} n={n:3d}  style={_rate(int(bucket['style']), n)}"
                f"  temporal={_rate(int(bucket['temporal']), n)}"
                f"  topic={_rate(int(bucket['topic']), n)}"
            )


def _usage_for_caller(snapshot: dict[str, Any], caller: str) -> dict[str, Any]:
    for row in snapshot.get("by_caller") or []:
        if str(row.get("caller") or "") == caller:
            return dict(row)
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_input_tokens": 0,
        "cost_cny": 0.0,
    }


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        nxt = current.__cause__ or current.__context__
        current = None if nxt is current else nxt
    return chain


def _exception_text(exc: BaseException) -> str:
    return " ".join(str(item) for item in _exception_chain(exc)).lower()


def _is_rate_limit(exc: BaseException) -> bool:
    from openbiliclaw.llm.base import LLMRateLimitError

    if any(isinstance(item, LLMRateLimitError) for item in _exception_chain(exc)):
        return True
    text = _exception_text(exc)
    return "rate limit" in text or "too many requests" in text or " 429" in text


def _is_hard_quota(exc: BaseException) -> bool:
    text = _exception_text(exc)
    markers = (
        "monthly usage limit",
        "allocated quota exceeded",
        "insufficient_quota",
        "insufficient balance",
        "creditserror",
        "region_blocked",
    )
    return any(marker in text for marker in markers)


def _pair_and_persist_rows(
    rows: list[dict[str, Any]],
    tagged: list[DiscoveredContent],
) -> tuple[list[tuple[dict[str, str], dict[str, str]]], list[dict[str, Any]]]:
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    persist_rows: list[dict[str, Any]] = []
    for row, content in zip(rows, tagged, strict=True):
        teacher = {
            "platform": str(row.get("source_platform") or "unknown"),
            "topic_group": " ".join(str(row.get("topic_group") or "").split()),
            "style_key": normalize_style_key(row.get("style_key")),
            "temporal_class": normalize_temporal_class(row.get("temporal_class")),
        }
        cheap = {
            "source": str(content.tag_channel_source or ""),
            "topic_group": str(content.tag_channel_topic_group or ""),
            "style_key": str(content.tag_channel_style_key or ""),
            "temporal_class": str(content.tag_channel_temporal_class or "unknown"),
        }
        pairs.append((teacher, cheap))
        if cheap["source"] != TAG_CHANNEL_SOURCE_LLM:
            continue
        persist_rows.append(
            {
                "candidate_id": int(row["id"]),
                "tag_channel_topic_group": content.tag_channel_topic_group,
                "tag_channel_style_key": content.tag_channel_style_key,
                "tag_channel_temporal_class": content.tag_channel_temporal_class,
                "tag_channel_source": TAG_CHANNEL_SOURCE_LLM,
                "tag_channel_model": content.tag_channel_model,
            }
        )
    return pairs, persist_rows


async def _tag_chunks(
    engine: ContentDiscoveryEngine,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_tokens: int,
    chunk_sleep: float,
    database: Database,
    persist: bool,
) -> tuple[list[DiscoveredContent], str, int]:
    """Tag in outer chunks so a later 429 still keeps earlier results."""

    tagged_all: list[DiscoveredContent] = []
    stop_reason = ""
    written = 0
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        contents = [row_to_content(row) for row in chunk_rows]
        delay = 15.0
        chunk_tagged: list[DiscoveredContent] | None = None
        failed = False
        for attempt in range(5):
            try:
                chunk_tagged = await engine.tag_content_batch(
                    contents,
                    batch_size=len(contents),
                    max_tokens=max_tokens,
                )
                break
            except Exception as exc:
                if _is_hard_quota(exc):
                    stop_reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  hard quota/auth at items {start + 1}-{start + len(chunk_rows)}; "
                        "keeping earlier chunks"
                    )
                    print(f"    {stop_reason}")
                    failed = True
                    chunk_tagged = contents
                    break
                if _is_rate_limit(exc) and attempt < 4:
                    print(
                        f"  rate-limited items {start + 1}-{start + len(chunk_rows)} "
                        f"attempt {attempt + 1}; sleep {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                stop_reason = f"{type(exc).__name__}: {exc}"
                print(f"  chunk failed items {start + 1}-{start + len(chunk_rows)}: {stop_reason}")
                failed = True
                chunk_tagged = contents
                break
        assert chunk_tagged is not None
        tagged_all.extend(chunk_tagged)
        if failed:
            tagged_all.extend(row_to_content(row) for row in rows[start + len(chunk_rows) :])
            return tagged_all, stop_reason, written
        _pairs, persist_rows = _pair_and_persist_rows(chunk_rows, chunk_tagged)
        if persist and persist_rows:
            written += database.update_discovery_candidate_tag_channel(persist_rows)
        done = start + len(chunk_rows)
        tagged_n = sum(
            1
            for item in chunk_tagged
            if str(item.tag_channel_source or "") == TAG_CHANNEL_SOURCE_LLM
        )
        print(
            f"  chunk {start + 1}-{done}: tagged {tagged_n}/{len(chunk_rows)}"
            f" persisted_total={written}"
        )
        if chunk_sleep > 0 and done < len(rows):
            await asyncio.sleep(chunk_sleep)
    return tagged_all, stop_reason, written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--chain",
        default="",
        help="Comma-separated instance ids; empty keeps config default_chain.",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to wait between successful chunks.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Call the channel but do not write tag_channel_* (still spends LLM).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config_path = args.config.expanduser().resolve()
    db_path = args.db.expanduser().resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}")
        return 1
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 1
    live_root = str(LIVE_ROOT.resolve())
    os.environ.setdefault("OPENBILICLAW_PROJECT_ROOT", live_root)

    config = load_config(config_path)
    database = Database(db_path)
    database.initialize()
    try:
        return _run_probe(args, config, config_path, db_path, database)
    finally:
        database.close()


def _run_probe(
    args: argparse.Namespace,
    config: object,
    config_path: Path,
    db_path: Path,
    database: Database,
) -> int:
    pool = load_untagged_teacher_rows(database)
    sample = stratified_sample(pool, limit=max(1, int(args.limit)), seed=int(args.seed))
    print("=== Wave 1 tag-channel probe ===")
    print(f"  config                {config_path}")
    print(f"  db                    {db_path}")
    print(f"  route                 {_route_label(config)}")
    print(f"  llm.timeout           {getattr(getattr(config, 'llm', None), 'timeout', '?')}")
    print(
        "  tag_channel_mode      "
        f"{getattr(getattr(config, 'discovery', None), 'tag_channel_mode', 'off')}"
        " (probe does not change this)"
    )
    print(f"  untagged allowlist    {len(pool)}")
    print(f"  sample                {len(sample)} (limit={args.limit} seed={args.seed})")
    print(f"  max_tokens            {args.max_tokens}")
    if str(args.chain or "").strip():
        print(f"  chain override        {args.chain}")
    platform_counts = Counter(str(row.get("source_platform") or "?") for row in sample)
    print(f"  sample platforms      {dict(platform_counts)}")
    if not sample:
        print("  nothing to tag; allowlist already has tag_channel_source")
        return 0

    baseline = eval_baseline(database)
    eval_calls = int(baseline["calls"])
    eval_tokens = int(baseline["prompt_tokens"]) + int(baseline["completion_tokens"])
    eval_per_call = (eval_tokens / eval_calls) if eval_calls else 0.0
    eval_per_item = eval_per_call / EVAL_CANDIDATES_PER_CALL_ESTIMATE if eval_per_call else 0.0
    print("=== evaluate_batch baseline (last 7 days) ===")
    print(f"  calls                 {eval_calls}")
    print(f"  tokens/call           {eval_per_call:.1f}")
    print(
        f"  tokens/candidate est  {eval_per_item:.1f}"
        f"  (÷ {EVAL_CANDIDATES_PER_CALL_ESTIMATE} default batch)"
    )
    print(f"  cost_cny              {float(baseline['cost_cny']):.4f}")

    usage_before = database.max_llm_usage_id()
    tmp = Path(tempfile.mkdtemp(prefix="obc-tag-probe-"))
    memory = MemoryManager(tmp)
    memory.initialize()
    chain = [item.strip().lower() for item in str(args.chain or "").split(",") if item.strip()]
    registry = build_llm_registry(config, fallback_order=chain or None)
    llm = LLMService(
        registry=registry,
        memory=memory,
        usage_recorder=UsageRecorder(sink=database),
        module_overrides=module_overrides_from_config(config),
        concurrency=llm_concurrency_from_config(config),
    )
    engine = ContentDiscoveryEngine(llm_service=llm, database=database)
    print(f"  default_provider      {registry.default_provider}")
    print(f"  fallback_order        {list(registry._fallback_order())}")
    print("=== calling discovery.tag_batch ===")
    tagged, stop_reason, written = asyncio.run(
        _tag_chunks(
            engine,
            sample,
            batch_size=max(1, int(args.batch_size)),
            max_tokens=max(16, int(args.max_tokens)),
            chunk_sleep=max(0.0, float(args.sleep)),
            database=database,
            persist=not args.no_persist,
        )
    )
    if stop_reason:
        print(f"  stopped early: {stop_reason}")

    pairs, persist_rows = _pair_and_persist_rows(sample, tagged)
    tagged_n = len(persist_rows)
    print(f"  tagged                {tagged_n} / {len(sample)}")
    print(f"  persisted             {written}  (teacher columns untouched)")

    snapshot = database.query_llm_usage_since_id(since_id=usage_before)
    tag_usage = _usage_for_caller(snapshot, TAG_CALLER)
    prompt = int(tag_usage.get("prompt_tokens") or 0)
    completion = int(tag_usage.get("completion_tokens") or 0)
    cached = int(tag_usage.get("cached_input_tokens") or 0)
    calls = int(tag_usage.get("calls") or 0)
    cost = float(tag_usage.get("cost_cny") or 0.0)
    total_tokens = prompt + completion
    per_tagged = (total_tokens / tagged_n) if tagged_n else 0.0
    per_requested = (total_tokens / len(sample)) if sample else 0.0
    cache_rate = (cached / prompt) if prompt else 0.0
    print("=== tag_batch cost ===")
    print(f"  calls                 {calls}")
    print(f"  prompt/completion     {prompt} / {completion}")
    print(f"  cached_input          {cached}  hit={cache_rate:.1%}")
    print(f"  tokens/tagged         {per_tagged:.1f}")
    print(f"  tokens/requested      {per_requested:.1f}")
    print(f"  cost_cny              {cost:.4f}")
    if eval_per_item > 0 and per_tagged > 0:
        print(f"  vs eval est           {per_tagged / eval_per_item:.2f}x tokens/tagged")
    print("=== teacher agreement ===")
    _print_agreement(agreement_stats(pairs))
    return 0 if tagged_n else 2


if __name__ == "__main__":
    raise SystemExit(main())
