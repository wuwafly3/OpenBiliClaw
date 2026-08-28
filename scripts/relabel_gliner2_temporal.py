"""Re-label the boundary temporal classes with the current teacher (LLM).

Re-runs the teacher over training-relevant rows currently labeled
``versioned`` / ``historical`` / ``evergreen``, using the updated evaluation
prompt (the referent-iteration discriminator), and rewrites ONLY the
temporal annotation in place via
``Database.update_discovery_candidate_temporal_relabel``. Relevance scores,
score provenance, status, and pool state are untouched, so the run is safe
against the live recommendation pool.

Usage::

    uv run --extra dev python scripts/relabel_gliner2_temporal.py \\
        --config E:/otherproject/OpenBiliClaw/config.toml \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --dry-run          # print scope without calling the LLM
        --limit 90         # pilot a subset before the full run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_BATCH_SIZE = 30

from openbiliclaw.config import (  # noqa: E402
    llm_concurrency_from_config,
    load_config,
)
from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent  # noqa: E402
from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService, ModuleOverride  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.memory.manager import MemoryManager  # noqa: E402
from openbiliclaw.soul.overrides import apply_overrides  # noqa: E402
from openbiliclaw.soul.profile import OnionProfile  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402

if TYPE_CHECKING:
    from openbiliclaw.discovery.engine import SoulProfile

EVAL_CALLER = "discovery.evaluate_batch"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--labels",
        default="versioned,historical,evergreen",
        help="Comma-separated current temporal classes to re-label.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--instance",
        default="",
        help="Pin evaluation routing to one LLM instance (e.g. openai-4). "
        "Empty = the configured default chain.",
    )
    parser.add_argument(
        "--eval-max-tokens",
        type=int,
        default=0,
        help="Override the discovery eval output budget (engine default 4096). "
        "DeepSeek relays accept large outputs; 16384 lets 20-30 item batches "
        "complete even when a relay forces thinking that would otherwise eat "
        "the 4096 budget. 0 = keep the engine default.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to wait between successful chunks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected scope and exit without calling the LLM.",
    )
    return parser.parse_args(argv)


def load_relabel_rows(database: Database, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    clause = ", ".join("?" for _ in labels)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, content_type, title,
               body_text, bvid, content_id, description, published_at, duration,
               temporal_class
        FROM discovery_candidates
        WHERE temporal_class IN ({clause})
          AND score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY evaluated_at DESC, id DESC
        """,
        (*labels, *sorted(LLM_JUDGMENT_SCORE_SOURCES)),
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


def content_to_relabel_evaluation(row: dict[str, Any], content: DiscoveredContent) -> dict[str, Any]:
    """Project the temporal group off an evaluated item onto a candidate row."""
    return {
        "candidate_id": int(row["id"]),
        "temporal_class": content.temporal_class,
        "temporal_confidence": content.temporal_confidence,
        "temporal_reason": content.temporal_reason,
        "temporal_policy_version": content.temporal_policy_version,
        "temporal_validity_mode": content.temporal_validity_mode,
        "temporal_valid_until": content.temporal_valid_until,
        "temporal_scope": content.temporal_scope,
        "temporal_state": content.temporal_state,
        "temporal_evidence": content.temporal_evidence,
        "temporal_next_review_at": content.temporal_next_review_at,
        "temporal_evaluated_at": content.temporal_evaluated_at,
        "temporal_evidence_complete": content.temporal_evidence_complete,
    }


def load_effective_profile(memory: MemoryManager) -> OnionProfile:
    soul_data = memory.get_layer("soul").data
    if not soul_data:
        raise RuntimeError("Soul profile has not been initialized yet.")
    profile = OnionProfile.from_dict(soul_data)
    return apply_overrides(profile, memory.load_profile_overrides())


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


def _is_transient_server_error(exc: BaseException) -> bool:
    """Transient provider failures worth retrying with backoff (mirrors the
    candidate evaluation pipeline's timeout / connection / server_error class)."""
    text = _exception_text(exc)
    markers = (
        "502",
        "503",
        "504",
        "500 ",
        "server error",
        "bad gateway",
        "service unavailable",
        "connection",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "empty response",
        "no provider",
    )
    return any(marker in text for marker in markers)


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


def _pin_instance_overrides(instance_id: str) -> dict[str, ModuleOverride]:
    """Force evaluation (and discovery) onto one instance; no global spill."""
    chain = (instance_id.strip().lower(),)
    pinned = ModuleOverride(chain=chain, custom_chain=True)
    return {"evaluation": pinned, "discovery": pinned}


def _override_eval_max_tokens(llm_service: Any, max_tokens: int) -> None:
    """Force the discovery eval's output budget for this script process only.

    The engine hardcodes ``max_tokens=4096`` for batch evaluation
    (``_evaluate_batch_once``). Some DeepSeek relays run thinking even when
    the call passes ``reasoning_effort=""``, consuming that budget and
    truncating 10+ item batches. Overriding to 16384 (well within DeepSeek's
    output ceiling) restores 20-30 item batches, which also means fewer LLM
    calls and more consistent labels across the relabel run.
    """

    original = llm_service.complete_structured_task

    async def wrapped(**kwargs: Any) -> Any:
        kwargs["max_tokens"] = max_tokens
        return await original(**kwargs)

    llm_service.complete_structured_task = wrapped  # type: ignore[method-assign]


def _durable_classes(
    database: Database,
    ids: list[int],
) -> dict[int, str]:
    """Reload durable temporal classes for the given candidate ids."""
    if not ids:
        return {}
    rows = database.get_discovery_candidates_by_ids(sorted(ids))
    return {int(row["id"]): str(row.get("temporal_class") or "") for row in rows}


async def _relabel_chunks(
    engine: ContentDiscoveryEngine,
    profile: SoulProfile,
    database: Database,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    chunk_sleep: float,
) -> tuple[int, Counter[tuple[str, str]], str]:
    """Evaluate in chunks and persist the temporal relabel per chunk."""
    updated_total = 0
    transitions: Counter[tuple[str, str]] = Counter()
    stop_reason = ""
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        contents = [row_to_content(row) for row in chunk_rows]
        delay = 15.0
        failed = False
        for attempt in range(5):
            try:
                await engine.evaluate_content_batch(
                    contents,
                    profile,
                    source_context="mixed",
                    batch_size=len(contents),
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
                    break
                if (_is_rate_limit(exc) or _is_transient_server_error(exc)) and attempt < 4:
                    kind = "rate-limited" if _is_rate_limit(exc) else "transient"
                    print(
                        f"  {kind} items {start + 1}-{start + len(chunk_rows)} "
                        f"attempt {attempt + 1}; sleep {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                stop_reason = f"{type(exc).__name__}: {exc}"
                print(f"  chunk failed items {start + 1}-{start + len(chunk_rows)}: {stop_reason}")
                failed = True
                break
        if failed:
            return updated_total, transitions, stop_reason

        evaluations = [
            content_to_relabel_evaluation(row, content)
            for row, content in zip(chunk_rows, contents, strict=True)
        ]
        updated = database.update_discovery_candidate_temporal_relabel(evaluations)
        updated_total += updated
        durable = _durable_classes(database, [int(row["id"]) for row in chunk_rows])
        for row in chunk_rows:
            old = str(row.get("temporal_class") or "")
            new = durable.get(int(row["id"]), old)
            if new != old:
                transitions[(old, new)] += 1
        done = start + len(chunk_rows)
        print(
            f"  chunk {start + 1}-{done}: updated {updated}/{len(chunk_rows)}"
            f" total_updated={updated_total}"
        )
        if chunk_sleep > 0 and done < len(rows):
            await asyncio.sleep(chunk_sleep)
    return updated_total, transitions, stop_reason


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
    os.environ.setdefault("OPENBILICLAW_PROJECT_ROOT", str(config_path.parent))

    labels = tuple(part.strip() for part in str(args.labels).split(",") if part.strip())
    if not labels:
        print("no labels selected")
        return 1

    config = load_config(config_path)
    # Standalone script: mirror [network] into the process-level routing so
    # overseas instances (e.g. tokenharbor) use the configured proxy policy.
    from openbiliclaw.network import set_outbound_proxy

    set_outbound_proxy(config.network.proxy, mode=config.network.mode)
    database = Database(db_path)
    database.initialize()
    try:
        rows = load_relabel_rows(database, labels)
        if args.limit and args.limit > 0:
            rows = rows[: int(args.limit)]
        counts = Counter(str(row.get("temporal_class") or "") for row in rows)
        platforms = Counter(str(row.get("source_platform") or "?") for row in rows)
        print("=== GLiNER2 temporal relabel (LLM re-annotation) ===")
        print(f"  config                {config_path}")
        print(f"  db                    {db_path}")
        print(f"  relabel classes       {', '.join(labels)}")
        print(f"  selected rows         {len(rows)}")
        print(f"  class distribution    {dict(counts.most_common())}")
        print(f"  platforms             {dict(platforms.most_common())}")
        if not rows:
            print("  nothing to re-label")
            return 0
        if args.dry_run:
            print("  dry-run               skip LLM; no DB writes")
            return 0

        usage_before = database.max_llm_usage_id()
        memory = MemoryManager(config.data_path, database=database)
        memory.initialize()
        try:
            profile = load_effective_profile(memory)
        except Exception as exc:
            print(f"  profile load failed: {exc}")
            return 1

        registry = build_llm_registry(config)
        instance_id = str(args.instance or "").strip().lower()
        module_overrides = _pin_instance_overrides(instance_id) if instance_id else None
        if instance_id:
            instances = getattr(getattr(config, "llm", None), "instances", {}) or {}
            if instance_id not in {str(name).strip().lower() for name in instances}:
                print(f"  LLM instance not found: {instance_id}")
                return 1
        llm = LLMService(
            registry=registry,
            memory=memory,
            usage_recorder=UsageRecorder(sink=database),
            module_overrides=module_overrides,
            concurrency=llm_concurrency_from_config(config),
        )
        engine = ContentDiscoveryEngine(
            llm_service=llm,
            database=database,
            eval_prefilter_mode="off",
            tag_channel_mode="off",
            relevance_scorer="llm",
            multimodal_evaluation_enabled=False,
        )
        engine._recent_viewed_content_keys = lambda: set()  # type: ignore[method-assign]
        if int(args.eval_max_tokens or 0) > 0:
            _override_eval_max_tokens(llm, int(args.eval_max_tokens))
            print(f"  eval_max_tokens       {int(args.eval_max_tokens)} (override)")
        print(f"  default_provider      {registry.default_provider}")
        print("=== calling discovery.evaluate_batch (temporal relabel, no score write) ===")
        updated, transitions, stop_reason = asyncio.run(
            _relabel_chunks(
                engine,
                profile,
                database,
                rows,
                batch_size=max(1, int(args.batch_size)),
                chunk_sleep=max(0.0, float(args.sleep)),
            )
        )
        if stop_reason:
            print(f"  stopped early: {stop_reason}")

        usage = database.query_llm_usage_since_id(since_id=usage_before)
        eval_usage = _usage_for_caller(usage, EVAL_CALLER)
        final_counts = Counter()
        for rid in [int(row["id"]) for row in rows]:
            durable = _durable_classes(database, [rid])
            if durable:
                final_counts[durable[rid]] += 1
        print("=== relabel summary ===")
        print(f"  rows evaluated        {len(rows)}")
        print(f"  rows touched          {updated}")
        print(f"  classes changed       {sum(transitions.values())}")
        if transitions:
            print("  transitions:")
            for (old, new), n in transitions.most_common():
                print(f"    {old:12s} -> {new:12s} {n}")
        print(f"  final class dist      {dict(final_counts.most_common())}")
        print(f"  calls                 {int(eval_usage.get('calls') or 0)}")
        print(f"  prompt/completion     {int(eval_usage.get('prompt_tokens') or 0)} / "
              f"{int(eval_usage.get('completion_tokens') or 0)}")
        print(f"  cost_cny              {float(eval_usage.get('cost_cny') or 0.0):.4f}")
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
