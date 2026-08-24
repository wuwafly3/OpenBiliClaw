"""Re-score old teacher rows under the live gate snapshot contract.

Rewrites labeling-time snapshots (drop recent-layer keys, drop page-shell
negatives), then re-runs the pinned teacher on those candidates. New scores
go to a versioned JSONL. ``discovery_candidates`` scores and digests are
never updated.

Usage::

    uv run python scripts/ml_gate_contract_relabel.py --dry-run
    uv run python scripts/ml_gate_contract_relabel.py --limit 45
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_OUT_DIR = LIVE_ROOT / "data" / "ml_artifacts"
DEFAULT_INSTANCE = "openai-4"
DEFAULT_BATCH_SIZE = 45
PINNED_TEACHER = "openai/deepseek-v4-flash"
EVAL_CALLER = "discovery.evaluate_batch"
FORBIDDEN_CANDIDATE_UPDATES = (
    "relevance_score",
    "llm_score_raw",
    "score_source",
    "teacher_model",
    "profile_digest",
    "negative_digest",
)

from export_ranking_dataset import effective_admission_threshold  # noqa: E402
from ml_teacher_self_consistency_probe import (  # noqa: E402
    _eval_chunks,
    _usage_for_caller,
    expected_teacher_identity,
    filter_rows_for_instance,
    load_effective_profile,
    load_teacher_allowlist_rows,
    pin_instance_overrides,
    resolve_llm_instance,
    resolve_replay_snapshot,
)

from openbiliclaw.config import llm_concurrency_from_config, load_config  # noqa: E402
from openbiliclaw.discovery.engine import ContentDiscoveryEngine  # noqa: E402
from openbiliclaw.discovery.eval_context import (  # noqa: E402
    GateContractRewrite,
    rewrite_snapshot_to_gate_contract,
)
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.memory.manager import MemoryManager  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402


def binary_label(score: float, source_strategy: object) -> int:
    return int(float(score) >= effective_admission_threshold(source_strategy))


def load_resume_ids(path: Path | None) -> set[int]:
    if path is None or not path.is_file():
        return set()
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        seen.add(int(payload["candidate_id"]))
    return seen


def collect_relabel_cohort(
    database: Database,
    *,
    provider_type: str,
    model: str,
    resume_ids: set[int] | None = None,
    limit: int | None = None,
) -> tuple[
    list[tuple[GateContractRewrite, list[dict[str, Any]]]],
    dict[str, int],
]:
    """Pinned teacher rows whose snapshot still needs the new gate contract."""

    skip = resume_ids or set()
    pool = load_teacher_allowlist_rows(database)
    eligible, filter_counts = filter_rows_for_instance(
        pool,
        provider_type=provider_type,
        model=model,
    )
    counts: Counter[str] = Counter(filter_counts)
    counts["pinned"] = int(counts.pop("kept", 0))
    counts["kept"] = 0
    by_rewrite: dict[tuple[str, str], tuple[GateContractRewrite, list[dict[str, Any]]]] = {}
    cache: dict[tuple[str, str], GateContractRewrite | None] = {}
    for row in eligible:
        candidate_id = int(row["id"])
        if candidate_id in skip:
            counts["skipped_resume"] += 1
            continue
        key = (
            str(row.get("profile_digest") or "").strip(),
            str(row.get("negative_digest") or "").strip(),
        )
        if not key[0]:
            counts["dropped_empty_digest"] += 1
            continue
        if key not in cache:
            snapshot = resolve_replay_snapshot(
                database,
                profile_digest=key[0],
                negative_digest=key[1],
            )
            if snapshot is None:
                cache[key] = None
            else:
                cache[key] = rewrite_snapshot_to_gate_contract(snapshot)
        rewritten = cache[key]
        if rewritten is None:
            counts["dropped_no_snapshot"] += 1
            continue
        if not rewritten.had_recent and rewritten.dropped_shell == 0:
            counts["already_new_contract"] += 1
            continue
        counts["kept"] += 1
        group_key = (
            rewritten.snapshot.profile_digest,
            rewritten.snapshot.negative_digest,
        )
        if group_key not in by_rewrite:
            by_rewrite[group_key] = (rewritten, [])
        by_rewrite[group_key][1].append(row)
        if limit is not None and counts["kept"] >= limit:
            break
    groups = list(by_rewrite.values())
    counts["groups"] = len(groups)
    return groups, dict(counts)


def cohort_platform_counts(
    groups: list[tuple[GateContractRewrite, list[dict[str, Any]]]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _rewrite, rows in groups:
        for row in rows:
            counts[str(row.get("source_platform") or "unknown") or "unknown"] += 1
    return dict(counts)


def print_cohort_stats(
    counts: dict[str, int],
    groups: list[tuple[GateContractRewrite, list[dict[str, Any]]]],
) -> None:
    platforms = cohort_platform_counts(groups)
    print("=== gate-contract relabel ===")
    print(f"  allowlist             {counts.get('allowlist', 0)}")
    print(f"  dropped model         {counts.get('dropped_model_mismatch', 0)}")
    print(f"  dropped provider      {counts.get('dropped_provider_mismatch', 0)}")
    print(f"  pinned                {counts.get('pinned', 0)}")
    print(f"  empty digest          {counts.get('dropped_empty_digest', 0)}")
    print(f"  no snapshot           {counts.get('dropped_no_snapshot', 0)}")
    print(f"  already new contract  {counts.get('already_new_contract', 0)}")
    print(f"  skipped resume        {counts.get('skipped_resume', 0)}")
    print(f"  kept                  {counts.get('kept', 0)}")
    print(f"  new digest groups     {counts.get('groups', 0)}")
    print(f"  platforms             {platforms}")
    for rewrite, rows in groups:
        print(
            "  group "
            f"{rewrite.snapshot.profile_digest[:12]}/"
            f"{rewrite.snapshot.negative_digest[:12]} "
            f"n={len(rows)} recent={int(rewrite.had_recent)} "
            f"shell={rewrite.dropped_shell}"
        )


def output_row(
    row: dict[str, Any],
    rewrite: GateContractRewrite,
    *,
    new_score: float,
    new_teacher_model: str,
    new_score_source: str,
) -> dict[str, Any]:
    strategy = str(row.get("source_strategy") or "")
    old_score = float(row["teacher_score"])
    return {
        "candidate_id": int(row["id"]),
        "source_platform": str(row.get("source_platform") or ""),
        "source_strategy": strategy,
        "old_profile_digest": rewrite.source_profile_digest,
        "old_negative_digest": rewrite.source_negative_digest,
        "new_profile_digest": rewrite.snapshot.profile_digest,
        "new_negative_digest": rewrite.snapshot.negative_digest,
        "old_llm_score_raw": old_score,
        "new_llm_score_raw": float(new_score),
        "old_y": binary_label(old_score, strategy),
        "new_y": binary_label(float(new_score), strategy),
        "old_teacher_model": str(row.get("teacher_model") or ""),
        "new_teacher_model": new_teacher_model,
        "new_score_source": new_score_source,
        "had_recent": rewrite.had_recent,
        "dropped_shell": rewrite.dropped_shell,
    }


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cohort stats without calling the LLM or writing snapshots.",
    )
    return parser.parse_args(argv)


def _run_relabel(
    args: argparse.Namespace,
    *,
    config: object,
    config_path: Path,
    db_path: Path,
    database: Database,
    instance_id: str,
    provider_type: str,
    model: str,
    expected_identity: str,
) -> int:
    resume_ids = load_resume_ids(args.resume)
    groups, counts = collect_relabel_cohort(
        database,
        provider_type=provider_type,
        model=model,
        resume_ids=resume_ids,
        limit=args.limit,
    )
    print_cohort_stats(counts, groups)
    print(f"  config                {config_path}")
    print(f"  db                    {db_path}")
    print(f"  pinned instance       {instance_id}")
    print(f"  expected identity     {expected_identity}")
    print("  persist               JSONL only; discovery_candidates untouched")
    if args.dry_run:
        print("  dry-run               skip LLM")
        return 0 if int(counts.get("kept", 0)) > 0 else 2
    if not groups:
        print("  nothing to relabel")
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir.expanduser().resolve()
    jsonl_path = out_dir / f"gate_contract_relabel_{stamp}.jsonl"
    meta_path = out_dir / f"gate_contract_relabel_{stamp}.meta.json"

    usage_before = database.max_llm_usage_id()
    memory = MemoryManager(config.data_path, database=database)
    memory.initialize()
    try:
        profile = load_effective_profile(memory)
    except Exception as exc:
        print(f"  profile load failed: {exc}")
        return 1

    registry = build_llm_registry(config, fallback_order=[instance_id])
    if not registry.is_chat_capable(instance_id):
        print(f"LLM instance not chat-capable: {instance_id}")
        return 1
    llm = LLMService(
        registry=registry,
        memory=memory,
        usage_recorder=UsageRecorder(sink=database),
        module_overrides=pin_instance_overrides(instance_id),
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
    print("=== calling discovery.evaluate_batch (no candidate UPDATE) ===")

    import asyncio

    written = 0
    stop_reason = ""
    upserted: set[tuple[str, str]] = set()
    for rewrite, group_rows in groups:
        snapshot = rewrite.snapshot
        print(f"  context group         n={len(group_rows)} digest={snapshot.profile_digest[:8]}…")
        pairs, group_stop = asyncio.run(
            _eval_chunks(
                engine,
                profile,
                group_rows,
                batch_size=max(1, int(args.batch_size)),
                chunk_sleep=max(0.0, float(args.sleep)),
                expected_identity=expected_identity,
                snapshot=snapshot,
            )
        )
        by_id = {int(item["candidate_id"]): item for item in pairs}
        out_rows: list[dict[str, Any]] = []
        for row in group_rows:
            item = by_id.get(int(row["id"]))
            if item is None:
                continue
            out_rows.append(
                output_row(
                    row,
                    rewrite,
                    new_score=float(item["replay_score"]),
                    new_teacher_model=str(item.get("replay_model") or expected_identity),
                    new_score_source="llm",
                )
            )
        if out_rows:
            append_jsonl(jsonl_path, out_rows)
            written += len(out_rows)
            key = (snapshot.profile_digest, snapshot.negative_digest)
            if key not in upserted:
                database.upsert_evaluation_context_snapshot(snapshot)
                upserted.add(key)
        if group_stop:
            stop_reason = group_stop
            print(f"  stopped early: {stop_reason}")
            break

    usage = database.query_llm_usage_since_id(since_id=usage_before)
    eval_usage = _usage_for_caller(usage, EVAL_CALLER)
    meta = {
        "instance_id": instance_id,
        "expected_identity": expected_identity,
        "filter_counts": dict(counts),
        "platforms": cohort_platform_counts(groups),
        "written": written,
        "jsonl": str(jsonl_path),
        "stop_reason": stop_reason,
        "usage": eval_usage,
        "notes": [
            "New scores are under compact_gate_evaluation_profile_summary.",
            "discovery_candidates scores and digests were not updated.",
            "Old snapshots with recent keys were kept.",
        ],
    }
    write_meta(meta_path, meta)
    print("=== relabel cost ===")
    print(f"  calls                 {eval_usage.get('calls', 0)}")
    print(f"  cost_cny              {float(eval_usage.get('cost_cny') or 0.0):.4f}")
    print(f"  written               {written}")
    print(f"  jsonl                 {jsonl_path}")
    print(f"  meta                  {meta_path}")
    if written <= 0:
        return 2
    return 0


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

    config = load_config(config_path)
    instance_id = str(args.instance or "").strip().lower()
    instance = resolve_llm_instance(config, instance_id)
    if instance is None:
        print(f"LLM instance not found: {instance_id}")
        return 1
    if not bool(getattr(instance, "enabled", True)):
        print(f"LLM instance disabled: {instance_id}")
        return 1
    provider_type = str(getattr(instance, "provider_type", "") or "").strip()
    model = str(getattr(instance, "model", "") or "").strip()
    if not provider_type or not model:
        print(f"LLM instance {instance_id} missing provider_type or model")
        return 1
    expected_identity = expected_teacher_identity(provider_type=provider_type, model=model)
    if expected_identity != PINNED_TEACHER:
        print(f"refusing to relabel with {expected_identity}; pinned teacher is {PINNED_TEACHER}")
        return 1

    database = Database(db_path)
    database.initialize()
    try:
        return _run_relabel(
            args,
            config=config,
            config_path=config_path,
            db_path=db_path,
            database=database,
            instance_id=instance_id,
            provider_type=provider_type,
            model=model,
            expected_identity=expected_identity,
        )
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
