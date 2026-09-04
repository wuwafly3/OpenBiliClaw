"""Replay JSONL new-contract scores against the same frozen snapshot.

Binds each row's ``new_profile_digest`` / ``new_negative_digest`` and
re-runs the pinned teacher. Compares the second draw to the first JSONL
``new_y`` (not the 8-20 old-contract score). Does not UPDATE
``discovery_candidates``.

Usage::

    uv run python scripts/ml_gate_contract_self_consistency.py --dry-run
    uv run python scripts/ml_gate_contract_self_consistency.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_JSONL = LIVE_ROOT / "data" / "ml_artifacts" / "gate_contract_relabel_20260824T030808Z.jsonl"
DEFAULT_OUT_DIR = LIVE_ROOT / "data" / "ml_artifacts"
DEFAULT_INSTANCE = "openai-4"
DEFAULT_BATCH_SIZE = 45
PINNED_TEACHER = "openai/deepseek-v4-flash"
EVAL_CALLER = "discovery.evaluate_batch"

from export_ranking_dataset import effective_admission_threshold  # noqa: E402
from ml_teacher_self_consistency_probe import (  # noqa: E402
    _eval_chunks,
    _print_stats,
    _usage_for_caller,
    admission_agreement_stats,
    expected_teacher_identity,
    is_near_threshold,
    load_effective_profile,
    pin_instance_overrides,
    resolve_llm_instance,
    resolve_replay_snapshot,
)

from openbiliclaw.config import llm_concurrency_from_config, load_config  # noqa: E402
from openbiliclaw.discovery.engine import ContentDiscoveryEngine  # noqa: E402
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.memory.manager import MemoryManager  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402

if TYPE_CHECKING:
    from openbiliclaw.discovery.eval_context import EvaluationContextSnapshot


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def load_resume_ids(path: Path | None) -> set[int]:
    if path is None or not path.is_file():
        return set()
    seen: set[int] = set()
    for payload in load_jsonl(path):
        seen.add(int(payload["candidate_id"]))
    return seen


def group_jsonl_by_new_digest(
    jsonl_rows: list[dict[str, Any]],
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for payload in jsonl_rows:
        key = (
            str(payload.get("new_profile_digest") or "").strip(),
            str(payload.get("new_negative_digest") or "").strip(),
        )
        if key not in groups:
            order.append(key)
        groups[key].append(payload)
    return [(key, groups[key]) for key in order]


def load_candidate_rows(database: Database, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, content_type, title,
               body_text, bvid, content_id, description, published_at, duration,
               teacher_model, score_source, llm_score_raw, relevance_score,
               profile_digest, negative_digest
        FROM discovery_candidates
        WHERE id IN ({placeholders})
        """,
        tuple(ids),
    )
    loaded: dict[int, dict[str, Any]] = {}
    for raw in cursor.fetchall():
        row = dict(raw)
        loaded[int(row["id"])] = row
    return loaded


def attach_first_draw(
    candidate: dict[str, Any],
    first: dict[str, Any],
) -> dict[str, Any]:
    row = dict(candidate)
    row["first_llm_score_raw"] = float(first["new_llm_score_raw"])
    row["first_y"] = int(first["new_y"])
    row["old_llm_score_raw"] = float(first["old_llm_score_raw"])
    row["old_y"] = int(first["old_y"])
    row["new_profile_digest"] = str(first["new_profile_digest"])
    row["new_negative_digest"] = str(first["new_negative_digest"])
    row["teacher_score"] = float(first["new_llm_score_raw"])
    return row


def collect_replay_groups(
    database: Database,
    jsonl_rows: list[dict[str, Any]],
    *,
    resume_ids: set[int] | None = None,
    limit: int | None = None,
) -> tuple[
    list[tuple[EvaluationContextSnapshot, list[dict[str, Any]]]],
    dict[str, int],
]:
    skip = resume_ids or set()
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for payload in jsonl_rows:
        counts["jsonl"] += 1
        candidate_id = int(payload["candidate_id"])
        if candidate_id in skip:
            counts["skipped_resume"] += 1
            continue
        if not str(payload.get("new_profile_digest") or "").strip():
            counts["dropped_empty_digest"] += 1
            continue
        selected.append(payload)
        counts["kept"] += 1
        if limit is not None and counts["kept"] >= limit:
            break

    candidate_rows = load_candidate_rows(database, [int(item["candidate_id"]) for item in selected])
    grouped = group_jsonl_by_new_digest(selected)
    replay: list[tuple[EvaluationContextSnapshot, list[dict[str, Any]]]] = []
    for key, payloads in grouped:
        snapshot = resolve_replay_snapshot(
            database,
            profile_digest=key[0],
            negative_digest=key[1],
        )
        if snapshot is None:
            counts["dropped_no_snapshot"] += len(payloads)
            continue
        rows: list[dict[str, Any]] = []
        for payload in payloads:
            candidate = candidate_rows.get(int(payload["candidate_id"]))
            if candidate is None:
                counts["dropped_missing_candidate"] += 1
                continue
            rows.append(attach_first_draw(candidate, payload))
        if rows:
            replay.append((snapshot, rows))
    counts["groups"] = len(replay)
    counts["replay_rows"] = sum(len(rows) for _snapshot, rows in replay)
    return replay, dict(counts)


def print_cohort_stats(
    counts: dict[str, int],
    groups: list[tuple[EvaluationContextSnapshot, list[dict[str, Any]]]],
) -> None:
    platforms: Counter[str] = Counter()
    for _snapshot, rows in groups:
        for row in rows:
            platforms[str(row.get("source_platform") or "unknown") or "unknown"] += 1
    print("=== gate-contract self-consistency ===")
    print(f"  jsonl                 {counts.get('jsonl', 0)}")
    print(f"  skipped resume        {counts.get('skipped_resume', 0)}")
    print(f"  empty digest          {counts.get('dropped_empty_digest', 0)}")
    print(f"  no snapshot           {counts.get('dropped_no_snapshot', 0)}")
    print(f"  missing candidate     {counts.get('dropped_missing_candidate', 0)}")
    print(f"  replay rows           {counts.get('replay_rows', 0)}")
    print(f"  new digest groups     {counts.get('groups', 0)}")
    print(f"  platforms             {dict(platforms)}")
    for snapshot, rows in groups:
        print(
            f"  group {snapshot.profile_digest[:12]}/{snapshot.negative_digest[:12]} n={len(rows)}"
        )


def pair_to_output(row: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    strategy = str(row.get("source_strategy") or "")
    first_score = float(row["first_llm_score_raw"])
    second_score = float(replay["replay_score"])
    first_y = int(row["first_y"])
    second_y = int(replay["replay_y"])
    floor = effective_admission_threshold(strategy)
    return {
        "candidate_id": int(row["id"]),
        "source_platform": str(row.get("source_platform") or ""),
        "source_strategy": strategy,
        "new_profile_digest": str(row["new_profile_digest"]),
        "new_negative_digest": str(row["new_negative_digest"]),
        "first_llm_score_raw": first_score,
        "second_llm_score_raw": second_score,
        "first_y": first_y,
        "second_y": second_y,
        "old_y": int(row["old_y"]),
        "flip": int(first_y != second_y),
        "floor": floor,
        "near_threshold": is_near_threshold(first_score, floor),
        "second_teacher_model": str(replay.get("replay_model") or ""),
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
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run(
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
    jsonl_path = args.jsonl.expanduser().resolve()
    if not jsonl_path.is_file():
        print(f"jsonl not found: {jsonl_path}")
        return 1
    jsonl_rows = load_jsonl(jsonl_path)
    groups, counts = collect_replay_groups(
        database,
        jsonl_rows,
        resume_ids=load_resume_ids(args.resume),
        limit=args.limit,
    )
    print_cohort_stats(counts, groups)
    print(f"  config                {config_path}")
    print(f"  db                    {db_path}")
    print(f"  jsonl                 {jsonl_path}")
    print(f"  pinned instance       {instance_id}")
    print("  persist               JSONL only; discovery_candidates untouched")
    if args.dry_run:
        print("  dry-run               skip LLM")
        return 0 if int(counts.get("replay_rows", 0)) > 0 else 2
    if not groups:
        print("  nothing to replay")
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir.expanduser().resolve()
    out_jsonl = out_dir / f"gate_contract_self_consistency_{stamp}.jsonl"
    meta_path = out_dir / f"gate_contract_self_consistency_{stamp}.meta.json"

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
    print("=== calling discovery.evaluate_batch (same new-contract snapshot) ===")

    written_rows: list[dict[str, Any]] = []
    stop_reason = ""
    for snapshot, group_rows in groups:
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
        chunk_out: list[dict[str, Any]] = []
        for row in group_rows:
            item = by_id.get(int(row["id"]))
            if item is None:
                continue
            chunk_out.append(pair_to_output(row, item))
        if chunk_out:
            append_jsonl(out_jsonl, chunk_out)
            written_rows.extend(chunk_out)
        if group_stop:
            stop_reason = group_stop
            print(f"  stopped early: {stop_reason}")
            break

    stats_pairs = [
        {
            "orig_y": row["first_y"],
            "replay_y": row["second_y"],
            "orig_score": row["first_llm_score_raw"],
            "replay_score": row["second_llm_score_raw"],
            "platform": row["source_platform"],
            "near_threshold": row["near_threshold"],
        }
        for row in written_rows
    ]
    stats = admission_agreement_stats(stats_pairs)
    flips = sum(int(row["flip"]) for row in written_rows)
    usage = database.query_llm_usage_since_id(since_id=usage_before)
    eval_usage = _usage_for_caller(usage, EVAL_CALLER)
    meta = {
        "instance_id": instance_id,
        "expected_identity": expected_identity,
        "source_jsonl": str(jsonl_path),
        "filter_counts": dict(counts),
        "written": len(written_rows),
        "flips": flips,
        "flip_rate": (flips / len(written_rows)) if written_rows else None,
        "stats": stats,
        "stop_reason": stop_reason,
        "usage": eval_usage,
        "jsonl": str(out_jsonl),
        "notes": [
            "Second draw used the same new-contract snapshot as the first JSONL.",
            "Flip rate is first_y vs second_y, not vs the 8-20 old-contract label.",
            "discovery_candidates scores were not updated.",
        ],
    }
    write_meta(meta_path, meta)
    print("=== self-consistency cost ===")
    print(f"  calls                 {eval_usage.get('calls', 0)}")
    print(f"  cost_cny              {float(eval_usage.get('cost_cny') or 0.0):.4f}")
    print(f"  written               {len(written_rows)}")
    print(
        "  flip rate             "
        f"{flips}/{len(written_rows)}"
        + (f" ({flips / len(written_rows):.3f})" if written_rows else "")
    )
    print("=== first vs second new-contract y ===")
    _print_stats(stats)
    print(f"  jsonl                 {out_jsonl}")
    print(f"  meta                  {meta_path}")
    if not written_rows:
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
    provider_type = str(getattr(instance, "provider_type", "") or "").strip()
    model = str(getattr(instance, "model", "") or "").strip()
    expected_identity = expected_teacher_identity(provider_type=provider_type, model=model)
    if expected_identity != PINNED_TEACHER:
        print(f"refusing replay with {expected_identity}; pinned teacher is {PINNED_TEACHER}")
        return 1

    database = Database(db_path)
    database.initialize()
    try:
        return _run(
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
