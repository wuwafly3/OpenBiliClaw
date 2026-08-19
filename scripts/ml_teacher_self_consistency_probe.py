"""Live measurement of teacher self-consistency (agreement ceiling).

Re-evaluates a stratified sample of teacher-allowlist rows through one
pinned LLM instance and compares the new admission labels with the stored
ones. That number is the theoretical ceiling for ML-vs-teacher agreement:
the student cannot out-agree a noisy teacher.

Usage::

    uv run --extra dev python scripts/ml_teacher_self_consistency_probe.py \\
        --config E:/otherproject/OpenBiliClaw/config.toml \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --instance openai-4 \\
        --limit 90

Only original rows whose ``teacher_model`` is an exact ``provider/model``
match with the pinned instance are eligible. Different adapters that share
a model id (``openai`` vs ``openai_compatible``) are excluded.

When ``profile_digest`` / ``negative_digest`` plus a matching
``evaluation_context_snapshots`` row exist, replay binds that frozen
compact profile and negative list. Mixed digest pairs never share an
eval batch. Legacy rows without a snapshot fall back to the current
effective profile unless ``--require-snapshot`` skips them.

Does not persist scores or backfill digests onto ``discovery_candidates``.
``llm_usage`` rows from the replay calls are recorded. Diagnostic, not a
merge gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_OUT = PROJECT_ROOT / "data" / "ml_artifacts" / "teacher_self_consistency_v1.json"
DEFAULT_INSTANCE = "openai-4"
DEFAULT_LIMIT = 90
DEFAULT_BATCH_SIZE = 45
NEAR_THRESHOLD_BAND = 0.05
EVAL_CALLER = "discovery.evaluate_batch"
# 2026-08-19 teacher-oracle OOF at FPR<=0.10 / default 0.50. Printed as
# context only; this probe does not retrain.
ML_AGREEMENT_AT_070 = 0.699
ML_AGREEMENT_AT_050 = 0.744
S15_AGREEMENT_GATE = 0.90

from export_ranking_dataset import (  # noqa: E402
    candidate_row_for_label,
    effective_admission_threshold,
    resolve_teacher_label,
)

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
    from openbiliclaw.discovery.eval_context import EvaluationContextSnapshot


def parse_teacher_identity(teacher_model: str) -> tuple[str, str]:
    """Split ``provider/model``; model may itself contain slashes."""

    text = str(teacher_model or "").strip()
    if not text:
        return "", ""
    if "/" not in text:
        return "", text
    provider, model = text.split("/", 1)
    return provider.strip(), model.strip()


def expected_teacher_identity(*, provider_type: str, model: str) -> str:
    """Identity stamped by ``_response_teacher_identity`` for this instance."""

    provider = str(provider_type or "").strip()
    model_name = str(model or "").strip()
    if provider and model_name:
        return f"{provider}/{model_name}"
    return model_name or provider


def teacher_identity_matches(
    teacher_model: str,
    *,
    provider: str,
    model: str,
) -> bool:
    """Exact provider and exact model name; no prefix, no adapter mix."""

    got_provider, got_model = parse_teacher_identity(teacher_model)
    return got_provider == str(provider or "").strip() and got_model == str(model or "").strip()


def filter_rows_for_instance(
    rows: list[dict[str, Any]],
    *,
    provider_type: str,
    model: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Keep rows whose stored teacher identity matches the pinned instance."""

    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    want_provider = str(provider_type or "").strip()
    want_model = str(model or "").strip()
    for row in rows:
        counts["allowlist"] += 1
        got_provider, got_model = parse_teacher_identity(str(row.get("teacher_model") or ""))
        if got_model != want_model:
            counts["dropped_model_mismatch"] += 1
            continue
        if got_provider != want_provider:
            counts["dropped_provider_mismatch"] += 1
            continue
        kept.append(row)
        counts["kept"] += 1
    return kept, counts


def is_near_threshold(
    score: float,
    floor: float,
    *,
    band: float = NEAR_THRESHOLD_BAND,
) -> bool:
    return abs(float(score) - float(floor)) <= float(band)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for cursor in range(index, end + 1):
            ranks[indexed[cursor][0]] = average
        index = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    count = len(xs)
    if count < 2 or count != len(ys):
        return None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / math.sqrt(denom_x * denom_y)


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def admission_agreement_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay vs original admission labels; original is the reference draw."""

    compared = len(pairs)
    true_pos = true_neg = false_pos = false_neg = 0
    abs_err = 0.0
    by_platform: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "agree": 0, "fp": 0, "fn": 0}
    )
    near = {"n": 0, "agree": 0}
    for item in pairs:
        orig_y = int(item["orig_y"])
        replay_y = int(item["replay_y"])
        agreed = orig_y == replay_y
        if orig_y == 1 and replay_y == 1:
            true_pos += 1
        elif orig_y == 0 and replay_y == 0:
            true_neg += 1
        elif orig_y == 0 and replay_y == 1:
            false_pos += 1
        else:
            false_neg += 1
        abs_err += abs(float(item["orig_score"]) - float(item["replay_score"]))
        platform = str(item.get("platform") or "unknown") or "unknown"
        bucket = by_platform[platform]
        bucket["n"] += 1
        if agreed:
            bucket["agree"] += 1
        if orig_y == 0 and replay_y == 1:
            bucket["fp"] += 1
        if orig_y == 1 and replay_y == 0:
            bucket["fn"] += 1
        if item.get("near_threshold"):
            near["n"] += 1
            if agreed:
                near["agree"] += 1
    orig_neg = true_neg + false_pos
    orig_pos = true_pos + false_neg
    scores_a = [float(item["orig_score"]) for item in pairs]
    scores_b = [float(item["replay_score"]) for item in pairs]
    return {
        "compared": compared,
        "agree": true_pos + true_neg,
        "agreement": ((true_pos + true_neg) / compared) if compared else None,
        "tp": true_pos,
        "tn": true_neg,
        "fp": false_pos,
        "fn": false_neg,
        "fpr": (false_pos / orig_neg) if orig_neg else None,
        "fnr": (false_neg / orig_pos) if orig_pos else None,
        "mae": (abs_err / compared) if compared else None,
        "spearman": spearman_rho(scores_a, scores_b),
        "by_platform": dict(by_platform),
        "near_threshold": near,
    }


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


def eval_context_key(row: dict[str, Any]) -> tuple[str, str]:
    """Grouping key so mixed labeling-time contexts never share a batch."""

    return (
        str(row.get("profile_digest") or "").strip(),
        str(row.get("negative_digest") or "").strip(),
    )


def group_rows_by_eval_context(
    rows: list[dict[str, Any]],
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """Partition rows by ``(profile_digest, negative_digest)`` in first-seen order."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for row in rows:
        key = eval_context_key(row)
        if key not in groups:
            order.append(key)
        groups[key].append(row)
    return [(key, groups[key]) for key in order]


def resolve_replay_snapshot(
    database: Database,
    *,
    profile_digest: str,
    negative_digest: str,
) -> EvaluationContextSnapshot | None:
    """Load a verified snapshot, or None for legacy empty / missing / corrupt rows."""

    digest = str(profile_digest or "").strip()
    if not digest:
        return None
    getter = getattr(database, "get_evaluation_context_snapshot", None)
    if not callable(getter):
        return None
    try:
        snapshot = getter(
            profile_digest=digest,
            negative_digest=str(negative_digest or "").strip(),
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "evaluation context snapshot lookup failed",
            exc_info=True,
        )
        return None
    if snapshot is None:
        return None
    if not getattr(snapshot, "digests_match", lambda: False)():
        return None
    return snapshot


def prepare_replay_groups(
    rows: list[dict[str, Any]],
    database: Database,
    *,
    require_snapshot: bool,
) -> tuple[
    list[tuple[EvaluationContextSnapshot | None, list[dict[str, Any]]]],
    dict[str, int],
]:
    """Group sample rows and attach a verified snapshot when one exists."""

    groups = group_rows_by_eval_context(rows)
    replay: list[tuple[EvaluationContextSnapshot | None, list[dict[str, Any]]]] = []
    hits = 0
    misses = 0
    skipped = 0
    for key, group_rows in groups:
        snapshot = resolve_replay_snapshot(
            database,
            profile_digest=key[0],
            negative_digest=key[1],
        )
        if snapshot is None:
            misses += len(group_rows)
            if require_snapshot:
                skipped += len(group_rows)
                continue
        else:
            hits += len(group_rows)
        replay.append((snapshot, group_rows))
    counts = {
        "groups": len(groups),
        "snapshot_hits": hits,
        "snapshot_misses": misses,
        "skipped_no_snapshot": skipped,
        "replay_rows": sum(len(group) for _snapshot, group in replay),
    }
    return replay, counts


def pin_instance_overrides(instance_id: str) -> dict[str, ModuleOverride]:
    """Force evaluation (and discovery) onto one instance; no global spill."""

    chain = (instance_id.strip().lower(),)
    pinned = ModuleOverride(chain=chain, custom_chain=True)
    return {"evaluation": pinned, "discovery": pinned}


def replay_chunk_usable(
    contents: list[Any],
    *,
    expected_identity: str,
) -> tuple[bool, str]:
    """Reject a chunk if any non-empty stamp is not the pinned identity."""

    identities = sorted(
        {
            str(getattr(item, "teacher_model", "") or "").strip()
            for item in contents
            if str(getattr(item, "teacher_model", "") or "").strip()
        }
    )
    if not identities:
        return False, "empty_teacher_model"
    if identities != [expected_identity]:
        return False, f"identity_mismatch:{','.join(identities)}"
    return True, expected_identity


def pair_replay(
    rows: list[dict[str, Any]],
    contents: list[Any],
    *,
    expected_identity: str,
) -> tuple[list[dict[str, Any]], str]:
    usable, reason = replay_chunk_usable(contents, expected_identity=expected_identity)
    if not usable:
        return [], reason
    pairs: list[dict[str, Any]] = []
    for row, content in zip(rows, contents, strict=True):
        source = str(getattr(content, "score_source", "") or "")
        raw = getattr(content, "llm_score_raw", None)
        stamp = str(getattr(content, "teacher_model", "") or "").strip()
        if source not in LLM_JUDGMENT_SCORE_SOURCES or raw is None:
            continue
        if stamp != expected_identity:
            continue
        orig_score = float(row["teacher_score"])
        replay_score = float(raw)
        strategy = str(row.get("source_strategy") or "")
        floor = effective_admission_threshold(strategy)
        pairs.append(
            {
                "candidate_id": int(row["id"]),
                "platform": str(row.get("source_platform") or "unknown") or "unknown",
                "strategy": strategy,
                "orig_score": orig_score,
                "replay_score": replay_score,
                "orig_y": int(orig_score >= floor),
                "replay_y": int(replay_score >= floor),
                "floor": floor,
                "near_threshold": is_near_threshold(orig_score, floor),
                "replay_model": stamp,
            }
        )
    return pairs, ""


def resolve_llm_instance(config: object, instance_id: str) -> Any:
    llm = getattr(config, "llm", None)
    instances = getattr(llm, "instances", {}) or {}
    if not isinstance(instances, dict):
        return None
    wanted = instance_id.strip().lower()
    mapping = {str(name).strip().lower(): inst for name, inst in instances.items()}
    return mapping.get(wanted)


def load_teacher_allowlist_rows(database: Database) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, content_type, title,
               body_text, bvid, content_id, description, published_at, duration,
               teacher_model, score_source, llm_score_raw, relevance_score,
               profile_digest, negative_digest
        FROM discovery_candidates
        WHERE score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY evaluated_at DESC, id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    loaded: list[dict[str, Any]] = []
    for raw in cursor.fetchall():
        row = dict(raw)
        resolved = resolve_teacher_label(candidate_row_for_label(row), None)
        if resolved is None:
            continue
        teacher_score, policy = resolved
        row["teacher_score"] = float(teacher_score)
        row["label_policy"] = policy
        loaded.append(row)
    return loaded


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


def load_effective_profile(memory: MemoryManager) -> OnionProfile:
    soul_data = memory.get_layer("soul").data
    if not soul_data:
        raise RuntimeError("Soul profile has not been initialized yet.")
    profile = OnionProfile.from_dict(soul_data)
    return apply_overrides(profile, memory.load_profile_overrides())


def _rate(hits: int | float | None, total: int | None) -> str:
    if hits is None or total is None or total <= 0:
        return "n/a"
    return f"{float(hits) / float(total):.3f} ({int(hits)}/{int(total)})"


def _fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to wait between successful chunks.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eligible pool / sample stats without calling the LLM.",
    )
    parser.add_argument(
        "--require-snapshot",
        action="store_true",
        help="Skip rows that have no verified evaluation_context_snapshots payload.",
    )
    return parser.parse_args(argv)


def _print_stats(stats: dict[str, Any]) -> None:
    compared = int(stats["compared"])
    agree = int(stats["agree"])
    print(f"  compared              {compared}")
    print(f"  admission agreement   {_rate(agree, compared)}")
    print(f"  FPR (orig y=0)        {_fmt_optional(stats['fpr'])}")
    print(f"  FNR (orig y=1)        {_fmt_optional(stats['fnr'])}")
    print(f"  MAE (raw score)       {_fmt_optional(stats['mae'])}")
    print(f"  Spearman rho          {_fmt_optional(stats['spearman'])}")
    near = stats.get("near_threshold") or {}
    print(
        "  near-threshold band    "
        f"{_rate(near.get('agree'), near.get('n'))}"
        f"  (|orig-floor|<={NEAR_THRESHOLD_BAND})"
    )
    by_platform = stats.get("by_platform") or {}
    if by_platform:
        print("  by platform:")
        for platform in sorted(by_platform):
            bucket = by_platform[platform]
            n = int(bucket["n"])
            print(
                f"    {platform:12s} n={n:3d}  agree={_rate(bucket['agree'], n)}"
                f"  fp={bucket['fp']} fn={bucket['fn']}"
            )
    print(f"  vs ML@0.70 / @0.50    {ML_AGREEMENT_AT_070:.3f} / {ML_AGREEMENT_AT_050:.3f}")
    print(f"  vs S1.5 agreement     {S15_AGREEMENT_GATE:.2f} (not a merge gate)")


async def _eval_chunks(
    engine: ContentDiscoveryEngine,
    profile: OnionProfile,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    chunk_sleep: float,
    expected_identity: str,
    snapshot: EvaluationContextSnapshot | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Evaluate in outer chunks so a later 429 still keeps earlier pairs.

    ``snapshot`` is bound for the whole group so prompt, cache key, and
    negatives stay consistent. ``None`` lets the engine freeze the current
    live compact profile (legacy rows). Does not persist candidate scores.
    """

    previous = getattr(engine, "_evaluation_context_override", None)
    engine._evaluation_context_override = snapshot
    try:
        return await _eval_chunks_bound(
            engine,
            profile,
            rows,
            batch_size=batch_size,
            chunk_sleep=chunk_sleep,
            expected_identity=expected_identity,
        )
    finally:
        engine._evaluation_context_override = previous


async def _eval_chunks_bound(
    engine: ContentDiscoveryEngine,
    profile: OnionProfile,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    chunk_sleep: float,
    expected_identity: str,
) -> tuple[list[dict[str, Any]], str]:
    pairs_all: list[dict[str, Any]] = []
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
                break
        if failed:
            return pairs_all, stop_reason
        chunk_pairs, identity_reason = pair_replay(
            chunk_rows,
            contents,
            expected_identity=expected_identity,
        )
        if identity_reason:
            stop_reason = identity_reason
            print(
                f"  identity guard items {start + 1}-{start + len(chunk_rows)}: "
                f"{identity_reason} (no silent provider mix; keeping earlier chunks)"
            )
            return pairs_all, stop_reason
        pairs_all.extend(chunk_pairs)
        done = start + len(chunk_rows)
        print(
            f"  chunk {start + 1}-{done}: compared {len(chunk_pairs)}/{len(chunk_rows)}"
            f" total_compared={len(pairs_all)}"
        )
        if chunk_sleep > 0 and done < len(rows):
            await asyncio.sleep(chunk_sleep)
    return pairs_all, stop_reason


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

    database = Database(db_path)
    database.initialize()
    try:
        return _run_probe(
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


def _run_probe(
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
    pool = load_teacher_allowlist_rows(database)
    eligible, filter_counts = filter_rows_for_instance(
        pool,
        provider_type=provider_type,
        model=model,
    )
    sample = stratified_sample(
        eligible,
        limit=max(1, int(args.limit)),
        seed=int(args.seed),
    )
    replay_groups, context_counts = prepare_replay_groups(
        sample,
        database,
        require_snapshot=bool(args.require_snapshot),
    )
    print("=== Wave 1 teacher self-consistency probe ===")
    print(f"  config                {config_path}")
    print(f"  db                    {db_path}")
    print(f"  pinned instance       {instance_id}")
    print(f"  expected identity     {expected_identity}")
    print("  fallback              none (custom_chain + fallback_order pin)")
    print(f"  allowlist             {filter_counts['allowlist']}")
    print(f"  dropped model         {filter_counts['dropped_model_mismatch']}")
    print(f"  dropped provider      {filter_counts['dropped_provider_mismatch']}")
    print(f"  eligible (exact)      {filter_counts['kept']}")
    print(f"  sample                {len(sample)} (limit={args.limit} seed={args.seed})")
    platform_counts = Counter(str(row.get("source_platform") or "?") for row in sample)
    print(f"  sample platforms      {dict(platform_counts)}")
    print(f"  eval-context groups   {context_counts['groups']}")
    print(
        "  snapshot hits/misses  "
        f"{context_counts['snapshot_hits']}/{context_counts['snapshot_misses']}"
    )
    if context_counts["skipped_no_snapshot"]:
        print(
            f"  skipped no-snapshot   {context_counts['skipped_no_snapshot']} (--require-snapshot)"
        )
    print(
        "  note                  snapshot replay when present; legacy empty "
        "digest uses current profile (understates the ceiling). Does not "
        "backfill digests onto old candidate rows."
    )
    if not sample:
        print("  nothing to replay; no exact teacher_model match")
        return 2
    if not replay_groups:
        print("  nothing to replay; no snapshot-backed rows")
        return 2
    if args.dry_run:
        print("  dry-run               skip LLM")
        return 0

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
    print(f"  default_provider      {registry.default_provider}")
    print(f"  fallback_order        {list(registry._fallback_order())}")
    print("=== calling discovery.evaluate_batch (no persist) ===")
    pairs: list[dict[str, Any]] = []
    stop_reason = ""
    for snapshot, group_rows in replay_groups:
        mode = "snapshot" if snapshot is not None else "live-profile"
        print(
            f"  context group         {mode} n={len(group_rows)} "
            f"digest={(snapshot.profile_digest[:8] + '…') if snapshot else '(legacy)'}"
        )
        group_pairs, group_stop = asyncio.run(
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
        pairs.extend(group_pairs)
        if group_stop:
            stop_reason = group_stop
            break
    if stop_reason:
        print(f"  stopped early: {stop_reason}")

    stats = admission_agreement_stats(pairs)
    snapshot = database.query_llm_usage_since_id(since_id=usage_before)
    eval_usage = _usage_for_caller(snapshot, EVAL_CALLER)
    prompt = int(eval_usage.get("prompt_tokens") or 0)
    completion = int(eval_usage.get("completion_tokens") or 0)
    cached = int(eval_usage.get("cached_input_tokens") or 0)
    calls = int(eval_usage.get("calls") or 0)
    cost = float(eval_usage.get("cost_cny") or 0.0)
    compared = int(stats["compared"])
    print("=== replay cost ===")
    print(f"  calls                 {calls}")
    print(f"  prompt/completion     {prompt} / {completion}")
    print(f"  cached_input          {cached}")
    print(f"  cost_cny              {cost:.4f}")
    print("=== teacher self-consistency ===")
    _print_stats(stats)

    artifact = {
        "instance_id": instance_id,
        "expected_identity": expected_identity,
        "provider_type": provider_type,
        "model": model,
        "seed": int(args.seed),
        "limit": int(args.limit),
        "batch_size": int(args.batch_size),
        "filter_counts": dict(filter_counts),
        "context_counts": dict(context_counts),
        "sample_platforms": dict(platform_counts),
        "stop_reason": stop_reason,
        "usage": {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_input_tokens": cached,
            "cost_cny": cost,
        },
        "stats": stats,
        "pairs": pairs,
        "comparisons": {
            "ml_agreement_at_0_70": ML_AGREEMENT_AT_070,
            "ml_agreement_at_0_50": ML_AGREEMENT_AT_050,
            "s1_5_agreement_gate": S15_AGREEMENT_GATE,
        },
        "notes": [
            "Replay uses evaluation_context_snapshots when profile_digest is present.",
            "Legacy empty-digest rows fall back to the current effective profile.",
            "Mixed (profile_digest, negative_digest) pairs never share an eval batch.",
            "Original rows required exact teacher_model == expected_identity.",
            "Replay chunks whose stamped identity differed were discarded, not mixed.",
            "discovery_candidates scores and digests were not updated.",
        ],
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"  artifact              {out_path}")
    if compared <= 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
