"""Cross-teacher-model replay of the frozen gate-contract snapshot pool.

Task 4b of the 2026-08-21 gate/ranker separation plan: replay the same
new-contract snapshot cohort that produced
``gate_contract_relabel_20260824T030808Z.jsonl`` (918 rows, no recent layer)
through a pinned non-default LLM instance (e.g. ``router`` / ``gpt-5.6-sol``) and compare its
admission labels against the ``openai/deepseek-v4-flash`` first draw.

One process = one draw. Run twice with different ``--tag`` values to get the
cross model's own two-draw self-consistency:

    uv run --extra dev python scripts/ml_gate_contract_cross_model_probe.py \
        --config E:/otherproject/OpenBiliClaw/config.toml \
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \
        --instance router --tag draw1 --plain-ua

Relay instances (Router and friends) need ``--plain-ua`` because their WAF
blocks the OpenAI SDK's default User-Agent, and they get a minimal request
body: only ``model`` + prompt + an output cap. ``api_flavor = "responses"`` is
the surface such relays document for reasoning-first models; the probe pins
gpt-5.6-sol over ``/v1/responses`` that way.

Replay binds the labeling-time ``evaluation_context_snapshots`` payload per
``(profile_digest, negative_digest)`` group; mixed digest pairs never share a
batch. Only ``llm_usage`` rows are recorded — ``discovery_candidates`` scores
and digests are not updated. The per-process eval cache means two separate
invocations are genuine independent draws.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_SOURCE = (
    LIVE_ROOT / "data" / "ml_artifacts" / "gate_contract_relabel_20260824T030808Z.jsonl"
)
DEFAULT_INSTANCE = "router"
DEFAULT_BATCH_SIZE = 45
EVAL_CALLER = "discovery.evaluate_batch"

from export_ranking_dataset import effective_admission_threshold  # noqa: E402
from ml_teacher_self_consistency_probe import (  # noqa: E402
    _eval_chunks,
    _usage_for_caller,
    admission_agreement_stats,
    eval_context_key,
    expected_teacher_identity,
    is_near_threshold,
    load_effective_profile,  # noqa: E402
    pin_instance_overrides,
    resolve_llm_instance,
    resolve_replay_snapshot,
)

from openbiliclaw.config import (  # noqa: E402
    load_config,
)
from openbiliclaw.discovery.engine import ContentDiscoveryEngine  # noqa: E402
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.memory.manager import MemoryManager  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--instance",
        default=DEFAULT_INSTANCE,
        help=f"config.toml LLM instance id to pin (default: {DEFAULT_INSTANCE}).",
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Override the pinned instance's model for this run only (e.g. "
            "claude-sonnet-5 on a relay instance configured for another slug). "
            "config.toml stays untouched; the teacher stamp, the identity "
            "guard and the output file names use the override."
        ),
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--tag",
        default="draw1",
        help="Output suffix, e.g. draw1 / draw2. One process = one draw.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to wait between successful chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only replay the first N rows of the cohort (0 = all). Smoke test only.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=LIVE_ROOT / "data" / "ml_artifacts",
    )
    parser.add_argument(
        "--plain-ua",
        action="store_true",
        help=(
            "Rebuild the pinned provider's client with a plain User-Agent. "
            "Some free relays (e.g. nodelocfree) block the OpenAI SDK's "
            "default UA at their WAF."
        ),
    )
    parser.add_argument(
        "--full-params",
        action="store_true",
        help=(
            "Send the provider's full request body instead of the minimal one. "
            "Off by default: relays answering for reasoning-first models "
            "(gpt-5 family) reject 'temperature' and refuse the structured-"
            "output knob, and a chunk that carries a knob the next chunk "
            "doesn't is not one measurement cohort."
        ),
    )
    parser.add_argument(
        "--no-model-alias",
        action="store_true",
        help=(
            "Do not normalize a relay's echoed model name back to the "
            "configured instance model. By default an echo like "
            "'openai:gpt-5.6-sol:flex' is aliased to 'gpt-5.6-sol' so the "
            "teacher-identity guard sees the instance that was actually "
            "pinned; the raw echoes are still recorded in the meta file."
        ),
    )
    parser.add_argument(
        "--require-content",
        action="store_true",
        help=(
            "Restrict the cohort to rows whose stored discovery_candidates row "
            "has non-empty body_text or description. Cross-model replays on "
            "title-only rows measure a refusal policy (external judges answer "
            "score 0 / 内容未知), not taste; this flag isolates the rows where "
            "the judge actually had something to read."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cohort / group stats without calling the LLM.",
    )
    return parser.parse_args(argv)


def load_source_cohort(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            strategy = str(raw.get("source_strategy") or "")
            floor = effective_admission_threshold(strategy)
            first_score = float(raw["new_llm_score_raw"])
            first_y = int(first_score >= floor)
            if first_y != int(raw["new_y"]):
                raise ValueError(
                    f"source row {raw['candidate_id']}: new_y {raw['new_y']} != "
                    f"recomputed {first_y} at floor {floor}"
                )
            rows.append(
                {
                    # Keys aliased so the imported _eval_chunks / pair_replay
                    # machinery treats the deepseek first draw as the reference.
                    "id": int(raw["candidate_id"]),
                    "candidate_id": int(raw["candidate_id"]),
                    "teacher_score": first_score,
                    "orig_y": first_y,
                    "first_score": first_score,
                    "first_y": first_y,
                    "old_y": int(raw["old_y"]),
                    "profile_digest": str(raw["new_profile_digest"] or ""),
                    "negative_digest": str(raw["new_negative_digest"] or ""),
                    "source_platform": str(raw.get("source_platform") or "unknown"),
                    "platform": str(raw.get("source_platform") or "unknown"),
                    "source_strategy": strategy,
                    "strategy": strategy,
                    "floor": floor,
                }
            )
    return rows


def fetch_candidate_rows(
    database: Database,
    candidate_ids: list[int],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for start in range(0, len(candidate_ids), 500):
        chunk = candidate_ids[start : start + 500]
        placeholders = ", ".join("?" for _ in chunk)
        cursor = database.conn.execute(
            f"""
            SELECT id, source_platform, source_strategy, content_type, title,
                   body_text, bvid, content_id, description, published_at, duration
            FROM discovery_candidates
            WHERE id IN ({placeholders})
            """,
            tuple(chunk),
        )
        for raw in cursor.fetchall():
            row = dict(raw)
            rows[int(row["id"])] = row
    return rows


# Only these request parameters reach the relay. Reasoning-first models reject
# ``temperature`` outright, Router answers 400 for a ``text.format`` /
# ``response_format`` knob it cannot translate for this route, and neither
# matters here: the teacher prompt already demands a JSON array and the replay
# compares judgments, not sampling. Trimming once up front also keeps every
# chunk of a draw byte-identical, which a send-then-retry-after-400 dance does
# not (the first chunk would have carried the knob, later ones would not).
_MINIMAL_REQUEST_KEYS: dict[str, tuple[str, ...]] = {
    "_create_chat_completion": ("model", "messages", "max_tokens", "extra_headers"),
    "_create_response": ("model", "input", "instructions", "max_output_tokens", "extra_headers"),
}


def _install_minimal_request_shim(
    registry: object,
    instance_id: str,
    *,
    alias_model: bool,
    observed: dict[str, list[str]],
) -> list[str]:
    """Pin the relay request to the parameters the evaluation contract needs.

    Probe-only monkeypatch on whichever send hook the instance's ``api_flavor``
    uses; the runtime provider is untouched. ``alias_model`` additionally
    rewrites a relay's echoed model name back to the configured instance model
    when the echo only carries a routing suffix (``openai:gpt-5.6-sol:flex``),
    so ``_response_teacher_identity`` stamps the pinned instance instead of
    tripping the identity guard. Raw echoes land in
    ``observed["upstream_models"]`` for the audit trail.
    """

    provider = getattr(registry, "_providers", {}).get(instance_id)
    if provider is None:
        return []
    if not any(callable(getattr(provider, attr, None)) for attr in _MINIMAL_REQUEST_KEYS):
        return []

    configured_model = str(getattr(provider, "_model", "") or "").strip()
    stripped: list[str] = []

    def _wrap(send_attr: str) -> None:
        original = getattr(provider, send_attr)
        keep = _MINIMAL_REQUEST_KEYS[send_attr]

        async def _send_minimal(**kwargs: object) -> object:
            for name in [key for key in kwargs if key not in keep]:
                kwargs.pop(name)
                if name not in stripped:
                    stripped.append(name)
            return await original(**kwargs)

        setattr(provider, send_attr, _send_minimal)

    for attr in _MINIMAL_REQUEST_KEYS:
        if callable(getattr(provider, attr, None)):
            _wrap(attr)

    if alias_model and configured_model:
        original_complete = provider.complete

        async def complete(*args: object, **kwargs: object) -> Any:
            response = await original_complete(*args, **kwargs)
            reported = str(getattr(response, "model", "") or "").strip()
            if not reported or reported == configured_model:
                return response
            if configured_model in reported:
                if reported not in observed["upstream_models"]:
                    observed["upstream_models"].append(reported)
                response.model = configured_model
            return response

        provider.complete = complete  # type: ignore[method-assign]

    return stripped


def _rebuild_client_with_plain_ua(registry: object, instance_id: str) -> bool:
    """Swap the pinned provider's SDK client for one with a plain User-Agent.

    Some free relays block the OpenAI SDK's default UA at their WAF (HTTP 403
    "Your request was blocked") while accepting the same request under any
    non-SDK UA. Probe-only workaround; the runtime provider is untouched.
    """

    provider = getattr(registry, "_providers", {}).get(instance_id)
    old_client = getattr(provider, "_client", None)
    if provider is None or old_client is None:
        return False
    from openai import AsyncOpenAI

    provider._client = AsyncOpenAI(
        api_key=old_client.api_key,
        base_url=old_client.base_url,
        max_retries=0,
        timeout=float(getattr(provider, "_timeout", 120.0)),
        default_headers={"User-Agent": "openbiliclaw-probe/1.0"},
    )
    return True


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
    source_path = args.source.expanduser().resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}")
        return 1
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 1
    if not source_path.is_file():
        print(f"source jsonl not found: {source_path}")
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
    override_model = str(args.model or "").strip()
    if override_model and override_model != model:
        # Run-scoped only: patch the loaded config object, never config.toml.
        instances = getattr(getattr(config, "llm", None), "instances", None)
        if isinstance(instances, dict):
            for key, entry in instances.items():
                if str(key).strip().lower() == instance_id:
                    entry.model = override_model
        model = override_model
        args.tag = f"{args.tag}-{override_model}"
        args.tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(args.tag))
        print(f"  model override          {instance_id} -> {override_model}")
    if not provider_type or not model:
        print(f"LLM instance {instance_id} missing provider_type or model")
        return 1
    expected_identity = expected_teacher_identity(provider_type=provider_type, model=model)

    cohort = load_source_cohort(source_path)

    database = Database(db_path)
    database.initialize()
    try:
        return _run_probe(
            args,
            config=config,
            config_path=config_path,
            source_path=source_path,
            database=database,
            instance_id=instance_id,
            provider_type=provider_type,
            model=model,
            expected_identity=expected_identity,
            cohort=cohort,
        )
    finally:
        database.close()


def _run_probe(
    args: argparse.Namespace,
    *,
    config: object,
    config_path: Path,
    source_path: Path,
    database: Database,
    instance_id: str,
    provider_type: str,
    model: str,
    expected_identity: str,
    cohort: list[dict[str, Any]],
) -> int:
    candidate_rows = fetch_candidate_rows(database, [r["candidate_id"] for r in cohort])
    missing = [r["candidate_id"] for r in cohort if r["candidate_id"] not in candidate_rows]
    if missing:
        print(f"  candidates missing from db: {len(missing)} (e.g. {missing[:5]})")
        return 1

    # The source JSONL carries labels, digests and floors — not the text the
    # teacher read. Rebuild the prompt payload from discovery_candidates so the
    # replay hands the cross model the same fields the self-consistency probe
    # hands the pinned teacher (``row_to_content`` reads exactly these keys).
    # Without this join every item renders as ``{"id":"0","title":""}`` and the
    # replay measures a refusal on empty input, not a difference in taste.
    for row in cohort:
        stored = candidate_rows[int(row["candidate_id"])]
        for field in (
            "content_type",
            "title",
            "body_text",
            "bvid",
            "content_id",
            "description",
            "published_at",
            "duration",
        ):
            row[field] = stored.get(field)

    textless = [
        int(row["candidate_id"])
        for row in cohort
        if not str(row.get("title") or "").strip()
        and not str(row.get("body_text") or "").strip()
        and not str(row.get("description") or "").strip()
    ]
    print(f"  rows with no prompt text {len(textless)}/{len(cohort)}")
    if len(textless) == len(cohort):
        print("  every row is textless; refusing to bill a model for empty input")
        return 2
    if bool(args.require_content):
        before = len(cohort)

        def _has_text(row: dict[str, Any]) -> bool:
            return bool(
                str(row.get("body_text") or "").strip() or str(row.get("description") or "").strip()
            )

        cohort = [row for row in cohort if _has_text(row)]
        print(
            f"  require-content         kept {len(cohort)}/{before} "
            f"(dropped {before - len(cohort)} title-only rows)"
        )
        if not cohort:
            print("  nothing to replay")
            return 2
    if args.limit and args.limit > 0:
        # Sliced after the content filter so a smoke test sees rich rows.
        cohort = cohort[: max(1, int(args.limit))]

    # Group by (profile_digest, negative_digest) in first-seen order.
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for row in cohort:
        key = eval_context_key(row)
        if key not in groups:
            order.append(key)
        groups[key].append(row)

    replay: list[tuple[Any, list[dict[str, Any]]]] = []
    snapshot_hits = snapshot_misses = 0
    for key in order:
        snapshot = resolve_replay_snapshot(database, profile_digest=key[0], negative_digest=key[1])
        if snapshot is None:
            snapshot_misses += len(groups[key])
            continue
        snapshot_hits += len(groups[key])
        replay.append((snapshot, groups[key]))

    print("=== gate contract cross-model replay ===")
    print(f"  config                {config_path}")
    print(f"  source cohort         {source_path.name} rows={len(cohort)}")
    print(f"  pinned instance       {instance_id}")
    print(f"  expected identity     {expected_identity}")
    print(f"  digest groups         {len(order)}")
    print(f"  snapshot hits/misses  {snapshot_hits}/{snapshot_misses}")
    if snapshot_misses:
        print("  rows without verified snapshots are skipped (no live fallback)")
    if not replay:
        print("  nothing to replay")
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
    observed: dict[str, list[str]] = {"upstream_models": []}
    stripped_params: list[str] = []
    if not bool(args.full_params):
        stripped_params = _install_minimal_request_shim(
            registry,
            instance_id,
            alias_model=not bool(args.no_model_alias),
            observed=observed,
        )
        print(
            "  minimal-request         installed "
            f"(model-alias={'on' if not args.no_model_alias else 'off'})"
        )
    if bool(args.plain_ua):
        if _rebuild_client_with_plain_ua(registry, instance_id):
            print("  plain-ua               provider client rebuilt (relay WAF workaround)")
        else:
            print("  plain-ua               no SDK client found; skipped")
    llm = LLMService(
        registry=registry,
        memory=memory,
        usage_recorder=UsageRecorder(sink=database),
        module_overrides=pin_instance_overrides(instance_id),
        concurrency=1,  # free relays choke on parallel requests; keep serial
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
    print("=== calling discovery.evaluate_batch (no persist) ===")
    pairs: list[dict[str, Any]] = []
    stop_reason = ""
    for snapshot, group_rows in replay:
        print(
            f"  context group         snapshot n={len(group_rows)} "
            f"digest={snapshot.profile_digest[:8]}…"
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
    usage_snapshot = database.query_llm_usage_since_id(since_id=usage_before)
    eval_usage = _usage_for_caller(usage_snapshot, EVAL_CALLER)
    usage = {
        "calls": int(eval_usage.get("calls") or 0),
        "prompt_tokens": int(eval_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(eval_usage.get("completion_tokens") or 0),
        "cached_input_tokens": int(eval_usage.get("cached_input_tokens") or 0),
        "cost_cny": float(eval_usage.get("cost_cny") or 0.0),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"gate_contract_cross_model_{instance_id}_{args.tag}_{stamp}.jsonl"
    pair_by_id = {int(item["candidate_id"]): item for item in pairs}
    written = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in cohort:
            item = pair_by_id.get(row["candidate_id"])
            record = {
                "candidate_id": row["candidate_id"],
                "source_platform": row["platform"],
                "source_strategy": row["strategy"],
                "first_llm_score_raw": row["first_score"],
                "first_y": row["first_y"],
                "old_y": row["old_y"],
                "floor": row["floor"],
                "near_threshold": bool(item and item.get("near_threshold"))
                or is_near_threshold(row["first_score"], row["floor"]),
                "replay_llm_score_raw": (item or {}).get("replay_score"),
                "replay_y": (item or {}).get("replay_y"),
                "replay_teacher_model": (item or {}).get("replay_model", ""),
                "agree": (int(item["orig_y"] == item["replay_y"]) if item else None),
                "profile_digest": row["profile_digest"],
                "negative_digest": row["negative_digest"],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    meta = {
        "instance_id": instance_id,
        "expected_identity": expected_identity,
        "provider_type": provider_type,
        "model": model,
        "tag": str(args.tag),
        "source_jsonl": str(source_path),
        "jsonl": str(jsonl_path),
        "filter_counts": {
            "cohort": len(cohort),
            "groups": len(order),
            "snapshot_hits": snapshot_hits,
            "snapshot_misses": snapshot_misses,
        },
        "limit": int(args.limit or 0),
        "require_content": bool(args.require_content),
        "stop_reason": stop_reason,
        "request_shape": {
            "minimal": not bool(args.full_params),
            "model_alias": not bool(args.no_model_alias),
            "stripped_params": stripped_params,
            "upstream_models": observed["upstream_models"],
        },
        "written": written,
        "usage": usage,
        "stats": stats,
        "notes": [
            "One process = one draw; per-process eval cache keeps draws independent.",
            "Reference labels are the deepseek-v4-flash first draw (source_jsonl new_*).",
            "Mixed (profile_digest, negative_digest) pairs never share an eval batch.",
            "Replay chunks whose stamped identity differed were discarded, not mixed.",
            "discovery_candidates scores and digests were not updated.",
        ],
    }
    meta_path = jsonl_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    compared = int(stats["compared"])
    print("=== replay cost ===")
    if not bool(args.full_params):
        print(f"  stripped params        {', '.join(stripped_params) or '(none)'}")
        if observed["upstream_models"]:
            print(f"  upstream model echoes  {', '.join(observed['upstream_models'])}")
    print(f"  calls                 {usage['calls']}")
    print(f"  prompt/completion     {usage['prompt_tokens']} / {usage['completion_tokens']}")
    print(f"  cached_input          {usage['cached_input_tokens']}")
    print(f"  cost_cny              {usage['cost_cny']:.4f}")
    print(f"=== cross-model vs deepseek first draw ({expected_identity}) ===")
    print(f"  compared              {compared}/{len(cohort)}")
    if compared:
        agree = int(stats["agree"])
        print(f"  admission agreement   {agree / compared:.3f} ({agree}/{compared})")
        print(
            f"  FPR (first y=0)       {stats['fpr']:.3f}"
            if stats["fpr"] is not None
            else "  FPR n/a"
        )
        print(
            f"  FNR (first y=1)       {stats['fnr']:.3f}"
            if stats["fnr"] is not None
            else "  FNR n/a"
        )
        print(
            f"  MAE (raw score)       {stats['mae']:.3f}"
            if stats["mae"] is not None
            else "  MAE n/a"
        )
        print(
            f"  Spearman rho          {stats['spearman']:.3f}"
            if stats["spearman"] is not None
            else "  Spearman n/a"
        )
        near = stats.get("near_threshold") or {}
        if near.get("n"):
            print(
                f"  near-threshold band    {near['agree'] / near['n']:.3f} "
                f"({near['agree']}/{near['n']})"
            )
        by_platform = stats.get("by_platform") or {}
        for platform in sorted(by_platform):
            bucket = by_platform[platform]
            n = int(bucket["n"])
            print(
                f"    {platform:12s} n={n:3d}  agree={bucket['agree'] / n:.3f}"
                f"  fp={bucket['fp']} fn={bucket['fn']}"
            )
    print(f"  jsonl                 {jsonl_path}")
    print(f"  meta                  {meta_path}")
    if compared <= 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
