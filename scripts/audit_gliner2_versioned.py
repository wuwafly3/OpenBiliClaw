"""Audit pass: re-check every ``versioned`` label for over-application noise.

The teacher historically over-applied ``versioned`` to content that merely
*mentions* a named object (game / software / model / device) even when the
content's value is timeless (design analysis, aesthetics, emotions, generic
tips). This pass re-runs the teacher over rows currently labeled
``versioned`` with a targeted audit instruction, and rewrites the temporal
annotation in place via
``Database.update_discovery_candidate_temporal_relabel`` (conservative
guard: neutral / low-confidence / invalid results keep the old label).

Reuses the relabel runner's plumbing: ``--instance`` pin, proxy-following
``set_outbound_proxy`` sync, ``--eval-max-tokens`` output budget override.

Usage::

    uv run --extra dev python scripts/audit_gliner2_versioned.py \\
        --config E:/otherproject/OpenBiliClaw/config.toml \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --dry-run         # print scope without LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_CONFIG = LIVE_ROOT / "config.toml"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_BATCH_SIZE = 30
# Route the audit through the "evaluation" bucket so --instance pinning applies
# (unknown callers fall to the default chain / "maintenance" bucket).
AUDIT_CALLER = "discovery.evaluate_batch"

from openbiliclaw.config import (  # noqa: E402
    llm_concurrency_from_config,
    load_config,
)
from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.llm.json_utils import extract_llm_json_object  # noqa: E402
from openbiliclaw.llm.registry import build_llm_registry  # noqa: E402
from openbiliclaw.llm.service import LLMService, ModuleOverride  # noqa: E402
from openbiliclaw.llm.usage_recorder import UsageRecorder  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402

if TYPE_CHECKING:
    from openbiliclaw.llm.base import LLMResponse

# Static audit system prompt (script-local, not part of the production eval
# cache namespace).
_AUDIT_SYSTEM_PROMPT = (
    "<task>\n"
    "你是内容时效分类复核员。以下候选当前都被标为 versioned,请逐条复核 "
    "temporal_class 是否过度应用。\n"
    "</task>\n\n"
    "<rules>\n"
    "1. 提到具名对象(游戏/软件/模型/设备/工具/IP)本身**不足以**判 versioned;"
    "只有当内容的**核心价值随该对象的具体版本更新而衰减**时才判 versioned"
    "(如:绑定版本号的教程/攻略、设备型号专属评测、版本更新说明)。\n"
    "2. 内容是设计分析、审美展示、情绪表达、角色鉴赏、通用技巧、幕后解析——"
    "即使提到具名对象/角色/游戏/软件——判 evergreen(价值不依赖版本)。\n"
    "3. 内容价值依赖近期语境(新发布、热点讨论、当下流行)判 current;"
    "纯回顾/档案判 historical;突发/实时判 breaking。\n"
    "4. 输出严格 JSON,顶层只含 results 数组,每项原样带回输入 id,包含 "
    "temporal_class / temporal_confidence(0-1) / temporal_reason(≤30字) / "
    "temporal_validity_mode / temporal_valid_until / temporal_scope / "
    "temporal_state / temporal_evidence。\n"
    "5. evergreen/historical 用 validity_mode=none、scope=none、state=unknown、"
    "evidence=\"\"。versioned/current/breaking 必须用非 none mode"
    "(通常 freshness_only + scope=core/hook + evidence=输入原文逐字摘录);"
    "只有明确写出截止时间才用 explicit_deadline。\n"
    "</rules>"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--instance",
        default="openai-4",
        help="Pin audit routing to one LLM instance (default openai-4).",
    )
    parser.add_argument(
        "--eval-max-tokens",
        type=int,
        default=16384,
        help="Output budget for the audit call (engine default 4096 truncates "
        "on DeepSeek relays that run thinking).",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all versioned rows")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected scope and exit without calling the LLM.",
    )
    return parser.parse_args(argv)


def load_versioned_rows(database: Database) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, source_strategy, title, description,
               temporal_class
        FROM discovery_candidates
        WHERE temporal_class = 'versioned'
          AND score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY evaluated_at DESC, id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    return [dict(row) for row in cursor.fetchall()]


def build_audit_messages(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = [
        {
            "id": str(row["id"]),
            "title": str(row.get("title") or ""),
            "description": str(row.get("description") or "")[:400],
        }
        for row in rows
    ]
    user = (
        "以下是待复核候选(当前标为 versioned):\n"
        + json.dumps(items, ensure_ascii=False, sort_keys=True)
    )
    return [
        {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_audit_results(
    response: LLMResponse,
    rows: list[dict[str, Any]],
    *,
    evaluated_at: str,
) -> tuple[list[dict[str, Any]], int]:
    """Map the audit response back onto candidate rows.

    The model commonly emits ``null`` for empty text fields and pairs
    ``freshness_only`` with ``state="active"``; both violate the strict v2
    temporal contract and would be neutralized by the storage guard. Coerce
    them here so valid audit verdicts actually persist.
    """
    payload = extract_llm_json_object(str(getattr(response, "content", "")).strip())
    if not isinstance(payload, dict):
        return [], 0
    results = payload.get("results")
    if not isinstance(results, list):
        return [], 0
    by_id = {str(row["id"]): row for row in rows}
    evaluations: list[dict[str, Any]] = []
    matched = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid not in by_id:
            continue
        matched += 1
        mode = _audit_text(item.get("temporal_validity_mode", "none")).lower()
        state = _audit_text(item.get("temporal_state", "unknown")).lower()
        if mode in {"none", "freshness_only", "explicit_deadline"}:
            state = "unknown"
        try:
            confidence = float(item.get("temporal_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evaluations.append(
            {
                "candidate_id": int(cid),
                "temporal_class": _audit_text(item.get("temporal_class", "unknown")),
                "temporal_confidence": confidence,
                "temporal_reason": _audit_text(item.get("temporal_reason", "")),
                "temporal_policy_version": "v2",
                "temporal_validity_mode": mode,
                "temporal_valid_until": _audit_text(item.get("temporal_valid_until", "")),
                "temporal_scope": _audit_text(item.get("temporal_scope", "none")).lower(),
                "temporal_state": state,
                "temporal_evidence": _audit_text(item.get("temporal_evidence", "")),
                "temporal_next_review_at": "",
                "temporal_evaluated_at": evaluated_at,
                "temporal_evidence_complete": True,
            }
        )
    return evaluations, matched


def _audit_text(value: object) -> str:
    return "" if value is None else str(value)


def _evaluated_at_now() -> str:
    return (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _pin_instance_overrides(instance_id: str) -> dict[str, ModuleOverride]:
    chain = (instance_id.strip().lower(),)
    pinned = ModuleOverride(chain=chain, custom_chain=True)
    return {"evaluation": pinned, "discovery": pinned}


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
    text = _exception_text(exc)
    markers = (
        "502", "503", "504", "500 ", "server error", "bad gateway",
        "service unavailable", "connection", "timed out", "timeout",
        "temporarily unavailable", "empty response", "no provider",
    )
    return any(marker in text for marker in markers)


def _is_hard_quota(exc: BaseException) -> bool:
    text = _exception_text(exc)
    markers = (
        "monthly usage limit", "allocated quota exceeded", "insufficient_quota",
        "insufficient balance", "creditserror", "region_blocked",
    )
    return any(marker in text for marker in markers)


def _usage_for_caller(snapshot: dict[str, Any], caller: str) -> dict[str, Any]:
    for row in snapshot.get("by_caller") or []:
        if str(row.get("caller") or "") == caller:
            return dict(row)
    return {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cached_input_tokens": 0, "cost_cny": 0.0,
    }


async def _audit_chunks(
    llm_service: LLMService,
    database: Database,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    chunk_sleep: float,
) -> tuple[int, Counter[tuple[str, str]], str]:
    from openbiliclaw.llm.task_options import without_core_memory_kwargs

    complete_structured = llm_service.complete_structured_task
    updated_total = 0
    transitions: Counter[tuple[str, str]] = Counter()
    stop_reason = ""
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        messages = build_audit_messages(chunk_rows)
        delay = 15.0
        failed = False
        response: LLMResponse | None = None
        for attempt in range(5):
            try:
                kwargs = {
                    "system_instruction": messages[0]["content"],
                    "user_input": messages[1]["content"],
                    "caller": AUDIT_CALLER,
                }
                kwargs.update(without_core_memory_kwargs(complete_structured))
                response = await complete_structured(**kwargs)
                break
            except Exception as exc:
                if _is_hard_quota(exc):
                    stop_reason = f"{type(exc).__name__}: {exc}"
                    print(f"  hard quota at items {start + 1}-{start + len(chunk_rows)}; keeping earlier chunks")
                    print(f"    {stop_reason}")
                    failed = True
                    break
                if (_is_rate_limit(exc) or _is_transient_server_error(exc)) and attempt < 4:
                    kind = "rate-limited" if _is_rate_limit(exc) else "transient"
                    print(f"  {kind} items {start + 1}-{start + len(chunk_rows)} attempt {attempt + 1}; sleep {delay:.0f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                stop_reason = f"{type(exc).__name__}: {exc}"
                print(f"  chunk failed items {start + 1}-{start + len(chunk_rows)}: {stop_reason}")
                failed = True
                break
        if failed or response is None:
            return updated_total, transitions, stop_reason

        evaluations, matched = parse_audit_results(
            response, chunk_rows, evaluated_at=_evaluated_at_now()
        )
        if matched == 0:
            stop_reason = f"no audit results matched items {start + 1}-{start + len(chunk_rows)}"
            print(f"  {stop_reason}")
            return updated_total, transitions, stop_reason
        by_id = {int(ev["candidate_id"]): ev for ev in evaluations}
        before = {int(row["id"]): str(row.get("temporal_class") or "") for row in chunk_rows}
        updated = database.update_discovery_candidate_temporal_relabel(
            [by_id[cid] for cid in before if cid in by_id]
        )
        updated_total += updated
        # Reload AFTER the write to count real transitions.
        dur_rows = database.get_discovery_candidates_by_ids(
            [int(row["id"]) for row in chunk_rows]
        )
        durable = {int(r["id"]): r for r in dur_rows}
        for cid, old in before.items():
            new = str(durable.get(cid, {}).get("temporal_class") or old)
            if new != old:
                transitions[(old, new)] += 1
        done = start + len(chunk_rows)
        print(f"  chunk {start + 1}-{done}: matched {matched}/{len(chunk_rows)} total_updated={updated_total}")
        if chunk_sleep > 0 and done < len(rows):
            await asyncio.sleep(chunk_sleep)
    return updated_total, transitions, stop_reason


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config_path = args.config.expanduser().resolve()
    db_path = args.db.expanduser().resolve()
    if not config_path.is_file() or not db_path.is_file():
        print(f"config/db not found: {config_path} / {db_path}")
        return 1
    os.environ.setdefault("OPENBILICLAW_PROJECT_ROOT", str(config_path.parent))

    config = load_config(config_path)
    from openbiliclaw.network import set_outbound_proxy

    set_outbound_proxy(config.network.proxy, mode=config.network.mode)
    database = Database(db_path)
    database.initialize()
    try:
        rows = load_versioned_rows(database)
        if args.limit and args.limit > 0:
            rows = rows[: int(args.limit)]
        platforms = Counter(str(row.get("source_platform") or "?") for row in rows)
        print("=== GLiNER2 versioned label audit ===")
        print(f"  config                {config_path}")
        print(f"  db                    {db_path}")
        print(f"  versioned rows        {len(rows)}")
        print(f"  platforms             {dict(platforms.most_common())}")
        if not rows:
            print("  nothing to audit")
            return 0
        if args.dry_run:
            print("  dry-run               skip LLM; no DB writes")
            return 0

        usage_before = database.max_llm_usage_id()
        instance_id = str(args.instance or "").strip().lower()
        registry = build_llm_registry(config)
        llm = LLMService(
            registry=registry,
            memory=None,  # type: ignore[arg-type]
            usage_recorder=UsageRecorder(sink=database),
            module_overrides=_pin_instance_overrides(instance_id),
            concurrency=llm_concurrency_from_config(config),
        )
        if int(args.eval_max_tokens or 0) > 0:
            _override_max_tokens(llm, int(args.eval_max_tokens))
        print(f"  pinned instance       {instance_id}")
        print(f"  eval_max_tokens       {int(args.eval_max_tokens)} (override)")
        print("=== calling versioned audit ===")
        updated, transitions, stop_reason = asyncio.run(
            _audit_chunks(
                llm,
                database,
                rows,
                batch_size=max(1, int(args.batch_size)),
                chunk_sleep=max(0.0, float(args.sleep)),
            )
        )
        if stop_reason:
            print(f"  stopped early: {stop_reason}")

        usage = database.query_llm_usage_since_id(since_id=usage_before)
        audit_usage = _usage_for_caller(usage, AUDIT_CALLER)
        print("=== audit summary ===")
        print(f"  rows evaluated        {len(rows)}")
        print(f"  rows touched          {updated}")
        print(f"  classes changed       {sum(transitions.values())}")
        if transitions:
            print("  transitions:")
            for (old, new), n in transitions.most_common():
                print(f"    {old:12s} -> {new:12s} {n}")
        final_counts = Counter()
        for rid in [int(row["id"]) for row in rows]:
            dur = database.get_discovery_candidates_by_ids([rid])
            if dur:
                final_counts[str(dur[0].get("temporal_class") or "")] += 1
        print(f"  final class dist      {dict(final_counts.most_common())}")
        print(f"  calls                 {int(audit_usage.get('calls') or 0)}")
        print(f"  cost_cny              {float(audit_usage.get('cost_cny') or 0.0):.4f}")
        return 0
    finally:
        database.close()


def _override_max_tokens(llm_service: Any, max_tokens: int) -> None:
    original = llm_service.complete_structured_task

    async def wrapped(**kwargs: Any) -> Any:
        kwargs["max_tokens"] = max_tokens
        return await original(**kwargs)

    llm_service.complete_structured_task = wrapped  # type: ignore[method-assign]


if __name__ == "__main__":
    raise SystemExit(main())
