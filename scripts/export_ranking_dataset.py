"""Snapshot the ml-ranking teacher-label dataset out of the live database.

Wave 0 step 3 (spec S0.3): ``evaluator_prefilter_shadow_audit`` is purged on a
30-day / row-cap schedule and ``discovery_candidates`` terminalizes on queue
pressure, so the training dataset must be exported to versioned files before
it ages out. This is the minimal durable version — run it before a
fixed-teacher collection window starts and periodically during it.

Row policy (``resolve_teacher_label``):
- score_source in the LLM-judgment allowlist with llm_score_raw -> that raw
  score (provenance policy; cap-zeroed rows keep their teacher judgment);
- legacy rows (score_source = '', written before provenance shipped):
  non-zero relevance_score is a genuine teacher judgment (caps only zero),
  zero rows are recovered from the shadow audit's at-time llm_score when
  possible, else dropped as ambiguous;
- any other score_source (prefilter/viewed/truncated/...) is dropped.

The frozen binary label follows spec S1.1: y = teacher_score >= per-row
effective admission threshold (exact strategy ``explore`` uses 0.58,
otherwise 0.60 — prefixes such as ``explore-backfill`` stay at the default).
Batch identity
(evaluated_at cluster), teacher_model, and the full shadow-audit aggregates
are exported alongside; title/description/body_text are included because the
export is local training data (stays on this machine) and the text-vector
feature (spec S1.2) needs re-embedding later — the embedding LRU cache hits
only ~12% of historical candidates.

Usage:
    python scripts/export_ranking_dataset.py [db] [--out DIR] [--max-rows N]

Reads the database read-only; writes <DIR>/dataset_<ts>.jsonl + .meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")
DEFAULT_OUT = Path("data/ml_ranking_dataset")

EXPORT_SCHEMA_VERSION = 1
ADMISSION_DEFAULT = 0.60
ADMISSION_EXPLORE = 0.58
# Keep in sync with openbiliclaw.discovery.admission.EXPLORE_STRATEGY.
EXPLORE_STRATEGY = "explore"

EVALUATED_STATUSES = (
    "evaluated",
    "rejected_low_score",
    "cached",
    "rejected_cache_admission",
    "rejected_temporal_stale",
    "rejected_franchise_quota",
)
# Keep in sync with openbiliclaw.discovery.score_source.LLM_JUDGMENT_SCORE_SOURCES
# (duplicated so this script runs without the package installed).
LLM_JUDGMENT_SCORE_SOURCES = frozenset({"llm", "cap_franchise", "cap_style"})

ENGAGEMENT_COLUMNS = (
    "view_count",
    "like_count",
    "favorite_count",
    "collect_count",
    "comment_count",
    "share_count",
    "danmaku_count",
    "reply_count",
    "retweet_count",
    "bookmark_count",
)

CANDIDATE_COLUMNS = (
    "id",
    "candidate_key",
    "status",
    "source_platform",
    "source_strategy",
    "content_type",
    "candidate_tier",
    "title",
    "description",
    "body_text",
    "published_at",
    "evaluated_at",
    "duration",
    "relevance_score",
    "score_source",
    "llm_score_raw",
    "teacher_model",
    "profile_digest",
    "negative_digest",
    "style_key",
    "temporal_class",
    "topic_group",
    "franchise_key",
    *ENGAGEMENT_COLUMNS,
)


def audit_hash(identity: str) -> str:
    payload = f"openbiliclaw:evaluator-prefilter:v1\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def effective_admission_threshold(source_strategy: object) -> float:
    """Per-row gate from spec S1.1; only exact ``explore`` uses the lower floor.

    Matches ``openbiliclaw.discovery.admission.effective_admission_threshold``
    for the default policy floor. A live requested threshold may raise a
    source's floor, but that is not frozen into the export label.
    """

    strategy = str(source_strategy or "").strip().lower()
    return ADMISSION_EXPLORE if strategy == EXPLORE_STRATEGY else ADMISSION_DEFAULT


def teacher_label_select_columns(conn: sqlite3.Connection) -> str:
    """Optional provenance columns, skipped on pre-taxonomy databases."""

    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(discovery_candidates)")}
    extra = [column for column in ("score_source", "llm_score_raw") if column in existing]
    return (", " + ", ".join(extra)) if extra else ""


def candidate_row_for_label(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Normalize a candidate row so ``resolve_teacher_label`` can read it."""

    record = dict(row)
    record.setdefault("score_source", "")
    record.setdefault("llm_score_raw", None)
    return record


def resolve_teacher_label(
    row: dict[str, Any],
    audit_llm_score: float | None,
) -> tuple[float, str] | None:
    """Teacher score for one candidate row plus its provenance policy.

    Returns None when no trustworthy teacher judgment exists.
    """

    score_source = str(row.get("score_source") or "")
    llm_score_raw = row.get("llm_score_raw")
    if score_source in LLM_JUDGMENT_SCORE_SOURCES:
        if llm_score_raw is None:
            return None
        return float(llm_score_raw), "provenance"
    if score_source == "":
        persisted = float(row.get("relevance_score") or 0.0)
        if persisted > 0.0:
            return persisted, "legacy_nonzero"
        if audit_llm_score is not None:
            return float(audit_llm_score), "legacy_audit_recovered"
        return None
    return None


def load_audit_aggregates(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Per-candidate shadow-audit aggregates (the perishable side)."""

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT candidate_hash, similarity, llm_score, would_filter,
               context_class, profile_digest, created_at
        FROM evaluator_prefilter_shadow_audit
        ORDER BY created_at ASC, id ASC
        """
    ):
        grouped[str(row["candidate_hash"])].append(row)
    aggregates: dict[str, dict[str, Any]] = {}
    for hash_key, entries in grouped.items():
        similarities = [float(e["similarity"]) for e in entries if e["similarity"] is not None]
        last = entries[-1]
        llm_scores = [e for e in entries if e["llm_score"] is not None]
        aggregates[hash_key] = {
            "sim_last": similarities[-1] if similarities else None,
            "sim_max": max(similarities) if similarities else None,
            "sim_min": min(similarities) if similarities else None,
            "sim_mean": sum(similarities) / len(similarities) if similarities else None,
            "audit_count": len(entries),
            "would_filter_last": int(last["would_filter"] or 0),
            "context_class": str(last["context_class"] or ""),
            "profile_digest_last": str(last["profile_digest"] or ""),
            "llm_score_last": (float(llm_scores[-1]["llm_score"]) if llm_scores else None),
        }
    return aggregates


def build_export_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], Counter]:
    audit = load_audit_aggregates(conn)
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(discovery_candidates)").fetchall()
    }
    # A pre-provenance database simply has no score_source / llm_score_raw /
    # teacher_model columns; those rows are legacy ('') by definition.
    columns = [column for column in CANDIDATE_COLUMNS if column in existing]
    placeholders = ", ".join("?" for _ in EVALUATED_STATUSES)
    rows = conn.execute(
        f"""SELECT {", ".join(columns)}
            FROM discovery_candidates
            WHERE status IN ({placeholders})""",
        EVALUATED_STATUSES,
    ).fetchall()

    export_rows: list[dict[str, Any]] = []
    policy_counts: Counter = Counter()
    for row in rows:
        record = dict(row)
        entry = audit.get(audit_hash(str(record["candidate_key"])))
        audit_score = entry["llm_score_last"] if entry else None
        resolved = resolve_teacher_label(record, audit_score)
        if resolved is None:
            policy_counts["dropped"] += 1
            continue
        teacher_score, policy = resolved
        policy_counts[policy] += 1
        threshold = effective_admission_threshold(str(record["source_strategy"]))
        export: dict[str, Any] = {
            "candidate_id": int(record["id"]),
            "candidate_key": str(record["candidate_key"]),
            "batch_id": str(record["evaluated_at"] or ""),
            "status": str(record["status"]),
            "platform": str(record["source_platform"] or ""),
            "strategy": str(record["source_strategy"] or ""),
            "content_type": str(record["content_type"] or ""),
            "candidate_tier": str(record["candidate_tier"] or ""),
            "teacher_score": teacher_score,
            "label_policy": policy,
            "score_source": str(record.get("score_source") or ""),
            "teacher_model": str(record.get("teacher_model") or ""),
            "profile_digest": str(record.get("profile_digest") or ""),
            "negative_digest": str(record.get("negative_digest") or ""),
            "y": 1 if teacher_score >= threshold else 0,
            "admission_threshold": threshold,
            "aux_labels": {
                "style_key": str(record["style_key"] or ""),
                "temporal_class": str(record["temporal_class"] or ""),
                "topic_group": str(record["topic_group"] or ""),
                "franchise_key": str(record["franchise_key"] or ""),
            },
            "text": {
                "title": str(record["title"] or ""),
                "description": str(record["description"] or ""),
                "body_text": str(record["body_text"] or ""),
            },
            "features": {
                "duration": int(record["duration"] or 0),
                "published_at": str(record["published_at"] or ""),
                **{col: int(record[col] or 0) for col in ENGAGEMENT_COLUMNS},
            },
        }
        if entry is not None:
            export["audit"] = {k: v for k, v in entry.items() if k != "llm_score_last"}
        export_rows.append(export)
    return export_rows, policy_counts


def snapshot_shadow_audit(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Full llm-joined audit rows — the 30-day/cap purge makes this urgent."""

    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT candidate_hash, platform_class, context_class, similarity,
                   threshold, would_filter, embedding_status, fail_open,
                   llm_score, admission_threshold, admission_result,
                   profile_digest, created_at, completed_at
            FROM evaluator_prefilter_shadow_audit
            WHERE llm_score IS NOT NULL
            ORDER BY created_at ASC, id ASC
            """
        )
    ]


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.database.exists():
        print(f"database not found: {args.database}", file=sys.stderr)
        return 1

    conn = connect_readonly(args.database)
    export_rows, policy_counts = build_export_rows(conn)
    audit_rows = snapshot_shadow_audit(conn)
    conn.close()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out / f"dataset_{stamp}.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_path = args.out / f"shadow_audit_{stamp}.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_model = Counter(row["teacher_model"] or "(unstamped)" for row in export_rows)
    by_y = Counter(row["y"] for row in export_rows)
    meta = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": stamp,
        "source_db": str(args.database),
        "rows": len(export_rows),
        "label_policy_counts": dict(policy_counts),
        "label_balance": {"y1": by_y.get(1, 0), "y0": by_y.get(0, 0)},
        "teacher_model_counts": dict(by_model),
        "admission_thresholds": {"default": ADMISSION_DEFAULT, "explore": ADMISSION_EXPLORE},
        "batch_count": len({row["batch_id"] for row in export_rows}),
        "shadow_audit_rows": len(audit_rows),
        "files": {"dataset": dataset_path.name, "shadow_audit": audit_path.name},
    }
    meta_path = args.out / f"dataset_{stamp}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"exported {len(export_rows)} rows ({len(audit_rows)} shadow-audit rows)")
    print(f"policy: {dict(policy_counts)}")
    print(f"labels: {meta['label_balance']}")
    print(f"teacher models: {dict(by_model)}")
    print(f"wrote: {dataset_path}")
    print(f"       {audit_path}")
    print(f"       {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
