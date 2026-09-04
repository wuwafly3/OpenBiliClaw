"""Train the Wave 1 admission classifier on teacher-labeled rows.

Labels follow spec S1.1 (``llm_score_raw`` vs per-row admission floor).
Features use teacher ``topic_group`` / ``style_key`` / ``temporal_class`` as
the training-time stand-in for the cheap tags channel (S1.2a). Teacher
scores themselves are never features.

Usage::

    uv run --extra ml python scripts/train_relevance_model.py \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --require-snapshot --dry-run

    uv run --extra ml python scripts/train_relevance_model.py \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --require-snapshot \\
        --relabel-jsonl data/ml_artifacts/gate_contract_relabel_20260824T030808Z.jsonl

Writes a versioned JSON artifact (weights, scaler, isotonic map, OOF
metrics). Does not change ``[discovery].relevance_scorer``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_ranking_dataset import (  # noqa: E402
    candidate_row_for_label,
    effective_admission_threshold,
    resolve_teacher_label,
    teacher_label_select_columns,
)

from openbiliclaw.discovery.eval_context import (  # noqa: E402
    GATE_RECENT_KEYS,
    EvaluationContextSnapshot,
    loads_snapshot_json,
    parse_negative_examples,
    parse_recall_pool,
)
from openbiliclaw.discovery.style_keys import normalize_style_key  # noqa: E402
from openbiliclaw.discovery.temporal import normalize_temporal_class  # noqa: E402
from openbiliclaw.ml.features import (  # noqa: E402
    ENGAGEMENT_COLUMNS,
    FEATURE_VERSION,
    encode_features,
)

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")
DEFAULT_OUT = PROJECT_ROOT / "data" / "ml_artifacts" / "admission_teacher_v1.json"
EVALUATED_STATUSES = (
    "evaluated",
    "rejected_low_score",
    "cached",
    "rejected_cache_admission",
    "rejected_temporal_stale",
    "rejected_franchise_quota",
)
FOLDS = 5


def audit_hash(identity: str) -> str:
    payload = f"openbiliclaw:evaluator-prefilter:v1\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def binary_label(teacher_score: float, source_strategy: object) -> int:
    return int(float(teacher_score) >= effective_admission_threshold(source_strategy))


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def select_or_null(existing: set[str], names: tuple[str, ...]) -> str:
    parts = [name if name in existing else f"NULL AS {name}" for name in names]
    return ", ".join(parts)


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_audit(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    latest: dict[str, sqlite3.Row] = {}
    existing = table_columns(conn, "evaluator_prefilter_shadow_audit")
    if "candidate_hash" not in existing:
        return latest
    digest = select_or_null(existing, ("profile_digest",))
    rows = conn.execute(
        f"""
        SELECT candidate_hash, similarity, llm_score, would_filter,
               context_class, {digest}
        FROM evaluator_prefilter_shadow_audit
        ORDER BY created_at ASC, id ASC
        """
    )
    for row in rows:
        latest[str(row["candidate_hash"])] = row
    return latest


def load_teacher_records(
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audit = load_audit(conn)
    extra = teacher_label_select_columns(conn)
    existing = table_columns(conn, "discovery_candidates")
    optional = select_or_null(
        existing,
        (
            "body_text",
            "rating_score",
            "source_rank",
            "profile_digest",
            "negative_digest",
        ),
    )
    rows = conn.execute(
        f"""
        SELECT candidate_key, status, source_platform, source_strategy,
               content_type, candidate_tier, title, description, duration,
               {optional},
               {", ".join(ENGAGEMENT_COLUMNS)},
               relevance_score, style_key, temporal_class, topic_group
               {extra}
        FROM discovery_candidates
        WHERE status IN ({", ".join("?" for _ in EVALUATED_STATUSES)})
        """,
        tuple(sorted(EVALUATED_STATUSES)),
    ).fetchall()

    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in rows:
        entry = audit.get(audit_hash(str(row["candidate_key"])))
        audit_score = None
        if entry is not None and entry["llm_score"] is not None:
            audit_score = float(entry["llm_score"])
        resolved = resolve_teacher_label(candidate_row_for_label(row), audit_score)
        if resolved is None:
            stats["dropped"] += 1
            continue
        teacher_score, policy = resolved
        stats[policy] += 1
        strategy = str(row["source_strategy"] or "")
        candidate_digest = str(row["profile_digest"] or "").strip()
        candidate_negative = str(row["negative_digest"] or "").strip()
        audit_digest = (
            str(entry["profile_digest"] or "").strip()
            if entry is not None and "profile_digest" in entry
            else ""
        )
        record: dict[str, Any] = {
            "candidate_key": str(row["candidate_key"] or ""),
            "teacher_score": float(teacher_score),
            "y": binary_label(teacher_score, strategy),
            "platform": str(row["source_platform"] or "unknown") or "unknown",
            "strategy": strategy or "unknown",
            "content_type": str(row["content_type"] or "video") or "video",
            "style": normalize_style_key(row["style_key"]) or "style_empty",
            "temporal": normalize_temporal_class(row["temporal_class"]),
            "topic": " ".join(str(row["topic_group"] or "").split()) or "topic_empty",
            "title_len": len(str(row["title"] or "")),
            "desc_len": len(str(row["description"] or "")),
            "body_len": len(str(row["body_text"] or "")),
            "duration_s": float(row["duration"] or 0),
            "rating_score": float(row["rating_score"] or 0),
            "source_rank": float(row["source_rank"] or 0),
            "candidate_profile_digest": candidate_digest,
            "candidate_negative_digest": candidate_negative,
            "profile_digest": candidate_digest or audit_digest,
            **{col: float(row[col] or 0) for col in ENGAGEMENT_COLUMNS},
        }
        if entry is not None and entry["similarity"] is not None:
            record["sim"] = float(entry["similarity"])
            record["would_filter"] = float(entry["would_filter"] or 0)
            record["context"] = str(entry["context_class"] or "other")
        records.append(record)
    return records, dict(stats)


def filter_records_with_candidate_profile_digest(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep teacher rows whose ``discovery_candidates.profile_digest`` is set.

    Prefilter-audit digests are ignored: those are not labeling-time snapshots.
    """

    kept = [
        record for record in records if str(record.get("candidate_profile_digest") or "").strip()
    ]
    return kept, len(records) - len(kept)


def load_relabel_overrides(path: Path, id_to_key: dict[int, str]) -> dict[str, dict[str, Any]]:
    """Map ``candidate_key`` → new-contract score/digest from a relabel JSONL."""

    overrides: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        key = id_to_key.get(int(payload["candidate_id"]))
        if not key:
            continue
        overrides[key] = payload
    return overrides


def apply_relabel_overrides(
    records: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Replace old-contract scores/digests with relabel JSONL values."""

    applied = 0
    updated: list[dict[str, Any]] = []
    for record in records:
        payload = overrides.get(str(record.get("candidate_key") or ""))
        if payload is None:
            updated.append(record)
            continue
        next_record = dict(record)
        new_score = float(payload["new_llm_score_raw"])
        next_record["teacher_score"] = new_score
        next_record["y"] = int(
            payload.get("new_y", binary_label(new_score, record.get("strategy")))
        )
        next_record["candidate_profile_digest"] = str(payload["new_profile_digest"])
        next_record["candidate_negative_digest"] = str(payload["new_negative_digest"])
        next_record["profile_digest"] = str(payload["new_profile_digest"])
        updated.append(next_record)
        applied += 1
    return updated, applied


def verified_snapshot_from_row(
    row: sqlite3.Row,
) -> EvaluationContextSnapshot | None:
    summary_raw = loads_snapshot_json(str(row["profile_summary_json"] or ""))
    if not isinstance(summary_raw, dict):
        return None
    snapshot = EvaluationContextSnapshot(
        profile_digest=str(row["profile_digest"] or ""),
        negative_digest=str(row["negative_digest"] or ""),
        profile_summary=dict(summary_raw),
        recall_pool=parse_recall_pool(loads_snapshot_json(str(row["recall_pool_json"] or ""))),
        negative_examples=parse_negative_examples(
            loads_snapshot_json(str(row["negative_examples_json"] or ""))
        ),
    )
    if not snapshot.digests_match():
        return None
    return snapshot


class _DigestMismatch:
    """Present snapshot row whose stored payload fails ``digests_match()``."""

    def digests_match(self) -> bool:
        return False


def snapshot_lookup(conn: sqlite3.Connection) -> Any:
    cache: dict[tuple[str, str], EvaluationContextSnapshot | _DigestMismatch | None] = {}

    def get_evaluation_context_snapshot(
        *,
        profile_digest: str,
        negative_digest: str,
    ) -> EvaluationContextSnapshot | _DigestMismatch | None:
        digest = str(profile_digest or "").strip()
        negative = str(negative_digest or "").strip()
        if not digest:
            return None
        key = (digest, negative)
        if key in cache:
            return cache[key]
        row = conn.execute(
            """
            SELECT profile_digest, negative_digest, schema_version,
                   profile_summary_json, recall_pool_json, negative_examples_json
            FROM evaluation_context_snapshots
            WHERE profile_digest = ? AND negative_digest = ?
            LIMIT 1
            """,
            (digest, negative),
        ).fetchone()
        if row is None:
            cache[key] = None
            return None
        snapshot = verified_snapshot_from_row(row)
        cache[key] = snapshot if snapshot is not None else _DigestMismatch()
        return cache[key]

    return get_evaluation_context_snapshot


def filter_records_with_verified_snapshot(
    records: list[dict[str, Any]],
    getter: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep rows whose digest pair loads a ``digests_match()`` snapshot.

    Empty digest, missing snapshot, and hash mismatch are counted separately
    and never enter the training matrix.
    """

    stats: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for record in records:
        digest = str(record.get("candidate_profile_digest") or "").strip()
        negative = str(record.get("candidate_negative_digest") or "").strip()
        if not digest or not negative:
            stats["dropped_empty_digest"] += 1
            continue
        snapshot = getter(profile_digest=digest, negative_digest=negative)
        if snapshot is None:
            stats["dropped_no_snapshot"] += 1
            continue
        match = getattr(snapshot, "digests_match", None)
        if callable(match) and not match():
            stats["dropped_digest_mismatch"] += 1
            continue
        summary = getattr(snapshot, "profile_summary", None)
        if isinstance(summary, dict) and any(key in summary for key in GATE_RECENT_KEYS):
            stats["dropped_old_contract"] += 1
            continue
        kept.append(record)
        stats["kept"] += 1
    return kept, dict(stats)


def encode_teacher_features(
    records: list[dict[str, Any]],
    *,
    strategy_names: list[str] | None = None,
    topic_names: list[str] | None = None,
    context_names: list[str] | None = None,
    platform_names: list[str] | None = None,
    content_type_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Compat wrapper around the shared encoder used at inference."""

    return encode_features(
        records,
        strategy_names=strategy_names,
        topic_names=topic_names,
        context_names=context_names,
        platform_names=platform_names,
        content_type_names=content_type_names,
    )


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    pred = (proba >= threshold).astype(int)
    neg = y == 0
    pos = y == 1
    fpr = float(((pred == 1) & neg).sum() / neg.sum()) if int(neg.sum()) else 0.0
    fnr = float(((pred == 0) & pos).sum() / pos.sum()) if int(pos.sum()) else 0.0
    brier = float(np.mean((proba - y) ** 2))
    auc = float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else 0.5
    return {
        "auc": auc,
        "agreement": float((pred == y).mean()),
        "fpr": fpr,
        "fnr": fnr,
        "brier": brier,
        "threshold": float(threshold),
        "n": float(len(y)),
        "n_pos": float(pos.sum()),
        "n_neg": float(neg.sum()),
    }


def pick_threshold(y: np.ndarray, proba: np.ndarray, *, fpr_max: float = 0.10) -> float:
    best_threshold = 0.5
    best_agreement = -1.0
    for raw in np.linspace(0.05, 0.95, 19):
        metrics = classification_metrics(y, proba, float(raw))
        if metrics["fpr"] > fpr_max:
            continue
        if metrics["agreement"] > best_agreement:
            best_agreement = metrics["agreement"]
            best_threshold = float(raw)
    return best_threshold


def _group_ids(records: list[dict[str, Any]]) -> np.ndarray:
    labels: list[str] = []
    for record in records:
        profile = str(record.get("profile_digest") or "").strip()
        negative = str(record.get("candidate_negative_digest") or "").strip()
        if profile:
            labels.append(f"{profile}:{negative}")
        else:
            labels.append(f"row:{record['candidate_key']}")
    unique = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.asarray([unique[label] for label in labels], dtype=int)


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{r['candidate_key']}\t{r['y']}\t{r['teacher_score']:.6f}"
        for r in sorted(records, key=lambda item: item["candidate_key"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train(
    records: list[dict[str, Any]],
    *,
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    x, feature_names, vocab = encode_teacher_features(records)
    y = np.asarray([r["y"] for r in records], dtype=int)
    platforms = np.asarray([r["platform"] for r in records])
    groups = _group_ids(records)
    oof = np.zeros(len(records), dtype=float)
    n_groups = len(set(groups.tolist()))
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_groups >= FOLDS:
        splitter = GroupKFold(n_splits=FOLDS)
        splits = list(splitter.split(x, y, groups))
    else:
        n_splits = max(2, min(FOLDS, n_pos, n_neg, len(records)))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=19)
        splits = list(splitter.split(x, y))

    fold_count = 0
    for train_idx, test_idx in splits:
        fold_count += 1
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        if len(set(y[train_idx].tolist())) < 2:
            oof[test_idx] = float(y[train_idx].mean())
            continue
        model = LogisticRegression(max_iter=4000, solver="lbfgs")
        model.fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrated = np.asarray(calibrator.fit_transform(oof, y), dtype=float)
    iso_x = getattr(calibrator, "X_thresholds_", getattr(calibrator, "X_thresholds", None))
    iso_y = getattr(calibrator, "y_thresholds_", getattr(calibrator, "y_thresholds", None))
    if iso_x is None or iso_y is None:
        raise RuntimeError("IsotonicRegression did not expose threshold arrays")
    if decision_threshold is None:
        threshold = pick_threshold(y, calibrated)
        threshold_policy = "oof_fpr_le_0.10"
    else:
        threshold = float(decision_threshold)
        threshold_policy = "fixed"
    overall = classification_metrics(y, calibrated, threshold)
    at_half = classification_metrics(y, calibrated, 0.5)
    by_platform: dict[str, dict[str, float]] = {}
    for platform in sorted(set(platforms.tolist())):
        mask = platforms == platform
        if int(mask.sum()) < 8 or len(set(y[mask].tolist())) < 2:
            continue
        by_platform[platform] = classification_metrics(y[mask], calibrated[mask], threshold)

    scaler = StandardScaler().fit(x)
    x_all = scaler.transform(x)
    final = LogisticRegression(max_iter=4000, solver="lbfgs")
    final.fit(x_all, y)
    return {
        "feature_version": FEATURE_VERSION,
        "tags_source": "teacher_oracle",
        "trained_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_rows": len(records),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_folds": fold_count,
        "n_groups": n_groups,
        "group_split": ("profile_negative_digest" if n_groups < len(records) else "per-row"),
        "dataset_fingerprint": dataset_fingerprint(records),
        "feature_names": feature_names,
        "vocab": vocab,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": final.coef_[0].tolist(),
        "intercept": float(final.intercept_[0]),
        "isotonic_x": np.asarray(iso_x, dtype=float).tolist(),
        "isotonic_y": np.asarray(iso_y, dtype=float).tolist(),
        "isotonic_fit": "oof_logistic_scores",
        "decision_threshold": threshold,
        "threshold_policy": threshold_policy,
        "oof_metrics": overall,
        "oof_metrics_at_0_5": at_half,
        "oof_metrics_by_platform": by_platform,
        "s1_5": {
            "auc_ge_0.80": overall["auc"] >= 0.80,
            "agreement_ge_0.90": overall["agreement"] >= 0.90,
            "fpr_le_0.10": overall["fpr"] <= 0.10,
            "fnr_le_0.15": overall["fnr"] <= 0.15,
            "brier_le_0.18": overall["brier"] <= 0.18,
        },
    }


def _print_metrics(payload: dict[str, Any]) -> None:
    overall = payload["oof_metrics"]
    print("=== teacher admission model ===")
    print(
        f"  rows                  {payload['n_rows']}  y1={payload['n_pos']} y0={payload['n_neg']}"
    )
    print(f"  groups/folds          {payload['n_groups']} / {payload['n_folds']}")
    if payload["n_groups"] == payload["n_rows"]:
        print("  grouping              per-row (no shared profile_digest)")
    else:
        print(f"  grouping              {payload['group_split']}")
    print(f"  features              {len(payload['feature_names'])}  {payload['feature_version']}")
    print(f"  tags_source           {payload['tags_source']}")
    print(f"  row_filter            {payload.get('row_filter', 'all_teacher')}")
    policy = str(payload.get("threshold_policy") or "oof_fpr_le_0.10")
    print(f"  threshold             {payload['decision_threshold']:.2f} ({policy})")
    print(f"  AUC                   {overall['auc']:.3f}")
    print(f"  agreement             {overall['agreement']:.3f}")
    print(f"  FPR / FNR             {overall['fpr']:.3f} / {overall['fnr']:.3f}")
    print(f"  Brier                 {overall['brier']:.3f}")
    at_half = payload.get("oof_metrics_at_0_5") or {}
    if at_half:
        print(
            f"  at 0.50               agr={at_half['agreement']:.3f} "
            f"fpr={at_half['fpr']:.3f} fnr={at_half['fnr']:.3f}"
        )
    print(
        "  S1.5 (info only)      "
        + ", ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in payload["s1_5"].items())
    )
    by_platform = payload.get("oof_metrics_by_platform") or {}
    if by_platform:
        print("  by platform:")
        for platform, metrics in by_platform.items():
            print(
                f"    {platform:12s} n={int(metrics['n']):4d}  "
                f"auc={metrics['auc']:.3f} agr={metrics['agreement']:.3f} "
                f"fpr={metrics['fpr']:.3f} fnr={metrics['fnr']:.3f}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--require-profile-digest",
        action="store_true",
        help=(
            "Train only on rows with a non-empty discovery_candidates.profile_digest "
            "(labeling-time stamp). Audit-only digests are excluded."
        ),
    )
    parser.add_argument(
        "--require-snapshot",
        action="store_true",
        help=(
            "Train only on rows whose digest pair loads a verified "
            "evaluation_context_snapshots payload (digests_match). "
            "Stricter than --require-profile-digest."
        ),
    )
    parser.add_argument(
        "--relabel-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional gate-contract relabel JSONL. Overrides teacher_score / y / "
            "digest pair for matching candidate ids before snapshot filtering."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filter stats and skip fitting / writing an artifact.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=(
            "Fixed operating point on calibrated OOF scores. Default is the S1.5 "
            "FPR<=0.10 search. Pass 0.5 to ignore that search."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.db.expanduser().resolve().is_file():
        print(f"database not found: {args.db}")
        return 1
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn required: uv run --extra ml python scripts/train_relevance_model.py")
        return 2
    db_path = args.db.expanduser().resolve()
    conn = connect_readonly(db_path)
    records, stats = load_teacher_records(conn)
    print(f"teacher rows: {len(records)}  provenance={stats}")
    if args.relabel_jsonl is not None:
        relabel_path = args.relabel_jsonl.expanduser().resolve()
        if not relabel_path.is_file():
            print(f"relabel jsonl not found: {relabel_path}")
            conn.close()
            return 1
        id_to_key = {
            int(row["id"]): str(row["candidate_key"] or "")
            for row in conn.execute("SELECT id, candidate_key FROM discovery_candidates")
        }
        overrides = load_relabel_overrides(relabel_path, id_to_key)
        records, applied = apply_relabel_overrides(records, overrides)
        print(f"  relabel-jsonl         {relabel_path.name} applied {applied}")
    row_filter = "all_teacher"
    filter_stats: dict[str, int] = {}
    if bool(args.require_snapshot):
        getter = snapshot_lookup(conn)
        records, filter_stats = filter_records_with_verified_snapshot(records, getter)
        row_filter = "verified_snapshot"
        print(
            "  require-snapshot kept "
            f"{len(records)}  empty={filter_stats.get('dropped_empty_digest', 0)} "
            f"missing={filter_stats.get('dropped_no_snapshot', 0)} "
            f"mismatch={filter_stats.get('dropped_digest_mismatch', 0)} "
            f"old_contract={filter_stats.get('dropped_old_contract', 0)}"
        )
    elif bool(args.require_profile_digest):
        records, dropped = filter_records_with_candidate_profile_digest(records)
        row_filter = "candidate_profile_digest"
        print(
            "  require-profile-digest kept "
            f"{len(records)}  dropped {dropped} (empty candidate digest)"
        )
    conn.close()
    if bool(args.dry_run):
        n_pos = sum(int(r["y"]) for r in records)
        print(f"  row_filter            {row_filter}")
        print(f"  kept y                y1={n_pos} y0={len(records) - n_pos}")
        print("  dry-run               skip fit")
        return 0 if records else 2
    if len(records) < 40:
        print("not enough teacher rows to train")
        return 1
    payload = train(records, decision_threshold=args.decision_threshold)
    payload["row_filter"] = row_filter
    if filter_stats:
        payload["snapshot_filter"] = filter_stats
    _print_metrics(payload)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  artifact              {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
