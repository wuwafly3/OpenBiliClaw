"""Offline probe: how much admission signal do teacher tags carry?

Decision experiment for the ml-ranking dual-pipeline question (2026-08-16):
`topic_group` / `style_key` / `temporal_class` are teacher outputs and
therefore banned as ML input features (label leakage). Two ways to recover
that information were on the table:

  A. pay a cheap tags-only LLM pass over every candidate, use real tags as
     features (dual pipeline);
  B. meta-distill tags with aux heads trained on historical labels.

This probe measures the oracle version of A — v1 adds the true teacher tags
as features, v2 is the deterministic baseline — under one protocol. The
decision rule agreed upstream:

  dRho < 0.02  -> adopt B (predicted tags are enough)
  dRho > 0.05  -> tags are the main signal, pay A's dual-pipeline cost

Measured result on the production database (see
docs/plans/2026-08-16-ml-tags-ablation-probe.md): dRho = +0.111 (rho 0.604
-> 0.715; AUC 0.799 -> 0.864), xhs rho +0.210 — decision: A.

Dataset assembly follows ``export_ranking_dataset.resolve_teacher_label``:
LLM-judgment ``score_source`` uses ``llm_score_raw``; legacy empty source
keeps a non-zero ``relevance_score`` or recovers from
``evaluator_prefilter_shadow_audit``; prefilter / viewed / truncated /
error rows are dropped even when ``relevance_score`` is non-zero. Labels
for the AUC view use the per-row effective admission threshold (exact
strategy ``explore`` uses 0.58, otherwise 0.60).

Usage:
    python scripts/ml_tags_ablation_probe.py [db] [--folds 5] [--seeds 3] \
        [--json out.json]

Requires numpy + scikit-learn (training environment only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from export_ranking_dataset import (  # noqa: E402
    candidate_row_for_label,
    effective_admission_threshold,
    resolve_teacher_label,
    teacher_label_select_columns,
)

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")

EVALUATED_STATUSES = frozenset(
    {
        "evaluated",
        "rejected_low_score",
        "cached",
        "rejected_cache_admission",
        "rejected_temporal_stale",
        "rejected_franchise_quota",
    }
)
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
TOP_STRATEGIES = 8
MIN_STYLE_CLASS_ROWS = 10
TOP_TOPICS = 15


def audit_hash(identity: str) -> str:
    payload = f"openbiliclaw:evaluator-prefilter:v1\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_audit(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    """Latest audit row per candidate identity hash (at-time similarity)."""

    latest: dict[str, sqlite3.Row] = {}
    for row in conn.execute(
        """
        SELECT candidate_hash, similarity, llm_score, would_filter, context_class
        FROM evaluator_prefilter_shadow_audit
        ORDER BY created_at ASC, id ASC
        """
    ):
        latest[str(row["candidate_hash"])] = row
    return latest


def build_records(
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audit = load_audit(conn)
    rows = conn.execute(
        f"""SELECT candidate_key, status, source_platform, source_strategy,
                   content_type, candidate_tier, title, description, body_text,
                   duration, {', '.join(ENGAGEMENT_COLUMNS)},
                   relevance_score, style_key, temporal_class, topic_group
                   {teacher_label_select_columns(conn)}
            FROM discovery_candidates
            WHERE status IN ({', '.join('?' for _ in EVALUATED_STATUSES)})""",
        tuple(sorted(EVALUATED_STATUSES)),
    ).fetchall()

    records: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for row in rows:
        entry = audit.get(audit_hash(str(row["candidate_key"])))
        audit_score = (
            float(entry["llm_score"])
            if entry is not None and entry["llm_score"] is not None
            else None
        )
        resolved = resolve_teacher_label(candidate_row_for_label(row), audit_score)
        if resolved is None:
            stats["dropped"] += 1
            continue
        label, policy = resolved
        stats[policy] += 1
        if policy == "legacy_audit_recovered":
            stats["label_recovered_from_audit"] += 1
        else:
            stats["label_from_candidates"] += 1
        record: dict[str, Any] = {
            "label": label,
            "platform": str(row["source_platform"] or "unknown"),
            "strategy": str(row["source_strategy"] or "unknown"),
            "content_type": str(row["content_type"] or "video"),
            "tier": str(row["candidate_tier"] or "primary"),
            "style": str(row["style_key"] or "style_empty"),
            "temporal": str(row["temporal_class"] or "unknown"),
            "topic": str(row["topic_group"] or "topic_empty"),
            "title_len": len(str(row["title"] or "")),
            "desc_len": len(str(row["description"] or "")),
            "body_len": len(str(row["body_text"] or "")),
            "duration_s": float(row["duration"] or 0),
            **{col: float(row[col] or 0) for col in ENGAGEMENT_COLUMNS},
        }
        if entry is not None and entry["similarity"] is not None:
            record["sim"] = float(entry["similarity"])
            record["would_filter"] = float(entry["would_filter"] or 0)
            record["context"] = str(entry["context_class"] or "other")
        records.append(record)
    return records, dict(stats)


def onehot(names: list[str], values: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(values), len(names)), dtype=float)
    for i, value in enumerate(values):
        matrix[i, index[value]] = 1.0
    return matrix


def build_features(
    records: list[dict[str, Any]],
    *,
    with_tags: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns: list[np.ndarray] = []

    def add(values: list[float]) -> None:
        columns.append(np.asarray(values, dtype=float))

    add([r.get("sim", 0.0) for r in records])
    add([1.0 if "sim" in r else 0.0 for r in records])
    add([r.get("would_filter", 0.0) for r in records])
    for col in ENGAGEMENT_COLUMNS:
        add([float(np.log1p(r[col])) for r in records])
        add([1.0 if r[col] > 0 else 0.0 for r in records])
    add([float(np.log1p(r["duration_s"])) for r in records])
    add([1.0 if r["duration_s"] > 0 else 0.0 for r in records])
    add([float(np.log1p(r["title_len"])) for r in records])
    add([float(np.log1p(r["desc_len"])) for r in records])
    add([float(np.log1p(r["body_len"])) for r in records])
    for platform in sorted({r["platform"] for r in records}):
        add([1.0 if r["platform"] == platform else 0.0 for r in records])
    strategies = [s for s, _ in Counter(r["strategy"] for r in records).most_common(TOP_STRATEGIES)]
    for strategy in strategies:
        add([1.0 if r["strategy"] == strategy else 0.0 for r in records])
    add([1.0 if r["strategy"] not in strategies else 0.0 for r in records])
    for context in sorted({r.get("context", "no_audit") for r in records}):
        add([1.0 if r.get("context") == context else 0.0 for r in records])
    x = np.column_stack(columns)

    style_keep = [
        c
        for c, n in Counter(r["style"] for r in records).most_common()
        if n >= MIN_STYLE_CLASS_ROWS
    ]
    if with_tags:
        # Oracle view of pipeline A: the tags a cheap LLM tagger would
        # produce. Same-call leakage is the point — A buys exactly this.
        style_names = style_keep + ["style_other"]
        temporal_names = sorted({r["temporal"] for r in records})
        top_topics = [c for c, _ in Counter(r["topic"] for r in records).most_common(TOP_TOPICS)]
        topic_names = [f"t_{t}" for t in top_topics] + ["t_other"]
        style_values = [
            s if s in style_keep else "style_other" for s in (r["style"] for r in records)
        ]
        topic_values = [
            f"t_{t}" if t in top_topics else "t_other" for t in (r["topic"] for r in records)
        ]
        x = np.hstack(
            [
                x,
                onehot(style_names, style_values),
                onehot(temporal_names, [r["temporal"] for r in records]),
                onehot(topic_names, topic_values),
            ]
        )

    y = np.asarray([r["label"] for r in records], dtype=float)
    y_binary = np.asarray(
        [1 if r["label"] >= effective_admission_threshold(r["strategy"]) else 0 for r in records],
        dtype=int,
    )
    platforms = np.asarray([r["platform"] for r in records])
    strata = [s if s in style_keep else "style_other" for s in (r["style"] for r in records)]
    return x, y, y_binary, platforms, strata


def evaluate(db_path: Path, folds: int, seeds: int) -> dict[str, Any]:
    conn = connect_readonly(db_path)
    records, stats = build_records(conn)
    conn.close()
    print(f"dataset: {len(records)} rows; provenance {stats}")

    summary: dict[str, Any] = {"dataset": stats, "rows": len(records)}
    variants: dict[str, dict[str, list[float]]] = {}
    for tag, with_tags in (("v1_oracle_tags", True), ("v2_baseline", False)):
        x, y, y_bin, platforms, strata = build_features(records, with_tags=with_tags)
        metrics: dict[str, list[float]] = {
            k: [] for k in ("rho", "rho_bilibili", "rho_xiaohongshu", "auc")
        }
        for seed in range(seeds):
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42 + seed)
            fold_strata = [f"{p}|{s}" for p, s in zip(platforms, strata, strict=True)]
            rare = {k for k, n in Counter(fold_strata).items() if n < folds}
            fold_strata = [s if s not in rare else f"{s.split('|')[0]}|rare" for s in fold_strata]
            for train_idx, test_idx in splitter.split(x, fold_strata):
                scaler = StandardScaler().fit(x[train_idx])
                x_train, x_test = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
                pred = (
                    RidgeCV(alphas=np.logspace(-3, 3, 13))
                    .fit(x_train, y[train_idx])
                    .predict(x_test)
                )
                metrics["rho"].append(float(spearmanr(pred, y[test_idx]).statistic))
                for platform, bucket in (
                    ("bilibili", "rho_bilibili"),
                    ("xiaohongshu", "rho_xiaohongshu"),
                ):
                    mask = platforms[test_idx] == platform
                    if mask.sum() >= 5:
                        metrics[bucket].append(
                            float(spearmanr(pred[mask], y[test_idx][mask]).statistic)
                        )
                proba = (
                    LogisticRegression(max_iter=3000)
                    .fit(x_train, y_bin[train_idx])
                    .predict_proba(x_test)[:, 1]
                )
                metrics["auc"].append(float(roc_auc_score(y_bin[test_idx], proba)))
        variants[tag] = metrics
        summary[tag] = {k: float(np.mean(v)) for k, v in metrics.items() if v}
        kept = summary[tag]
        print(
            f"{tag:14s} dims={x.shape[1]:3d} rho={kept['rho']:.3f} "
            f"rho_bili={kept.get('rho_bilibili', float('nan')):.3f} "
            f"rho_xhs={kept.get('rho_xiaohongshu', float('nan')):.3f} "
            f"AUC={kept['auc']:.3f}"
        )

    delta_rho = summary["v1_oracle_tags"]["rho"] - summary["v2_baseline"]["rho"]
    summary["delta_rho"] = delta_rho
    print(f"delta rho = {delta_rho:+.3f}")
    if delta_rho < 0.02:
        print("decision: dRho < 0.02 -> adopt B (tags meta-distillation)")
    elif delta_rho > 0.05:
        print("decision: dRho > 0.05 -> tags are the main signal; pay A's dual-pipeline cost")
    else:
        print("decision: 0.02 <= dRho <= 0.05 -> ambiguous zone; re-measure with more data")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn is required for this training-environment probe", file=sys.stderr)
        return 2
    if not args.database.exists():
        print(f"database not found: {args.database}", file=sys.stderr)
        return 1

    summary = evaluate(args.database, args.folds, args.seeds)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"metrics written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
