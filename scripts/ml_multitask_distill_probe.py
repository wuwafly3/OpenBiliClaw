"""Offline probe: does multi-task distillation of style/temporal help relevance?

Experiment for ml-ranking problem #2 (feature circular dependency):
``style_key`` / ``temporal_class`` are LLM evaluation outputs, so they cannot
be input features of a distilled scorer. This probe trains the auxiliary
targets jointly with the relevance score (shared-trunk MLP), a two-stage
variant (predicted aux probabilities as features), and single-task baselines,
then compares held-out rank correlation.

Dataset assembly (read-only):
- ``discovery_candidates`` rows with an eval outcome; teacher label =
  ``relevance_score`` when non-zero (genuine LLM judgment — caps only zero),
  otherwise the score recovered from ``evaluator_prefilter_shadow_audit``
  (cap-zeroed rows keep their at-time ``llm_score`` there).
- Rows that are zero with no audit join are dropped (ambiguous origin).
- Features are evaluation-time deterministic quantities only: shadow-audit
  similarity aggregates, engagement log-counts + availability masks, duration,
  published age, text length stats, platform/strategy/context one-hots.

Usage:
    python scripts/ml_multitask_distill_probe.py [path/to/openbiliclaw.db] \
        [--folds 5] [--seeds 3] [--json out.json]

Requires numpy + scikit-learn (training environment only; nothing here ships
in the runtime package).
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

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")

# Candidate statuses whose relevance fields come from a completed evaluation.
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
# Style classes below this count are merged into "style_other" (tiny classes
# make macro-F1 noise dominate the probe).
MIN_STYLE_CLASS_ROWS = 10
TEMPORAL_MERGES = {"breaking": "current"}
TOP_STRATEGIES = 8
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
SEEDS_DEFAULT = 3
MLP_HIDDEN = (64, 32)
# Multi-task MSE balance: the score head is replicated in the target vector
# so its loss is not drowned by the 11 + 5 one-hot auxiliary dimensions.
SCORE_HEAD_WEIGHTS = (1, 5)


def audit_hash(identity: str) -> str:
    payload = f"openbiliclaw:evaluator-prefilter:v1\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_age_days(value: str, reference: str) -> float | None:
    if not value:
        return None

    def _parse(text: str) -> datetime | None:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    published = _parse(str(value))
    ref = _parse(str(reference)) if reference else None
    if published is None:
        return None
    if ref is None:
        ref = datetime.now(UTC)
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    return max(0.0, (ref - published).total_seconds() / 86400.0)


def load_audit(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Aggregate at-time prefilter audits per candidate identity hash."""

    rows = conn.execute(
        """
        SELECT candidate_hash, similarity, llm_score, would_filter,
               context_class, created_at
        FROM evaluator_prefilter_shadow_audit
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_hash"])].append(row)
    aggregated: dict[str, dict[str, Any]] = {}
    for hash_key, entries in grouped.items():
        similarities = [float(e["similarity"]) for e in entries if e["similarity"] is not None]
        last = entries[-1]
        aggregated[hash_key] = {
            "teacher_score": last["llm_score"],
            "sim_last": similarities[-1] if similarities else None,
            "sim_max": max(similarities) if similarities else None,
            "sim_min": min(similarities) if similarities else None,
            "sim_mean": float(np.mean(similarities)) if similarities else None,
            "sim_std": float(np.std(similarities)) if len(similarities) > 1 else 0.0,
            "audit_count": len(entries),
            "would_filter_last": int(last["would_filter"] or 0),
            "context_class": str(last["context_class"] or "other"),
        }
    return aggregated


def build_dataset(conn: sqlite3.Connection) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    audit = load_audit(conn)
    columns = (
        "candidate_key, status, source_platform, source_strategy, content_type, "
        "candidate_tier, title, description, body_text, published_at, evaluated_at, "
        "duration, " + ", ".join(ENGAGEMENT_COLUMNS) + ", "
        "relevance_score, style_key, temporal_class"
    )
    rows = conn.execute(
        f"SELECT {columns} FROM discovery_candidates WHERE status IN "
        f"({', '.join('?' for _ in EVALUATED_STATUSES)})",
        tuple(sorted(EVALUATED_STATUSES)),
    ).fetchall()

    data: dict[str, list[Any]] = defaultdict(list)
    stats = Counter()
    for row in rows:
        identity = str(row["candidate_key"])
        entry = audit.get(audit_hash(identity))
        persisted = float(row["relevance_score"] or 0.0)
        if persisted > 0.0:
            label = persisted  # caps only zero scores; non-zero is a genuine judgment
            stats["label_from_candidates"] += 1
        elif entry is not None and entry["teacher_score"] is not None:
            label = float(entry["teacher_score"])  # cap-zeroed, recovered at-time
            stats["label_recovered_from_audit"] += 1
        else:
            stats["dropped_ambiguous_zero"] += 1
            continue

        style = str(row["style_key"] or "")
        temporal = TEMPORAL_MERGES.get(
            str(row["temporal_class"] or ""), str(row["temporal_class"] or "")
        )
        age = parse_age_days(str(row["published_at"] or ""), str(row["evaluated_at"] or ""))
        record = {
            "label": label,
            "style": style,
            "temporal": temporal,
            "platform": str(row["source_platform"] or "unknown"),
            "strategy": str(row["source_strategy"] or "unknown"),
            "content_type": str(row["content_type"] or "video"),
            "tier": str(row["candidate_tier"] or "primary"),
            "title_len": len(str(row["title"] or "")),
            "desc_len": len(str(row["description"] or "")),
            "body_len": len(str(row["body_text"] or "")),
            "duration_s": float(row["duration"] or 0),
            "age_days": age,
            **{col: float(row[col] or 0) for col in ENGAGEMENT_COLUMNS},
        }
        if entry is not None:
            record.update(
                {
                    "sim_last": entry["sim_last"],
                    "sim_max": entry["sim_max"],
                    "sim_min": entry["sim_min"],
                    "sim_mean": entry["sim_mean"],
                    "sim_std": entry["sim_std"],
                    "audit_count": float(entry["audit_count"]),
                    "would_filter_last": float(entry["would_filter_last"]),
                    "context": entry["context_class"],
                }
            )
        data["records"].append(record)

    # Label sanity: rows with both sources should agree (re-evals under a
    # different profile digest legitimately drift; report the spread).
    for row in rows:
        persisted = float(row["relevance_score"] or 0.0)
        entry = audit.get(audit_hash(str(row["candidate_key"])))
        if persisted > 0.0 and entry is not None and entry["teacher_score"] is not None:
            delta = abs(persisted - float(entry["teacher_score"]))
            data.setdefault("label_deltas", []).append(delta)

    return data, stats


def style_classes(styles: list[str]) -> tuple[list[str], dict[str, str]]:
    counts = Counter(s for s in styles if s)
    keep = sorted(c for c, n in counts.items() if n >= MIN_STYLE_CLASS_ROWS)
    mapping = {c: (c if c in keep else "style_other") for c in counts}
    return keep + ["style_other"], mapping


def featurize(
    records: list[dict[str, Any]],
    *,
    strategies: list[str],
    contexts: list[str],
    style_names: list[str],
    temporal_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: list[float]) -> None:
        feature_names.append(name)
        columns.append(np.asarray(values, dtype=float))

    sim_keys = ("sim_last", "sim_max", "sim_min", "sim_mean", "sim_std")
    for key in sim_keys:
        add(f"{key}", [float(r.get(key)) if r.get(key) is not None else 0.0 for r in records])
        add(f"{key}_avail", [0.0 if r.get(key) is None else 1.0 for r in records])
    add("audit_count", [r.get("audit_count", 0.0) for r in records])
    add("would_filter_last", [r.get("would_filter_last", 0.0) for r in records])
    for col in ENGAGEMENT_COLUMNS:
        add(f"log1p_{col}", [float(np.log1p(r.get(col, 0.0))) for r in records])
        add(f"{col}_avail", [1.0 if r.get(col, 0.0) > 0 else 0.0 for r in records])
    add("log1p_duration", [float(np.log1p(r.get("duration_s", 0.0))) for r in records])
    add("duration_avail", [1.0 if r.get("duration_s", 0.0) > 0 else 0.0 for r in records])
    add("age_days", [r.get("age_days") if r.get("age_days") is not None else 0.0 for r in records])
    add("age_avail", [1.0 if r.get("age_days") is not None else 0.0 for r in records])
    add("log1p_title_len", [float(np.log1p(r["title_len"])) for r in records])
    add("log1p_desc_len", [float(np.log1p(r["desc_len"])) for r in records])
    add("log1p_body_len", [float(np.log1p(r["body_len"])) for r in records])

    platforms = sorted({r["platform"] for r in records})
    for platform in platforms:
        add(f"platform={platform}", [1.0 if r["platform"] == platform else 0.0 for r in records])
    for strategy in strategies:
        add(
            f"strategy={strategy}",
            [1.0 if r["strategy"] == strategy else 0.0 for r in records],
        )
    for context in contexts:
        add(
            f"context={context}",
            [1.0 if r.get("context") == context else 0.0 for r in records],
        )
    content_types = sorted({r["content_type"] for r in records})
    for ctype in content_types:
        add(
            f"content_type={ctype}",
            [1.0 if r["content_type"] == ctype else 0.0 for r in records],
        )
    tiers = sorted({r["tier"] for r in records})
    for tier in tiers:
        add(f"tier={tier}", [1.0 if r["tier"] == tier else 0.0 for r in records])

    x = np.column_stack(columns)
    y = np.asarray([r["label"] for r in records], dtype=float)

    def onehot(names: list[str], values: list[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((len(values), len(names)), dtype=float)
        for i, value in enumerate(values):
            matrix[i, index[value]] = 1.0
        return matrix

    style_y = onehot(style_names, [r["style"] for r in records])
    temporal_y = onehot(temporal_names, [r["temporal"] for r in records])
    return x, y, style_y, temporal_y, feature_names


def fit_mlp_multi(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=3000,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(x_train, np.asarray(y_train, dtype=float).reshape(len(y_train), -1))
    return model


def evaluate(
    db_path: Path,
    folds: int,
    seeds: int,
) -> dict[str, Any]:
    conn = connect_readonly(db_path)
    data, stats = build_dataset(conn)
    conn.close()
    records = data["records"]
    print(f"dataset: {len(records)} rows; provenance {dict(stats)}")
    deltas = data.get("label_deltas") or []
    if deltas:
        print(
            "label consistency |persisted - audit|: "
            f"mean={np.mean(deltas):.3f} p90={np.percentile(deltas, 90):.3f} max={max(deltas):.2f}"
        )

    style_names, style_map = style_classes([r["style"] for r in records])
    for record in records:
        record["style"] = style_map[record["style"]]
    temporal_names = sorted({r["temporal"] for r in records})
    strategies = [s for s, _ in Counter(r["strategy"] for r in records).most_common(TOP_STRATEGIES)]
    for record in records:
        if record["strategy"] not in strategies:
            record["strategy"] = "strategy_other"
    strategies = strategies + ["strategy_other"]
    contexts = sorted({r.get("context", "no_audit") for r in records})
    for record in records:
        record.setdefault("context", "no_audit")

    x, y, style_y, temporal_y, feature_names = featurize(
        records,
        strategies=strategies,
        contexts=contexts,
        style_names=style_names,
        temporal_names=temporal_names,
    )
    print(f"features: {x.shape[1]} ({len(feature_names)}) x {x.shape[0]} rows")
    print(f"style classes: {len(style_names)}; temporal classes: {len(temporal_names)}")
    style_majority = max(style_y.mean(axis=0))
    temporal_majority = max(temporal_y.mean(axis=0))
    print(
        f"majority baselines: style acc={style_majority:.3f}, temporal acc={temporal_majority:.3f}"
    )

    platforms = np.asarray([r["platform"] for r in records])
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    aux_results: dict[str, list[float]] = defaultdict(list)
    rng_master = np.random.default_rng(0)
    from sklearn.model_selection import StratifiedKFold

    model_names = [
        "ridge",
        "hgb",
        "mlp_single",
        *(f"mlp_multi_w{w}" for w in SCORE_HEAD_WEIGHTS),
        *(f"mlp_multi_z_w{w}" for w in SCORE_HEAD_WEIGHTS),
        "two_stage",
    ]
    for seed in range(seeds):
        fold_seed = int(rng_master.integers(0, 2**31 - 1)) + seed
        # Stratify the fold split on platform x style bucket so per-platform
        # coverage stays balanced in every fold.
        style_arg = style_y.argmax(axis=1)
        strata = [f"{p}|{s}" for p, s in zip(platforms, style_arg, strict=True)]
        rare = {k for k, n in Counter(strata).items() if n < folds}
        strata = [s if s not in rare else f"{s.split('|')[0]}|rare" for s in strata]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
        for train_idx, test_idx in splitter.split(x, strata):
            scaler = StandardScaler().fit(x[train_idx])
            x_train, x_test = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]
            style_train, style_test = style_y[train_idx], style_y[test_idx]
            temporal_train, temporal_test = temporal_y[train_idx], temporal_y[test_idx]

            models: dict[str, Any] = {}
            models["ridge"] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_train, y_train)
            models["hgb"] = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, random_state=seed
            ).fit(x_train, y_train)
            models["mlp_single"] = fit_mlp_multi(x_train, y_train, seed)
            for weight in SCORE_HEAD_WEIGHTS:
                targets = np.hstack(
                    [y_train.reshape(-1, 1)] * weight + [style_train, temporal_train]
                )
                models[f"mlp_multi_w{weight}"] = fit_mlp_multi(x_train, targets, seed)
            # z-scored score block: one-hot targets are ~N(0.08, 0.28) while a
            # raw score column has far smaller variance, so the shared-MSE
            # trunk otherwise spends its capacity on the auxiliary blocks.
            y_mean, y_std = float(y_train.mean()), float(y_train.std() or 1.0)
            for weight in SCORE_HEAD_WEIGHTS:
                z = ((y_train - y_mean) / y_std).reshape(-1, 1)
                targets = np.hstack([z] * weight + [style_train, temporal_train])
                models[f"mlp_multi_z_w{weight}"] = fit_mlp_multi(x_train, targets, seed)

            # Two-stage: aux classifiers predict probabilities that become
            # score-model features (the serving-realistic architecture).
            style_clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                x_train, style_train.argmax(axis=1)
            )
            temporal_clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                x_train, temporal_train.argmax(axis=1)
            )
            x2_train = np.hstack(
                [x_train, style_clf.predict_proba(x_train), temporal_clf.predict_proba(x_train)]
            )
            x2_test = np.hstack(
                [x_test, style_clf.predict_proba(x_test), temporal_clf.predict_proba(x_test)]
            )
            models["two_stage"] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x2_train, y_train)
            two_stage_test_x = x2_test

            for name, model in models.items():
                if name == "two_stage":
                    pred = model.predict(two_stage_test_x)
                elif name.startswith("mlp_multi_z"):
                    pred = model.predict(x_test)[:, 0] * y_std + y_mean
                elif name.startswith("mlp_multi"):
                    pred = model.predict(x_test)[:, 0]
                else:
                    pred = np.asarray(model.predict(x_test)).ravel()
                rho_all = spearmanr(pred, y_test).statistic
                results[name]["rho_all"].append(float(rho_all))
                for platform in ("bilibili", "xiaohongshu"):
                    mask = platforms[test_idx] == platform
                    if mask.sum() >= 5:
                        rho_p = spearmanr(pred[mask], y_test[mask]).statistic
                        results[name][f"rho_{platform}"].append(float(rho_p))
                results[name]["mae"].append(float(np.mean(np.abs(pred - y_test))))

            # Auxiliary-head quality from the shared-trunk model (w5).
            multi = models["mlp_multi_w5"]
            outputs = multi.predict(x_test)
            style_block = outputs[
                :, SCORE_HEAD_WEIGHTS[-1] : SCORE_HEAD_WEIGHTS[-1] + len(style_names)
            ]
            temporal_block = outputs[:, SCORE_HEAD_WEIGHTS[-1] + len(style_names) :]
            aux_results["mlp_multi_style_acc"].append(
                float((style_block.argmax(axis=1) == style_test.argmax(axis=1)).mean())
            )
            aux_results["mlp_multi_temporal_acc"].append(
                float((temporal_block.argmax(axis=1) == temporal_test.argmax(axis=1)).mean())
            )
            aux_results["two_stage_style_acc"].append(
                float((style_clf.predict(x_test) == style_test.argmax(axis=1)).mean())
            )
            aux_results["two_stage_temporal_acc"].append(
                float((temporal_clf.predict(x_test) == temporal_test.argmax(axis=1)).mean())
            )

    print()
    print(f"{'model':16s} {'spearman':>16s} {'mae':>8s} {'rho_bili':>16s} {'rho_xhs':>16s}")
    summary: dict[str, Any] = {"dataset": dict(stats), "rows": len(records)}
    for name in model_names:
        metrics = results[name]
        rho = metrics["rho_all"]
        print(
            f"{name:16s} {np.mean(rho):7.3f}±{np.std(rho):5.3f} "
            f"{np.mean(metrics['mae']):8.3f} "
            f"{np.mean(metrics['rho_bilibili']):7.3f}±{np.std(metrics['rho_bilibili']):5.3f} "
            f"{np.mean(metrics['rho_xiaohongshu']):7.3f}±{np.std(metrics['rho_xiaohongshu']):5.3f}"
        )
        summary[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
    print()
    print("auxiliary heads (held-out accuracy vs majority):")
    summary["aux"] = {}
    for key, values in aux_results.items():
        majority = style_majority if "style" in key else temporal_majority
        print(f"  {key:28s} {np.mean(values):.3f}±{np.std(values):.3f} (majority {majority:.3f})")
        summary["aux"][key] = {"acc": float(np.mean(values)), "majority": float(majority)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--json", type=Path, default=None, help="dump metrics summary here")
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
        print(f"\nmetrics written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
