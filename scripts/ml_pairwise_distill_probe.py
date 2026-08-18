"""Offline probe: pairwise (batch-relative) distillation vs pointwise MSE.

Experiment for ml-ranking problem #3: the teacher scores candidates inside
batches of ~30 with a listwise rubric, so the same content's absolute score
shifts with its batch neighbors. Measured on the real database: between-batch
std of batch means is 0.118 — as large as the mean within-batch std (0.137),
with batch means spanning 0.31..0.80. Absolute regression must spend capacity
fitting that level shift; a within-batch pairwise loss (RankNet-style) is
invariant to it by construction.

Batch identity is recovered from ``discovery_candidates.evaluated_at`` —
one CURRENT_TIMESTAMP per persisted evaluation batch, clusters of <= 30.

Teacher labels follow ``export_ranking_dataset.resolve_teacher_label``:
LLM-judgment ``score_source`` uses ``llm_score_raw``; legacy empty source
keeps a non-zero ``relevance_score`` or recovers from the shadow audit;
prefilter / viewed / truncated / error rows are dropped even when
``relevance_score`` is non-zero.

This probe trains identical numpy MLPs (64-32-1, Adam) differing only in the
loss — pointwise MSE / batch-pairwise logistic / hybrid — plus a Ridge
reference, then routes every model through an isotonic calibration layer fit
on train-fold predictions (the S1.5 mechanism: thresholds live on the
teacher's absolute scale, so ML scores must be mapped back before the
0.60 / 0.75 lines mean anything). Metrics mirror spec S1.4: Spearman rho
(global + per platform), pooled top-25 Jaccard, admission-line agreement and
false-admit rate at 0.60, delight-line agreement at 0.75.

Usage:
    python scripts/ml_pairwise_distill_probe.py [db] [--folds 5] [--seeds 3] \
        [--epochs 400] [--json out.json]

Requires numpy + scikit-learn (training environment only).
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from export_ranking_dataset import (  # noqa: E402
    candidate_row_for_label,
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
# Pairs whose teacher gap is below this are treated as ties and skipped:
# the rubric's 0.05 granularity is noise, not signal.
PAIR_MARGIN = 0.05
MAX_PAIRS_PER_EPOCH = 2000
ADMISSION_LINE = 0.60
DELIGHT_LINE = 0.75
TOP_K = 25


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
            "audit_count": float(len(entries)),
            "would_filter_last": float(last["would_filter"] or 0),
            "context_class": str(last["context_class"] or "other"),
        }
    return aggregated


def build_dataset(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rows with untruncated teacher labels + recovered batch identity.

    Provenance policy identical to ``export_ranking_dataset.resolve_teacher_label``.
    """

    audit = load_audit(conn)
    columns = (
        "candidate_key, status, source_platform, source_strategy, content_type, "
        "candidate_tier, title, description, body_text, published_at, evaluated_at, "
        "duration, " + ", ".join(ENGAGEMENT_COLUMNS) + ", "
        "relevance_score" + teacher_label_select_columns(conn)
    )
    rows = conn.execute(
        f"SELECT {columns} FROM discovery_candidates WHERE status IN "
        f"({', '.join('?' for _ in EVALUATED_STATUSES)})",
        tuple(sorted(EVALUATED_STATUSES)),
    ).fetchall()

    records: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for row in rows:
        entry = audit.get(audit_hash(str(row["candidate_key"])))
        audit_score = (
            float(entry["teacher_score"])
            if entry is not None and entry["teacher_score"] is not None
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
        age = parse_age_days(str(row["published_at"] or ""), str(row["evaluated_at"] or ""))
        record: dict[str, Any] = {
            "label": label,
            "batch": str(row["evaluated_at"] or ""),
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
            record.update(entry)
        records.append(record)
    return records, dict(stats)


def featurize(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic evaluation-time features (no teacher outputs)."""

    feature_names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: list[float]) -> None:
        feature_names.append(name)
        columns.append(np.asarray(values, dtype=float))

    for key in ("sim_last", "sim_max", "sim_min", "sim_mean", "sim_std"):
        add(key, [float(r.get(key)) if r.get(key) is not None else 0.0 for r in records])
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
    for platform in sorted({r["platform"] for r in records}):
        add(f"platform={platform}", [1.0 if r["platform"] == platform else 0.0 for r in records])
    strategies = [s for s, _ in Counter(r["strategy"] for r in records).most_common(TOP_STRATEGIES)]
    for strategy in strategies:
        add(f"strategy={strategy}", [1.0 if r["strategy"] == strategy else 0.0 for r in records])
    add(
        "strategy=other",
        [1.0 if r["strategy"] not in strategies else 0.0 for r in records],
    )
    for context in sorted({r.get("context_class", "no_audit") for r in records}):
        add(
            f"context={context}",
            [1.0 if r.get("context_class") == context else 0.0 for r in records],
        )
    for ctype in sorted({r["content_type"] for r in records}):
        add(f"content_type={ctype}", [1.0 if r["content_type"] == ctype else 0.0 for r in records])
    for tier in sorted({r["tier"] for r in records}):
        add(f"tier={tier}", [1.0 if r["tier"] == tier else 0.0 for r in records])

    x = np.column_stack(columns)
    y = np.asarray([r["label"] for r in records], dtype=float)
    batches = np.asarray([r["batch"] for r in records])
    return x, y, batches


# ── numpy MLP: identical architecture for every loss variant ────────────────


class NumpyMLP:
    """Two-hidden-layer ReLU MLP with Adam, trained full-batch.

    Kept in numpy so the pointwise/pairwise/hybrid variants differ ONLY in
    the loss; this also prototypes the S1.2 runtime-inference constraint.
    """

    def __init__(self, dim_in: int, seed: int, hidden: tuple[int, int] = (64, 32)) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2.0 / dim_in), (dim_in, hidden[0]))
        self.b1 = np.zeros(hidden[0])
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden[0]), (hidden[0], hidden[1]))
        self.b2 = np.zeros(hidden[1])
        self.w3 = rng.normal(0, np.sqrt(2.0 / hidden[1]), (hidden[1], 1))
        self.b3 = np.zeros(1)
        self._adam_state: dict[str, dict[str, np.ndarray]] = {}
        self._t = 0

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("w1", self.w1),
            ("b1", self.b1),
            ("w2", self.w2),
            ("b2", self.b2),
            ("w3", self.w3),
            ("b3", self.b3),
        ]

    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        out = (h2 @ self.w3 + self.b3).ravel()
        return out, (x, h1, h2)

    def backward(
        self,
        grad_out: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> dict[str, np.ndarray]:
        x, h1, h2 = cache
        grads: dict[str, np.ndarray] = {}
        grad_out = grad_out.reshape(-1, 1)
        grads["w3"] = h2.T @ grad_out
        grads["b3"] = grad_out.sum(axis=0)
        g2 = grad_out @ self.w3.T
        g2 = g2 * (h2 > 0)
        grads["w2"] = h1.T @ g2
        grads["b2"] = g2.sum(axis=0)
        g1 = g2 @ self.w2.T
        g1 = g1 * (h1 > 0)
        grads["w1"] = x.T @ g1
        grads["b1"] = g1.sum(axis=0)
        return grads

    def adam_step(self, grads: dict[str, np.ndarray], lr: float, weight_decay: float) -> None:
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in self.params():
            g = grads[name] + weight_decay * param
            state = self._adam_state.setdefault(
                name, {"m": np.zeros_like(param), "v": np.zeros_like(param)}
            )
            state["m"] = beta1 * state["m"] + (1 - beta1) * g
            state["v"] = beta2 * state["v"] + (1 - beta2) * (g * g)
            m_hat = state["m"] / (1 - beta1**self._t)
            v_hat = state["v"] / (1 - beta2**self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]


def sample_pairs(y: np.ndarray, batches: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices (i, j) of same-batch rows whose teacher gap clears the margin."""

    by_batch: dict[str, list[int]] = defaultdict(list)
    for idx, batch in enumerate(batches):
        by_batch[str(batch)].append(idx)
    pairs: list[tuple[int, int]] = []
    for indices in by_batch.values():
        for pos_i in range(len(indices)):
            for pos_j in range(pos_i + 1, len(indices)):
                i, j = indices[pos_i], indices[pos_j]
                if y[i] > y[j] + PAIR_MARGIN:
                    pairs.append((i, j))
                elif y[j] > y[i] + PAIR_MARGIN:
                    pairs.append((j, i))
    if len(pairs) > MAX_PAIRS_PER_EPOCH:
        chosen = rng.choice(len(pairs), size=MAX_PAIRS_PER_EPOCH, replace=False)
        pairs = [pairs[k] for k in chosen]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    batches: np.ndarray,
    *,
    seed: int,
    mode: str,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> NumpyMLP:
    model = NumpyMLP(x.shape[1], seed=seed)
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        out, cache = model.forward(x)
        grads_out = np.zeros_like(y)
        if mode in ("pointwise", "hybrid"):
            residual = out - y
            grads_out += 2.0 * residual / len(y)
        if mode in ("pairwise", "hybrid"):
            pairs = sample_pairs(y, batches, rng)
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                diff = out[i] - out[j]
                # softplus(-diff) loss; d/diff = -(1 - sigmoid(diff))
                coef = -(1.0 - 1.0 / (1.0 + np.exp(-diff)))
                np.add.at(grads_out, i, coef / len(pairs))
                np.add.at(grads_out, j, -coef / len(pairs))
        grads = model.backward(grads_out, cache)
        model.adam_step(grads, lr=lr, weight_decay=weight_decay)
    return model


def topk_jaccard(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    top_pred = set(np.argsort(-pred)[:k].tolist())
    top_true = set(np.argsort(-y)[:k].tolist())
    return len(top_pred & top_true) / len(top_pred | top_true)


def within_batch_metric(
    pred: np.ndarray,
    y: np.ndarray,
    batches: np.ndarray,
    *,
    kind: str,
) -> float:
    """Mean per-batch Spearman / top-k Jaccard over test batches.

    The teacher's listwise rubric only orders candidates inside their own
    batch; cross-batch comparability is manufactured later by the isotonic
    layer. Pooled global metrics are therefore biased against batch-relative
    objectives — the spec S1.4 contract ("同一候选池") is within-batch.
    """

    values: list[float] = []
    for batch in np.unique(batches):
        mask = batches == batch
        if mask.sum() < 5:
            continue
        if kind == "rho":
            values.append(float(spearmanr(pred[mask], y[mask]).statistic))
        else:
            k = min(10, int(mask.sum()) // 2)
            values.append(topk_jaccard(pred[mask], y[mask], k))
    return float(np.mean(values)) if values else 0.0


def line_metrics(cal: np.ndarray, y: np.ndarray, line: float) -> tuple[float, float]:
    agree = float(np.mean((cal >= line) == (y >= line)))
    admitted = cal >= line
    false_admit = float(np.mean(y[admitted] < line)) if admitted.any() else 0.0
    return agree, false_admit


def evaluate(db_path: Path, folds: int, seeds: int, epochs: int) -> dict[str, Any]:
    conn = connect_readonly(db_path)
    records, stats = build_dataset(conn)
    conn.close()
    x, y, batches = featurize(records)
    platforms = np.asarray([r["platform"] for r in records])
    print(f"dataset: {len(records)} rows; provenance {stats}")
    print(f"features: {x.shape[1]}; batches: {len(set(batches.tolist()))}")
    batch_means = [y[batches == b].mean() for b in set(batches.tolist())]
    print(
        f"batch-level shift: between-batch std of means={np.std(batch_means):.3f}, "
        f"mean within-batch std={np.mean([y[batches == b].std() for b in set(batches.tolist())]):.3f}"
    )

    variant_names = ["ridge", "mlp_pointwise", "mlp_pairwise", "mlp_hybrid"]
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rng_master = np.random.default_rng(11)
    for seed in range(seeds):
        fold_seed = int(rng_master.integers(0, 2**31 - 1)) + seed
        strata = [f"{p}|{'hi' if v >= 0.6 else 'lo'}" for p, v in zip(platforms, y, strict=True)]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
        for train_idx, test_idx in splitter.split(x, strata):
            scaler = StandardScaler().fit(x[train_idx])
            x_train, x_test = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            preds: dict[str, np.ndarray] = {}
            preds["ridge"] = (
                RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_train, y_train).predict(x_test)
            )
            for mode, name in (
                ("pointwise", "mlp_pointwise"),
                ("pairwise", "mlp_pairwise"),
                ("hybrid", "mlp_hybrid"),
            ):
                model = train_mlp(
                    x_train,
                    y_train,
                    batches[train_idx],
                    seed=seed,
                    mode=mode,
                    epochs=epochs,
                )
                preds[name] = model.predict(x_test)

            for name, pred in preds.items():
                results[name]["rho"].append(float(spearmanr(pred, y_test).statistic))
                results[name]["jaccard25"].append(topk_jaccard(pred, y_test, TOP_K))
                results[name]["rho_inbatch"].append(
                    within_batch_metric(pred, y_test, batches[test_idx], kind="rho")
                )
                results[name]["jac_inbatch"].append(
                    within_batch_metric(pred, y_test, batches[test_idx], kind="jaccard")
                )
                # S1.5 calibration: isotonic fit on train-fold predictions.
                # In-sample for the MLP variants (slight optimism, noted in
                # the probe doc); the ranking metrics above are unaffected.
                train_model = (
                    RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_train, y_train).predict(x_train)
                    if name == "ridge"
                    else train_mlp(
                        x_train,
                        y_train,
                        batches[train_idx],
                        seed=seed,
                        mode=name.removeprefix("mlp_"),
                        epochs=epochs,
                    ).predict(x_train)
                )
                isotonic = IsotonicRegression(out_of_bounds="clip").fit(train_model, y_train)
                cal = isotonic.predict(pred)
                agree_adm, false_admit = line_metrics(cal, y_test, ADMISSION_LINE)
                agree_delight, _ = line_metrics(cal, y_test, DELIGHT_LINE)
                results[name]["agree_adm"].append(agree_adm)
                results[name]["false_admit"].append(false_admit)
                results[name]["agree_delight"].append(agree_delight)
                results[name]["mae_cal"].append(float(np.mean(np.abs(cal - y_test))))
                for platform in ("bilibili", "xiaohongshu"):
                    mask = platforms[test_idx] == platform
                    if mask.sum() >= 5:
                        results[name][f"rho_{platform}"].append(
                            float(spearmanr(pred[mask], y_test[mask]).statistic)
                        )

    print()
    header = (
        f"{'model':14s} {'rho_global':>14s} {'rho_inbatch':>14s} {'jac_in':>8s} "
        f"{'jac@25':>8s} {'adm-agree':>9s} {'false-adm':>9s} {'delight':>8s} {'mae':>6s}"
    )
    print(header)
    summary: dict[str, Any] = {"dataset": stats, "rows": len(records)}
    for name in variant_names:
        metrics = results[name]
        print(
            f"{name:14s} "
            f"{np.mean(metrics['rho']):6.3f}±{np.std(metrics['rho']):5.3f} "
            f"{np.mean(metrics['rho_inbatch']):6.3f}±{np.std(metrics['rho_inbatch']):5.3f} "
            f"{np.mean(metrics['jac_inbatch']):8.3f} "
            f"{np.mean(metrics['jaccard25']):8.3f} "
            f"{np.mean(metrics['agree_adm']):9.3f} "
            f"{np.mean(metrics['false_admit']):9.3f} "
            f"{np.mean(metrics['agree_delight']):8.3f} "
            f"{np.mean(metrics['mae_cal']):6.3f}"
        )
        summary[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
    print()
    print("S1.4-aligned targets for reference: rho>=0.75, jaccard@25>=0.80,")
    print("admission agreement>=0.90 with false-admit<=0.10 (per-platform too).")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
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

    summary = evaluate(args.database, args.folds, args.seeds, args.epochs)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nmetrics written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
