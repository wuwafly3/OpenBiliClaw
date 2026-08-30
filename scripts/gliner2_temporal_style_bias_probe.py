"""Probe: do GLiNER's low-confidence / mislabeled temporal calls carry a style bias?

Runs the v4 adapter over the val set with a JOINT schema (temporal + style,
independent decoder, per-task confidence) and tests three claims that would
justify a style-based fallback:

1. Mislabel concentration: are temporal errors biased toward certain styles?
2. Low-confidence bands: do low-confidence temporal calls err more, and what
   styles do they land in?
3. Fallback potential: for low-confidence calls, does replacing the model's
   argmax with the training-set MAP ``P(temporal | style)`` (from the joint
   prior) improve accuracy?

Usage::

    uv run --extra dev python scripts/gliner2_temporal_style_bias_probe.py \\
        --adapter data/ml_artifacts/gliner2_temporal_lora_v4/best
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_VAL = LIVE_ROOT / "data" / "gliner2_training" / "temporal_val.jsonl"
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_ADAPTER = LIVE_ROOT / "data" / "ml_artifacts" / "gliner2_temporal_lora_v4" / "best"
BASE_MODEL = "fastino/gliner2.5-multi-v1"

from export_gliner2_training_data import TEMPORAL_DESCRIPTIONS  # noqa: E402

from openbiliclaw.discovery.gliner_tagger import cjk_ratio  # noqa: E402
from openbiliclaw.discovery.style_keys import STYLE_KEY_DEFINITIONS  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402

CJK_THRESHOLD = 0.2
TEMPORAL_LABELS = ("breaking", "current", "versioned", "evergreen", "historical")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=BASE_MODEL)
    return parser.parse_args(argv)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def load_val_with_style(
    val_path: Path,
    db_path: Path,
) -> list[dict[str, str]]:
    """Load val rows and attach gold style by matching the exported input text."""
    rows: list[dict[str, str]] = []
    for line in val_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entry = obj["output"]["classifications"][0]
        rows.append(
            {
                "text": str(obj["input"] or ""),
                "gold_temporal": str(entry["true_label"][0] or ""),
            }
        )

    # Build {exported input text -> style_key} from the teacher allowlist.
    style_by_text: dict[str, str] = {}
    database = Database(db_path)
    database.initialize()
    try:
        placeholders = ", ".join("?" for _ in ("llm", "cap_franchise", "cap_style"))
        cursor = database.conn.execute(
            f"""
            SELECT title, description, style_key
            FROM discovery_candidates
            WHERE score_source IN ({placeholders})
              AND llm_score_raw IS NOT NULL
              AND style_key IS NOT NULL AND style_key != ''
            """,
            ("llm", "cap_franchise", "cap_style"),
        )
        for title, description, style_key in cursor.fetchall():
            text = _norm(f"{title} {description}")[:400]
            style_by_text[text] = str(style_key)
    finally:
        database.close()

    matched = 0
    for row in rows:
        style = style_by_text.get(_norm(row["text"])[:400])
        if style:
            matched += 1
        row["gold_style"] = style or ""
    print(f"  gold-style matched     {matched}/{len(rows)}")
    return rows


def build_joint_prior(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    """MAP P(temporal | style) from the val's own gold (proxy for the train prior)."""
    prior: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["gold_style"]:
            prior[row["gold_style"]][row["gold_temporal"]] += 1
    return {
        style: Counter(dict(counter))
        for style, counter in prior.items()
        if sum(counter.values()) > 0
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    val_path = args.val.expanduser().resolve()
    if not val_path.is_file():
        print(f"val jsonl not found: {val_path}")
        return 1

    rows = load_val_with_style(val_path, args.db.expanduser().resolve())
    print(f"=== temporal x style bias probe ({len(rows)} rows) ===")

    from gliner2 import AutoExtractor
    from gliner2.classification import ClassificationConfig, ClassificationSchema
    from gliner2.classification.engine import Classifier

    style_desc = {key: f"{key}: {desc}" for key, desc in STYLE_KEY_DEFINITIONS}
    schema = (
        ClassificationSchema()
        .single("temporal", dict(TEMPORAL_DESCRIPTIONS))
        .single("style", style_desc)
    )
    config = ClassificationConfig(
        decoder="independent",
        candidate_threshold=0.0,
        include_confidence=True,
        max_candidates_per_task=64,
    )

    model = AutoExtractor.from_pretrained(str(args.base_model), map_location="cpu")
    adapter = args.adapter.expanduser().resolve()
    if adapter.is_dir():
        model.load_adapter(str(adapter))
        print(f"  adapter               {adapter}")
    clf = Classifier(model)

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        use_char = cjk_ratio(row["text"]) >= CJK_THRESHOLD
        model.set_word_splitter("char" if use_char else "whitespace")
        result = clf.classify(row["text"], schema, config=config)
        pred_t = str(result.value("temporal") or "")
        conf_t = float(result.confidence("temporal") or 0.0)
        pred_s = str(result.value("style") or "")
        results.append(
            {
                "gold_t": row["gold_temporal"],
                "gold_s": row["gold_style"],
                "pred_t": pred_t,
                "conf_t": conf_t,
                "pred_s": pred_s,
                "wrong": pred_t != row["gold_temporal"],
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    n = len(results)
    wrong = [r for r in results if r["wrong"]]
    print(f"  temporal exact        {(n - len(wrong)) / n:.1%}  ({n - len(wrong)}/{n})")

    # 1. Mislabel concentration by style (gold + predicted).
    err_by_gold_style: Counter[str] = Counter(r["gold_s"] or "?" for r in wrong)
    err_by_pred_style: Counter[str] = Counter(r["pred_s"] or "?" for r in wrong)
    all_by_gold_style = Counter(r["gold_s"] or "?" for r in results)
    print("\n-- [1] mislabel concentration --")
    print("  error rate by GOLD style:")
    for style, tot in sorted(all_by_gold_style.items(), key=lambda kv: -kv[1]):
        if tot == 0:
            continue
        err = err_by_gold_style[style]
        print(f"    {style:16s} n={tot:4d}  err={err / tot:5.1%}")
    print("  error count by PRED style:")
    for style, cnt in err_by_pred_style.most_common():
        print(f"    {style:16s} {cnt}")

    # 2. Confidence bands.
    print("\n-- [2] confidence bands --")
    bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    for lo, hi in bands:
        band = [r for r in results if lo <= r["conf_t"] < hi]
        if not band:
            continue
        err = sum(1 for r in band if r["wrong"])
        styles = Counter(r["pred_s"] or "?" for r in band)
        top = ", ".join(f"{s}:{c}" for s, c in styles.most_common(3))
        print(f"    conf [{lo:.2f},{hi:.2f}) n={len(band):3d}  err={err / len(band):5.1%}  "
              f"top pred styles: {top}")

    # 3. Fallback potential with the MAP P(temporal | gold_style) prior.
    print("\n-- [3] fallback: MAP P(temporal | style) at low confidence --")
    prior = build_joint_prior(rows)
    for low_threshold in (0.7, 0.85):
        low = [r for r in results if r["conf_t"] < low_threshold]
        if not low:
            continue
        model_wrong = sum(1 for r in low if r["wrong"])
        fallback_wrong = 0
        fallback_covered = 0
        for r in low:
            dist = prior.get(r["pred_s"])
            if not dist:
                fallback_wrong += 1
                continue
            fallback_covered += 1
            fallback_pick = dist.most_common(1)[0][0]
            if fallback_pick != r["gold_t"]:
                fallback_wrong += 1
        n_low = len(low)
        print(f"  conf < {low_threshold}: n={n_low}  model err={model_wrong / n_low:.1%}  "
              f"MAP-by-pred-style err={fallback_wrong / n_low:.1%} "
              f"(covered {fallback_covered}/{n_low})")

        # Per-pred-style fallback accuracy detail.
        detail: dict[str, dict[str, Any]] = {}
        for r in low:
            d = detail.setdefault(
                r["pred_s"] or "?",
                {"n": 0, "model_ok": 0, "map_ok": 0, "gold": Counter()},
            )
            d["n"] += 1
            d["model_ok"] += int(not r["wrong"])
            dist = prior.get(r["pred_s"])
            pick = dist.most_common(1)[0][0] if dist else None
            if pick == r["gold_t"]:
                d["map_ok"] += 1
            d["gold"][r["gold_t"]] += 1
        print("    per pred-style on low-conf rows (model_ok vs MAP_ok vs gold dist):")
        for style, d in sorted(detail.items(), key=lambda kv: -kv[1]["n"]):
            print(f"      {style:16s} n={d['n']:3d} model={d['model_ok']:3d} "
                  f"map={d['map_ok']:3d}  gold={dict(d['gold'].most_common(3))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
