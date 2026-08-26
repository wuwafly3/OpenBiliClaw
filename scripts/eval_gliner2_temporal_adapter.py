"""Evaluate a GLiNER2 temporal LoRA adapter against the held-out val set.

Loads ``fastino/gliner2.5-multi-v1`` (plus an optional adapter), runs
script-routed temporal classification over the validation JSONL and reports
overall / zh / en exact match plus per-class recall and confusion summary.
Zero-shot numbers print first whenever no adapter is loaded yet, so one run
gives both sides of the comparison.

Usage::

    uv run --extra dev python scripts/eval_gliner2_temporal_adapter.py \\
        --adapter data/ml_artifacts/gliner2_temporal_lora/best
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_VAL = LIVE_ROOT / "data" / "gliner2_training" / "temporal_val.jsonl"
DEFAULT_ADAPTER = LIVE_ROOT / "data" / "ml_artifacts" / "gliner2_temporal_lora" / "best"
BASE_MODEL = "fastino/gliner2.5-multi-v1"
CJK_THRESHOLD = 0.2

from export_gliner2_training_data import TEMPORAL_DESCRIPTIONS  # noqa: E402

from openbiliclaw.discovery.gliner_tagger import cjk_ratio  # noqa: E402
from openbiliclaw.discovery.temporal import TEMPORAL_CLASSES  # noqa: E402


def load_val(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entry = obj["output"]["classifications"][0]
        rows.append(
            {
                "text": str(obj["input"] or ""),
                "gold": str(entry["true_label"][0] or ""),
            }
        )
    return rows


def run_eval(model: Any, rows: list[dict[str, str]], tag: str) -> None:
    schema_labels = {k: f"{k}: {d}" for k, d in TEMPORAL_DESCRIPTIONS.items()}
    hits = Counter()
    totals = Counter()
    pred_counts: Counter[str] = Counter()
    for row in rows:
        use_char = cjk_ratio(row["text"]) >= CJK_THRESHOLD
        model.set_word_splitter("char" if use_char else "whitespace")
        schema = model.create_schema().classification(
            "temporal", schema_labels, cls_threshold=0.0
        )
        result = model.extract(row["text"], schema) or {}
        pred = str(result.get("temporal") or "")
        lang = "zh" if use_char else "en"
        for key, ok in ((lang, pred == row["gold"]), ("all", pred == row["gold"])):
            totals[key] += 1
            hits[key] += int(ok)
        totals[f"class:{row['gold']}"] += 1
        hits[f"class:{row['gold']}"] += int(pred == row["gold"])
        pred_counts[pred] += 1
    print(f"[{tag}]")
    for lang in ("all", "zh", "en"):
        n = totals[lang]
        print(f"  {lang:4s} exact {_pct(hits[lang], n)}  ({hits[lang]}/{n})")
    print("  per-class recall:")
    for cls in sorted(TEMPORAL_CLASSES):
        n = totals[f"class:{cls}"]
        if n:
            print(f"    {cls:11s} {_pct(hits[f'class:{cls}'], n)}  ({n} rows)")
    print(f"  prediction distribution: {dict(pred_counts.most_common())}")


def _pct(hits: int, total: int) -> str:
    return f"{hits / total:.1%}" if total else "n/a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip the baseline pass (saves ~half the runtime).",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from gliner2 import AutoExtractor

    val_path = args.val.expanduser().resolve()
    if not val_path.is_file():
        print(f"val jsonl not found: {val_path}")
        return 1
    rows = load_val(val_path)
    print(f"=== GLiNER2 temporal adapter eval ({len(rows)} rows) ===")

    model = AutoExtractor.from_pretrained(str(args.base_model), map_location="cpu")
    if not args.skip_zero_shot:
        run_eval(model, rows, "zero-shot")
    adapter = args.adapter.expanduser().resolve()
    if adapter.is_dir():
        model.load_adapter(str(adapter))
        run_eval(model, rows, f"adapter:{adapter.name}")
    else:
        print(f"adapter dir missing ({adapter}); zero-shot only")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main_async()))
