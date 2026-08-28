"""Diagnose the versioned class: val confusion + training-label quality.

1. Runs the v2 adapter (and zero-shot) over the val set and prints every
   versioned-gold row: title + gold vs predicted class, so we can see exactly
   what "versioned" content is being mislabeled as.
2. Samples versioned rows from the training JSONL to eyeball label quality.

Usage::

    uv run --extra dev python scripts/gliner2_temporal_versioned_diag.py \\
        --adapter data/ml_artifacts/gliner2_temporal_lora_v2/best
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_VAL = LIVE_ROOT / "data" / "gliner2_training" / "temporal_val.jsonl"
DEFAULT_TRAIN = LIVE_ROOT / "data" / "gliner2_training" / "temporal_train.jsonl"
DEFAULT_ADAPTER = LIVE_ROOT / "data" / "ml_artifacts" / "gliner2_temporal_lora_v2" / "best"
BASE_MODEL = "fastino/gliner2.5-multi-v1"

from openbiliclaw.discovery.gliner_tagger import cjk_ratio  # noqa: E402

CJK_THRESHOLD = 0.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--sample-train", type=int, default=25)
    return parser.parse_args(argv)


def load_jsonl_rows(path: Path, label: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entry = obj["output"]["classifications"][0]
        gold = str(entry["true_label"][0] or "")
        if label is not None and gold != label:
            continue
        rows.append({"text": str(obj["input"] or ""), "gold": gold})
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    val_path = args.val.expanduser().resolve()
    train_path = args.train.expanduser().resolve()
    if not val_path.is_file() or not train_path.is_file():
        print(f"missing data: {val_path} / {train_path}")
        return 1

    # ---- (1) versioned confusion on val ----
    versioned_val = load_jsonl_rows(val_path, label="versioned")
    print(f"=== versioned val rows: {len(versioned_val)} ===")

    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(str(args.base_model), map_location="cpu")
    schema_labels = {
        "breaking": "breaking: 刚发生/首发的突发内容，时效以小时计",
        "current": "current: 近期热点或当下流行，时效以天到周计",
        "evergreen": "evergreen: 长期有效的教程/科普/作品内容，价值不依赖时间",
        "historical": "historical: 对已闭合事件或过去年代的回顾、考据、档案",
        "versioned": "versioned: 指涉可识别且仍在迭代的具体对象，更新后价值衰减",
    }

    def classify(rows: list[dict[str, str]], tag: str) -> Counter[str]:
        preds: Counter[str] = Counter()
        for row in rows:
            use_char = cjk_ratio(row["text"]) >= CJK_THRESHOLD
            model.set_word_splitter("char" if use_char else "whitespace")
            schema = model.create_schema().classification("temporal", schema_labels, cls_threshold=0.0)
            result = model.extract(row["text"], schema) or {}
            pred = str(result.get("temporal") or "")
            preds[pred] += 1
        return preds

    print("  zero-shot pred distribution (versioned-gold rows):")
    print(f"    {dict(classify(versioned_val, 'zero-shot').most_common())}")

    adapter = args.adapter.expanduser().resolve()
    if adapter.is_dir():
        model.load_adapter(str(adapter))
        print("  v2-adapter pred distribution (versioned-gold rows):")
        preds = classify(versioned_val, "adapter")
        print(f"    {dict(preds.most_common())}")
        # print each versioned row with its prediction
        print("  per-row (v2 adapter):")
        for row in versioned_val:
            use_char = cjk_ratio(row["text"]) >= CJK_THRESHOLD
            model.set_word_splitter("char" if use_char else "whitespace")
            schema = model.create_schema().classification("temporal", schema_labels, cls_threshold=0.0)
            result = model.extract(row["text"], schema) or {}
            pred = str(result.get("temporal") or "")
            marker = "OK " if pred == "versioned" else "MIS"
            text = row["text"].replace("\n", " ")[:70]
            print(f"    [{marker}] gold=versioned pred={pred:10s} | {text}")
    else:
        print(f"  adapter missing: {adapter}")

    # ---- (2) training-set versioned label quality ----
    versioned_train = load_jsonl_rows(train_path, label="versioned")
    print(f"\n=== versioned train rows: {len(versioned_train)} ===")
    n = min(int(args.sample_train), len(versioned_train))
    print(f"  sample {n} titles:")
    for row in versioned_train[:n]:
        print(f"    - {row['text'].replace(chr(10), ' ')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
