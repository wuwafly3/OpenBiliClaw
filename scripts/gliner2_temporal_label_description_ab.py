"""A/B the temporal label descriptions on zero-shot GLiNER2.5 (no adapter).

Runs the base model (``fastino/gliner2.5-multi-v1``) with NO LoRA on the held-out
val set, restricted to a label subset, once with the current
``TEMPORAL_DESCRIPTIONS`` and once with the pre-2026-08 descriptions, so the
description change is the only variable. Reuses ``eval_gliner2_temporal_adapter``
by monkeypatching its module-level descriptions.

Usage::

    uv run --extra dev python scripts/gliner2_temporal_label_description_ab.py \\
        --labels versioned,historical,evergreen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_VAL = LIVE_ROOT / "data" / "gliner2_training" / "temporal_val.jsonl"
BASE_MODEL = "fastino/gliner2.5-multi-v1"

import eval_gliner2_temporal_adapter as ev  # noqa: E402
from export_gliner2_training_data import TEMPORAL_DESCRIPTIONS as NEW_DESC  # noqa: E402

# Pre-2026-08 descriptions (before the referent-iteration discriminator).
OLD_DESC = {
    "breaking": "刚发生或首发的突发内容，时效以小时计",
    "current": "近期热点或当下流行，时效以天到周计",
    "evergreen": "长期有效的教程、科普、作品内容，不依赖时间",
    "historical": "回顾历史事件、过往年代的内容",
    "versioned": "与特定版本号强绑定，版本更新后即过时",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--labels",
        default="versioned,historical,evergreen",
        help="Comma-separated labels to restrict both schema and rows to.",
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
    rows = ev.load_val(val_path)
    labels = tuple(
        cls
        for cls in (part.strip() for part in str(args.labels).split(","))
        if cls in ev.TEMPORAL_CLASSES
    )
    print(f"=== GLiNER2 temporal label-description A/B ({len(rows)} rows) ===")
    print(f"restricting to labels: {', '.join(labels)}")

    model = AutoExtractor.from_pretrained(str(args.base_model), map_location="cpu")
    ev.TEMPORAL_DESCRIPTIONS = NEW_DESC
    ev.run_eval(model, rows, "new-descriptions", labels=labels)
    ev.TEMPORAL_DESCRIPTIONS = OLD_DESC
    ev.run_eval(model, rows, "old-descriptions", labels=labels)
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main_async()))
