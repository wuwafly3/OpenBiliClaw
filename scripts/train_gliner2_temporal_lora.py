"""LoRA fine-tuning pilot for the GLiNER2.5 temporal classification task.

Trains a LoRA adapter on top of ``fastino/gliner2.5-multi-v1`` using the
teacher-washed JSONL produced by ``export_gliner2_training_data.py``.
CPU-friendly by default: small epoch count, adapter-only checkpoint.

The model's word splitter is switched to ``char`` for training — it keeps
Latin words intact and splits CJK per character, which matches the
inference-side routing (whitespace tokenization collapses Chinese
sentences into single tokens).

Usage::

    uv run --extra dev python scripts/train_gliner2_temporal_lora.py \\
        --train data/gliner2_training/temporal_train.jsonl \\
        --val data/gliner2_training/temporal_val.jsonl \\
        --max-train 1200 --epochs 2          # pilot; drop --max-train for full
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_TRAIN = LIVE_ROOT / "data" / "gliner2_training" / "temporal_train.jsonl"
DEFAULT_VAL = LIVE_ROOT / "data" / "gliner2_training" / "temporal_val.jsonl"
DEFAULT_OUT = LIVE_ROOT / "data" / "ml_artifacts" / "gliner2_temporal_lora"
BASE_MODEL = "fastino/gliner2.5-multi-v1"


def maybe_subsample(path: Path, max_rows: int, seed: int) -> Path | None:
    """Write a shuffled subset of the train JSONL to a sibling temp file."""

    if max_rows <= 0:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_rows:
        return None
    rng = random.Random(seed)
    rng.shuffle(lines)
    subset = path.with_name(f"{path.stem}_subset{max_rows}.jsonl")
    subset.write_text("\n".join(lines[:max_rows]) + "\n", encoding="utf-8")
    print(f"  pilot subsample       {max_rows}/{len(lines)} -> {subset.name}")
    return subset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Cap training rows for the pilot run; 0 uses the full JSONL.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    train_path = args.train.expanduser().resolve()
    val_path = args.val.expanduser().resolve()
    if not train_path.is_file() or not val_path.is_file():
        print(f"missing data: {train_path} / {val_path}")
        return 1

    print("=== GLiNER2 temporal LoRA pilot ===")
    print(f"  base model            {args.base_model}")
    print(f"  train / val           {train_path.name} / {val_path.name}")

    model = AutoExtractor.from_pretrained(str(args.base_model), map_location="cpu")
    # char splitter keeps Latin words whole and splits CJK per character —
    # correct for our mixed zh/en corpus under one global setting.
    try:
        model.set_word_splitter("char")
        print("  word splitter         char")
    except Exception as exc:
        print(f"  word splitter         kept default ({exc})")

    out_dir = args.out_dir.expanduser().resolve()
    config = TrainingConfig(
        output_dir=str(out_dir),
        experiment_name=f"temporal_lora_r{int(args.lora_r)}",
        num_epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        gradient_accumulation_steps=max(1, int(args.gradient_accumulation)),
        encoder_lr=float(args.encoder_lr),
        task_lr=float(args.task_lr),
        use_lora=True,
        lora_r=int(args.lora_r),
        lora_alpha=16.0,
        lora_dropout=0.0,
        save_adapter_only=True,
        eval_strategy="epoch",
        save_best=True,
        logging_steps=10,
        seed=int(args.seed),
        # Windows spawn + pickled worker processes break inside the sandboxed
        # runtime; in-process loading is fast enough for these data sizes.
        num_workers=0,
    )

    trainer = ExtractorTrainer(model, config)
    train_data = (
        maybe_subsample(train_path, int(args.max_train), int(args.seed)) or str(train_path)
    )
    started = time.monotonic()
    results = trainer.train(train_data=train_data, eval_data=str(val_path))
    elapsed = time.monotonic() - started
    print("=== done ===")
    print(f"  best val metric       {results.get('best_metric')}")
    print(f"  total steps           {results.get('total_steps')}")
    print(f"  wall time             {elapsed / 60:.1f} min")
    print(f"  adapter dir           {out_dir}")
    best = out_dir / "best"
    if best.is_dir():
        print(f"  best checkpoint       {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
