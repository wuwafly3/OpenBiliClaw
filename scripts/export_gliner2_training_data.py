"""Export teacher-labeled candidates as GLiNER2 classification JSONL.

Washes the discovery teacher allowlist into the format documented in
GLiNER2 tutorial 8 (train_data): one JSON object per line::

    {"input": "...", "output": {"classifications": [
        {"task": "temporal", "labels": [...], "true_label": ["evergreen"],
         "label_descriptions": {...}}]}}

Washing rules: teacher allowlist only (``score_source`` LLM-judgment +
non-null ``llm_score_raw``); drop ``unknown``/empty labels; require
complete temporal evidence and confidence >= 0.5 for the temporal task;
dedupe on normalized title+description; stratified 90/10 split by label
with a fixed seed.

Usage::

    uv run --extra dev python scripts/export_gliner2_training_data.py \\
        --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \\
        --out-dir data/gliner2_training --task temporal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_ROOT = Path("E:/otherproject/OpenBiliClaw")
DEFAULT_DB = LIVE_ROOT / "data" / "openbiliclaw.db"
DEFAULT_OUT_DIR = LIVE_ROOT / "data" / "gliner2_training"

TEMPORAL_LABELS = ("breaking", "current", "evergreen", "historical", "versioned")
TEMPORAL_DESCRIPTIONS = {
    "breaking": "刚发生或首发的突发内容，时效以小时计",
    "current": "近期热点或当下流行，时效以天到周计",
    "evergreen": "长期有效的教程、科普、作品内容，价值不依赖时间",
    "historical": "对已闭合事件或过去年代的回顾、考据、档案",
    "versioned": "指涉可识别且仍在迭代的具体对象，更新后价值衰减",
}
STYLE_KEYS = [key for key, _ in (
    ("deep_focus", "深度专注:原理、结构、系统分析、长线思考"),
    ("quick_scan", "快速扫信息:热点、更新、短知识、资讯变化"),
    ("hands_on", "跟做学习:教程、攻略、实操步骤、解决问题"),
    ("decision_support", "辅助决策:测评、盘点、对比、购买前参考"),
    ("story_immersion", "叙事沉浸:纪录片、人物故事、事件复盘"),
    ("opinion_sparring", "观点碰撞:评论、立场、辩论、锐评"),
    ("social_chat", "陪聊/对谈:闲聊、访谈、播客式内容"),
    ("daily_wander", "日常漫游:vlog、生活流、低目标浏览"),
    ("mood_release", "情绪释放:搞笑、整活、吐槽、二创"),
    ("aesthetic_browse", "审美浏览:视觉、混剪、空镜、作品展示"),
    ("ambient_companion", "背景陪伴:背景音乐、白噪音、长时间陪伴"),
    ("live_pulse", "现场脉冲:直播切片、演出现场、赛事高光"),
    ("curiosity_spark", "新鲜猎奇:奇怪事实、冷门切口、意外发现"),
)]

from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES  # noqa: E402
from openbiliclaw.storage.database import Database  # noqa: E402


def load_rows(database: Database, *, task: str) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in LLM_JUDGMENT_SCORE_SOURCES)
    cursor = database.conn.execute(
        f"""
        SELECT id, source_platform, title, description,
               temporal_class, temporal_confidence, temporal_evidence_complete,
               style_key
        FROM discovery_candidates
        WHERE score_source IN ({placeholders})
          AND llm_score_raw IS NOT NULL
        ORDER BY id DESC
        """,
        tuple(sorted(LLM_JUDGMENT_SCORE_SOURCES)),
    )
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    dropped_unknown = dropped_low_conf = dropped_dupes = 0
    for row in cursor.fetchall():
        record = dict(row)
        text = " ".join(
            f"{record.get('title') or ''} {record.get('description') or ''}".split()
        )
        if not text.strip():
            continue
        if task == "temporal":
            label = str(record.get("temporal_class") or "").strip().lower()
            if not label or label == "unknown":
                dropped_unknown += 1
                continue
            if float(record.get("temporal_confidence") or 0.0) < 0.5:
                dropped_low_conf += 1
                continue
            if int(record.get("temporal_evidence_complete") or 0) != 1:
                dropped_low_conf += 1
                continue
        else:
            from openbiliclaw.discovery.style_keys import normalize_style_key

            label = normalize_style_key(record.get("style_key"))
            if not label:
                dropped_unknown += 1
                continue
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            dropped_dupes += 1
            continue
        seen_hashes.add(digest)
        rows.append({"id": int(record["id"]), "text": text[:400], "label": label})
    print(f"  washed rows           {len(rows)}")
    print(f"  dropped unknown/empty {dropped_unknown}")
    print(f"  dropped low-conf/incomplete {dropped_low_conf}")
    print(f"  dropped duplicates    {dropped_dupes}")
    return rows


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for label in sorted(by_label):
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * val_ratio)) if len(bucket) >= 4 else 1
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    task: str,
    labels: tuple[str, ...],
    descriptions: dict[str, str],
) -> None:
    payload_labels = [str(label) for label in labels]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            entry = {
                "task": task,
                "labels": payload_labels,
                "true_label": [row["label"]],
            }
            if descriptions:
                entry["label_descriptions"] = {
                    k: v for k, v in descriptions.items() if k in payload_labels
                }
            handle.write(
                json.dumps(
                    {"input": row["text"], "output": {"classifications": [entry]}},
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", choices=("temporal", "style"), default="temporal")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    db_path = args.db.expanduser().resolve()
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 1

    task = str(args.task)
    labels = TEMPORAL_LABELS if task == "temporal" else tuple(STYLE_KEYS)
    descriptions = TEMPORAL_DESCRIPTIONS if task == "temporal" else {}

    database = Database(db_path)
    database.initialize()
    try:
        print(f"=== GLiNER2 export: {task} ===")
        rows = load_rows(database, task=task)
    finally:
        database.close()
    if len(rows) < 50:
        print("too few usable rows")
        return 2

    train, val = stratified_split(rows, val_ratio=float(args.val_ratio), seed=int(args.seed))
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{task}_train.jsonl"
    val_path = out_dir / f"{task}_val.jsonl"
    write_jsonl(train_path, train, task=task, labels=labels, descriptions=descriptions)
    write_jsonl(val_path, val, task=task, labels=labels, descriptions=descriptions)

    print("=== split ===")
    print(f"  train {len(train)}  -> {train_path}")
    print(f"  val   {len(val)}  -> {val_path}")
    print(f"  train dist {dict(Counter(r['label'] for r in train).most_common())}")
    print(f"  val   dist {dict(Counter(r['label'] for r in val).most_common())}")

    # Leakage guard: no input text may appear in both splits.
    train_texts = {r["text"] for r in train}
    leaked = sum(1 for r in val if r["text"] in train_texts)
    print(f"  leakage check: {leaked} overlapping texts between train and val")
    return 0 if leaked == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
