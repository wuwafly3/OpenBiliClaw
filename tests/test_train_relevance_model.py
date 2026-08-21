"""Unit tests for the teacher-oracle admission trainer.

The script lives in ``scripts/`` (not a package), so it is loaded via
importlib. These tests cover the S1.1 label floor and the feature-isolation
contract: teacher tags are allowed, teacher scores and franchise are not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")

_SPEC = importlib.util.spec_from_file_location(
    "train_relevance_model",
    Path(__file__).resolve().parents[1] / "scripts" / "train_relevance_model.py",
)
assert _SPEC is not None and _SPEC.loader is not None
train_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_module)


def test_binary_label_uses_explore_floor() -> None:
    assert train_module.binary_label(0.58, "explore") == 1
    assert train_module.binary_label(0.59, "search") == 0
    assert train_module.binary_label(0.60, "search") == 1
    assert train_module.binary_label(0.59, "explore-backfill") == 0


def _record(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_key": "k1",
        "teacher_score": 0.8,
        "y": 1,
        "platform": "bilibili",
        "strategy": "search",
        "content_type": "video",
        "style": "deep_focus",
        "temporal": "evergreen",
        "topic": "机器学习",
        "title_len": 12,
        "desc_len": 40,
        "body_len": 0,
        "duration_s": 600.0,
        "rating_score": 0.0,
        "source_rank": 0.0,
        "profile_digest": "abc",
        "sim": 0.4,
        "would_filter": 0.0,
        "context": "eval",
        **{col: 0.0 for col in train_module.ENGAGEMENT_COLUMNS},
    }
    row.update(overrides)
    return row


def test_teacher_tag_features_exclude_scores_and_franchise() -> None:
    matrix, names, vocab = train_module.encode_teacher_features(
        [
            _record(),
            _record(
                candidate_key="k2",
                y=0,
                teacher_score=0.2,
                style="quick_scan",
                temporal="current",
                topic="新闻",
                platform="xiaohongshu",
            ),
        ]
    )
    joined = " ".join(names)
    assert "style=deep_focus" in names
    assert "temporal=evergreen" in names
    assert "topic=机器学习" in names or any(name.startswith("topic=") for name in names)
    assert "relevance_score" not in joined
    assert "llm_score_raw" not in joined
    assert "teacher_score" not in joined
    assert "franchise" not in joined
    assert matrix.shape == (2, len(names))
    assert vocab["topics"]


def test_filter_records_with_candidate_profile_digest_ignores_audit_fallback() -> None:
    kept, dropped = train_module.filter_records_with_candidate_profile_digest(
        [
            _record(candidate_key="k1", candidate_profile_digest="abc", profile_digest="abc"),
            _record(candidate_key="k2", candidate_profile_digest="", profile_digest="audit"),
            _record(candidate_key="k3", candidate_profile_digest="  ", profile_digest="xyz"),
        ]
    )
    assert [row["candidate_key"] for row in kept] == ["k1"]
    assert dropped == 2
