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


class _SnapshotLookup:
    def __init__(self, mapping: dict[tuple[str, str], object]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, str]] = []

    def get_evaluation_context_snapshot(
        self,
        *,
        profile_digest: str,
        negative_digest: str,
    ) -> object | None:
        key = (profile_digest, negative_digest)
        self.calls.append(key)
        return self.mapping.get(key)


class _MismatchSnapshot:
    def digests_match(self) -> bool:
        return False


class _OldContractSnapshot:
    profile_summary = {"recent_awareness": [{"observation": "x"}]}

    def digests_match(self) -> bool:
        return True


def test_filter_records_with_verified_snapshot_drops_missing_and_mismatch() -> None:
    lookup = _SnapshotLookup(
        {
            ("ok", "neg"): object(),
            ("bad", "neg"): _MismatchSnapshot(),
            ("old", "neg"): _OldContractSnapshot(),
        }
    )
    kept, stats = train_module.filter_records_with_verified_snapshot(
        [
            _record(
                candidate_key="keep",
                candidate_profile_digest="ok",
                candidate_negative_digest="neg",
                profile_digest="ok",
            ),
            _record(
                candidate_key="empty",
                candidate_profile_digest="",
                candidate_negative_digest="",
                profile_digest="audit",
            ),
            _record(
                candidate_key="profile_only",
                candidate_profile_digest="ok",
                candidate_negative_digest="",
                profile_digest="ok",
            ),
            _record(
                candidate_key="missing",
                candidate_profile_digest="gone",
                candidate_negative_digest="neg",
                profile_digest="gone",
            ),
            _record(
                candidate_key="mismatch",
                candidate_profile_digest="bad",
                candidate_negative_digest="neg",
                profile_digest="bad",
            ),
            _record(
                candidate_key="old_contract",
                candidate_profile_digest="old",
                candidate_negative_digest="neg",
                profile_digest="old",
            ),
        ],
        lookup.get_evaluation_context_snapshot,
    )
    assert [row["candidate_key"] for row in kept] == ["keep"]
    assert stats["kept"] == 1
    assert stats["dropped_empty_digest"] == 2
    assert stats["dropped_no_snapshot"] == 1
    assert stats["dropped_digest_mismatch"] == 1
    assert stats["dropped_old_contract"] == 1
    assert ("gone", "neg") in lookup.calls
    assert ("bad", "neg") in lookup.calls


def test_apply_relabel_overrides_uses_new_score_and_new_digest() -> None:
    records = [
        _record(
            candidate_key="bilibili:BVKEEP",
            teacher_score=0.80,
            y=1,
            candidate_profile_digest="oldp",
            candidate_negative_digest="oldn",
            profile_digest="oldp",
        ),
        _record(
            candidate_key="bilibili:BVSKIP",
            teacher_score=0.20,
            y=0,
            candidate_profile_digest="other",
            candidate_negative_digest="othern",
            profile_digest="other",
        ),
    ]
    overrides = {
        "bilibili:BVKEEP": {
            "new_llm_score_raw": 0.41,
            "new_y": 0,
            "new_profile_digest": "newp",
            "new_negative_digest": "newn",
        }
    }
    updated, applied = train_module.apply_relabel_overrides(records, overrides)
    assert applied == 1
    by_key = {row["candidate_key"]: row for row in updated}
    keep = by_key["bilibili:BVKEEP"]
    assert keep["teacher_score"] == 0.41
    assert keep["y"] == 0
    assert keep["candidate_profile_digest"] == "newp"
    assert keep["candidate_negative_digest"] == "newn"
    assert keep["profile_digest"] == "newp"
    skip = by_key["bilibili:BVSKIP"]
    assert skip["teacher_score"] == 0.20
    assert skip["y"] == 0
    assert skip["candidate_profile_digest"] == "other"
