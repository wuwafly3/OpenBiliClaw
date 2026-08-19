"""Unit tests for the teacher self-consistency probe helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ml_teacher_self_consistency_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "ml_teacher_self_consistency_probe.py",
)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_parse_teacher_identity_splits_provider_and_model() -> None:
    assert probe.parse_teacher_identity("openai/deepseek-v4-flash") == (
        "openai",
        "deepseek-v4-flash",
    )
    assert probe.parse_teacher_identity("openai_compatible/deepseek-v4-flash") == (
        "openai_compatible",
        "deepseek-v4-flash",
    )
    assert probe.parse_teacher_identity("") == ("", "")


def test_teacher_identity_matches_requires_exact_model_and_provider() -> None:
    assert probe.teacher_identity_matches(
        "openai/deepseek-v4-flash",
        provider="openai",
        model="deepseek-v4-flash",
    )
    assert not probe.teacher_identity_matches(
        "openai_compatible/deepseek-v4-flash",
        provider="openai",
        model="deepseek-v4-flash",
    )
    assert not probe.teacher_identity_matches(
        "openai/deepseek-v4-flash-thinking",
        provider="openai",
        model="deepseek-v4-flash",
    )
    assert not probe.teacher_identity_matches(
        "openai/deepseek-v4-flash",
        provider="openai",
        model="deepseek-v4-flash-thinking",
    )


def test_filter_rows_for_instance_drops_other_provider_and_thinking_variant() -> None:
    rows = [
        {"id": 1, "teacher_model": "openai/deepseek-v4-flash"},
        {"id": 2, "teacher_model": "openai_compatible/deepseek-v4-flash"},
        {"id": 3, "teacher_model": "openai/deepseek-v4-flash-thinking"},
        {"id": 4, "teacher_model": "openai/deepseek-v4-flash"},
    ]
    kept, counts = probe.filter_rows_for_instance(
        rows,
        provider_type="openai",
        model="deepseek-v4-flash",
    )
    assert [row["id"] for row in kept] == [1, 4]
    assert counts["allowlist"] == 4
    assert counts["kept"] == 2
    assert counts["dropped_provider_mismatch"] == 1
    assert counts["dropped_model_mismatch"] == 1


def test_expected_teacher_identity_matches_response_stamp() -> None:
    assert (
        probe.expected_teacher_identity(
            provider_type="openai",
            model="deepseek-v4-flash",
        )
        == "openai/deepseek-v4-flash"
    )


def test_pin_instance_overrides_uses_custom_chain_only() -> None:
    overrides = probe.pin_instance_overrides("openai-4")
    assert set(overrides) == {"evaluation", "discovery"}
    evaluation = overrides["evaluation"]
    assert evaluation.custom_chain is True
    assert evaluation.chain == ("openai-4",)


def test_replay_chunk_usable_rejects_provider_mix() -> None:
    ok, reason = probe.replay_chunk_usable(
        [SimpleNamespace(teacher_model="openai/deepseek-v4-flash")],
        expected_identity="openai/deepseek-v4-flash",
    )
    assert ok is True
    assert reason == "openai/deepseek-v4-flash"

    mixed, mix_reason = probe.replay_chunk_usable(
        [
            SimpleNamespace(teacher_model="openai/deepseek-v4-flash"),
            SimpleNamespace(teacher_model="openai_compatible/deepseek-v4-flash"),
        ],
        expected_identity="openai/deepseek-v4-flash",
    )
    assert mixed is False
    assert mix_reason.startswith("identity_mismatch:")


def test_pair_replay_skips_non_llm_sources_and_uses_raw_score() -> None:
    rows = [
        {
            "id": 10,
            "source_platform": "bilibili",
            "source_strategy": "search",
            "teacher_score": 0.72,
        },
        {
            "id": 11,
            "source_platform": "xiaohongshu",
            "source_strategy": "explore",
            "teacher_score": 0.50,
        },
    ]
    contents = [
        SimpleNamespace(
            score_source="llm",
            llm_score_raw=0.41,
            teacher_model="openai/deepseek-v4-flash",
        ),
        SimpleNamespace(
            score_source="viewed",
            llm_score_raw=None,
            teacher_model="",
        ),
    ]
    pairs, reason = probe.pair_replay(
        rows,
        contents,
        expected_identity="openai/deepseek-v4-flash",
    )
    assert reason == ""
    assert len(pairs) == 1
    assert pairs[0]["candidate_id"] == 10
    assert pairs[0]["orig_y"] == 1
    assert pairs[0]["replay_y"] == 0
    assert pairs[0]["replay_score"] == 0.41


def test_admission_agreement_stats_and_spearman() -> None:
    pairs = [
        {
            "orig_y": 1,
            "replay_y": 1,
            "orig_score": 0.9,
            "replay_score": 0.8,
            "platform": "bilibili",
            "near_threshold": False,
        },
        {
            "orig_y": 0,
            "replay_y": 0,
            "orig_score": 0.2,
            "replay_score": 0.1,
            "platform": "bilibili",
            "near_threshold": False,
        },
        {
            "orig_y": 0,
            "replay_y": 1,
            "orig_score": 0.58,
            "replay_score": 0.70,
            "platform": "xiaohongshu",
            "near_threshold": True,
        },
    ]
    stats = probe.admission_agreement_stats(pairs)
    assert stats["compared"] == 3
    assert stats["agree"] == 2
    assert stats["agreement"] == pytest.approx(2 / 3)
    assert stats["fp"] == 1
    assert stats["fn"] == 0
    assert stats["fpr"] == pytest.approx(0.5)
    assert stats["by_platform"]["xiaohongshu"]["n"] == 1
    assert stats["near_threshold"]["n"] == 1
    assert stats["near_threshold"]["agree"] == 0
    assert probe.spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert probe.spearman_rho([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


def test_group_rows_by_eval_context_keeps_mixed_digests_apart() -> None:
    rows = [
        {"id": 1, "profile_digest": "aaa", "negative_digest": "n1"},
        {"id": 2, "profile_digest": "bbb", "negative_digest": "n1"},
        {"id": 3, "profile_digest": "aaa", "negative_digest": "n1"},
        {"id": 4, "profile_digest": "", "negative_digest": ""},
    ]
    groups = probe.group_rows_by_eval_context(rows)
    assert [key for key, _rows in groups] == [("aaa", "n1"), ("bbb", "n1"), ("", "")]
    assert [row["id"] for row in groups[0][1]] == [1, 3]
    assert [row["id"] for row in groups[1][1]] == [2]
    assert [row["id"] for row in groups[2][1]] == [4]


def test_prepare_replay_groups_require_snapshot_skips_legacy_and_missing() -> None:
    from openbiliclaw.discovery.eval_context import (
        EvaluationContextSnapshot,
        compute_negative_digest,
        compute_profile_digest,
    )

    summary = {"interests": [{"name": "吉他", "weight": 0.9}]}
    snapshot = EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, []),
        negative_digest=compute_negative_digest([]),
        profile_summary=summary,
        recall_pool=[],
        negative_examples=[],
    )

    class _SnapshotDB:
        def get_evaluation_context_snapshot(self, *, profile_digest: str, negative_digest: str):
            if profile_digest == snapshot.profile_digest:
                return snapshot
            return None

    rows = [
        {
            "id": 1,
            "profile_digest": snapshot.profile_digest,
            "negative_digest": snapshot.negative_digest,
        },
        {"id": 2, "profile_digest": "", "negative_digest": ""},
        {"id": 3, "profile_digest": "missingdigest", "negative_digest": snapshot.negative_digest},
    ]
    replay, counts = probe.prepare_replay_groups(rows, _SnapshotDB(), require_snapshot=True)
    assert counts["groups"] == 3
    assert counts["snapshot_hits"] == 1
    assert counts["snapshot_misses"] == 2
    assert counts["skipped_no_snapshot"] == 2
    assert len(replay) == 1
    bound, group = replay[0]
    assert bound is snapshot
    assert [row["id"] for row in group] == [1]

    live_replay, live_counts = probe.prepare_replay_groups(
        rows, _SnapshotDB(), require_snapshot=False
    )
    assert live_counts["replay_rows"] == 3
    assert live_replay[1][0] is None
    assert [row["id"] for row in live_replay[1][1]] == [2]
