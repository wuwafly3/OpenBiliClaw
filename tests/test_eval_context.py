"""Unit tests for evaluation-context snapshot helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbiliclaw.discovery.eval_context import (
    GATE_RECENT_KEYS,
    EvaluationContextSnapshot,
    compute_negative_digest,
    compute_profile_digest,
    dumps_snapshot_json,
    loads_snapshot_json,
    parse_negative_examples,
    parse_recall_pool,
    profile_digest_payload,
    rewrite_snapshot_to_gate_contract,
)
from openbiliclaw.llm.prompt_cache import stable_json_digest
from openbiliclaw.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path


def test_profile_digest_matches_engine_payload_shape() -> None:
    summary = {"interests": [{"name": "吉他", "weight": 0.9}], "disliked_topics": ["推销"]}
    recall_pool = [("长尾兴趣", "音乐", 0.2)]
    digest = compute_profile_digest(summary, recall_pool)
    assert digest == stable_json_digest(profile_digest_payload(summary, recall_pool))
    as_lists = [["长尾兴趣", "音乐", 0.2]]
    assert compute_profile_digest(summary, as_lists) == digest


def test_snapshot_round_trip_json_keeps_digests() -> None:
    summary = {
        "core_traits": ["好奇"],
        "interests": [{"name": "强化学习", "weight": 0.8}],
        "disliked_topics": ["课程广告"],
        "recent_awareness": [{"observation": "最近在看系统课"}],
    }
    recall_pool = [("冷门框架", "工程", 0.1)]
    negatives = [{"title": "三天学会AI", "reason": "quick_exit", "age_days": 2}]
    snapshot = EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, recall_pool),
        negative_digest=compute_negative_digest(negatives),
        profile_summary=summary,
        recall_pool=recall_pool,
        negative_examples=negatives,
    )
    assert snapshot.digests_match()

    encoded = dumps_snapshot_json(
        {
            "profile_summary": snapshot.profile_summary,
            "recall_pool": snapshot.recall_pool,
            "negative_examples": snapshot.negative_examples,
        }
    )
    loaded = loads_snapshot_json(encoded)
    assert isinstance(loaded, dict)
    restored = EvaluationContextSnapshot(
        profile_digest=snapshot.profile_digest,
        negative_digest=snapshot.negative_digest,
        profile_summary=dict(loaded["profile_summary"]),
        recall_pool=parse_recall_pool(loaded["recall_pool"]),
        negative_examples=parse_negative_examples(loaded["negative_examples"]),
    )
    assert restored.digests_match()
    assert restored.profile_summary["interests"][0]["name"] == "强化学习"


def test_corrupt_snapshot_fails_digest_check() -> None:
    snapshot = EvaluationContextSnapshot(
        profile_digest="not-a-real-digest",
        negative_digest=compute_negative_digest([]),
        profile_summary={"interests": []},
        recall_pool=[],
        negative_examples=[],
    )
    assert snapshot.digests_match() is False


def _sample_snapshot() -> EvaluationContextSnapshot:
    summary = {"interests": [{"name": "吉他", "weight": 0.9}], "disliked_topics": ["推销"]}
    recall_pool = [("长尾兴趣", "音乐", 0.2)]
    negatives = [{"title": "三天学会AI", "reason": "quick_exit", "age_days": 2}]
    return EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, recall_pool),
        negative_digest=compute_negative_digest(negatives),
        profile_summary=summary,
        recall_pool=recall_pool,
        negative_examples=negatives,
    )


def test_evaluation_context_snapshot_round_trips_in_database(tmp_path: Path) -> None:
    database = Database(tmp_path / "eval-ctx.db")
    database.initialize()
    snapshot = _sample_snapshot()
    assert database.upsert_evaluation_context_snapshot(snapshot) is True
    loaded = database.get_evaluation_context_snapshot(
        profile_digest=snapshot.profile_digest,
        negative_digest=snapshot.negative_digest,
    )
    assert loaded is not None
    assert loaded.digests_match()
    assert loaded.profile_summary["interests"][0]["name"] == "吉他"
    assert loaded.negative_examples[0]["title"] == "三天学会AI"


def test_evaluation_context_snapshot_mismatch_is_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "eval-ctx-mismatch.db")
    database.initialize()
    snapshot = _sample_snapshot()
    assert database.upsert_evaluation_context_snapshot(snapshot) is True
    database.conn.execute(
        """
        UPDATE evaluation_context_snapshots
        SET profile_summary_json = ?
        WHERE profile_digest = ? AND negative_digest = ?
        """,
        (
            '{"interests":[{"name":"被篡改","weight":0.1}]}',
            snapshot.profile_digest,
            snapshot.negative_digest,
        ),
    )
    database.conn.commit()
    assert (
        database.get_evaluation_context_snapshot(
            profile_digest=snapshot.profile_digest,
            negative_digest=snapshot.negative_digest,
        )
        is None
    )
    corrupt = EvaluationContextSnapshot(
        profile_digest="not-a-real-digest",
        negative_digest=snapshot.negative_digest,
        profile_summary=snapshot.profile_summary,
        recall_pool=snapshot.recall_pool,
        negative_examples=snapshot.negative_examples,
    )
    assert database.upsert_evaluation_context_snapshot(corrupt) is False


def test_rewrite_snapshot_drops_recent_and_shell_titles() -> None:
    """Old snapshots become the live gate slice; ChatGLM stays, cheers titles go."""

    summary = {
        "interests": [{"name": "吉他", "weight": 0.9}],
        "disliked_topics": ["推销"],
        "recent_awareness": [{"observation": "最近在看系统课"}],
        "active_insights": [{"claim": "想把旧镜头用起来"}],
        "speculative_interests": [{"domain": "机械钟"}],
    }
    recall_pool = [("长尾兴趣", "音乐", 0.2)]
    negatives = [
        {
            "title": "哔哩哔哩 (゜-゜)つロ 干杯~-bilibili",
            "reason": "explicit_negative",
            "age_days": 0,
        },
        {"title": "ChatGLM", "reason": "explicit_negative", "age_days": 1},
        {"title": "保姆级教程一期通关", "reason": "quick_exit", "age_days": 2},
    ]
    original = EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, recall_pool),
        negative_digest=compute_negative_digest(negatives),
        profile_summary=summary,
        recall_pool=recall_pool,
        negative_examples=negatives,
    )
    first = rewrite_snapshot_to_gate_contract(original)
    second = rewrite_snapshot_to_gate_contract(original)
    rewritten = first.snapshot
    titles = [str(item.get("title") or "") for item in rewritten.negative_examples]

    assert first.had_recent is True
    assert first.dropped_shell == 1
    assert first.digest_changed is True
    assert rewritten.digests_match()
    assert all(key not in rewritten.profile_summary for key in GATE_RECENT_KEYS)
    assert "哔哩哔哩 (゜-゜)つロ 干杯~-bilibili" not in titles
    assert "ChatGLM" in titles
    assert "保姆级教程一期通关" in titles
    assert original.profile_summary["recent_awareness"][0]["observation"] == "最近在看系统课"
    assert first.snapshot.profile_digest == second.snapshot.profile_digest
    assert first.snapshot.negative_digest == second.snapshot.negative_digest
    assert first.snapshot.negative_examples == second.snapshot.negative_examples


def test_rewrite_gate_snapshot_is_idempotent() -> None:
    summary = {
        "interests": [{"name": "吉他", "weight": 0.9}],
        "disliked_topics": ["推销"],
        "recent_awareness": [{"observation": "最近在看系统课"}],
    }
    recall_pool = [("长尾兴趣", "音乐", 0.2)]
    negatives = [{"title": "ChatGLM", "reason": "explicit_negative", "age_days": 1}]
    original = EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, recall_pool),
        negative_digest=compute_negative_digest(negatives),
        profile_summary=summary,
        recall_pool=recall_pool,
        negative_examples=negatives,
    )
    once = rewrite_snapshot_to_gate_contract(original)
    twice = rewrite_snapshot_to_gate_contract(once.snapshot)
    assert once.had_recent is True
    assert twice.had_recent is False
    assert twice.dropped_shell == 0
    assert twice.digest_changed is False
    assert twice.snapshot.profile_digest == once.snapshot.profile_digest
    assert twice.snapshot.negative_digest == once.snapshot.negative_digest
