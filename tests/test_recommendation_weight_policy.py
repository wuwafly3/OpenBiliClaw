"""Tests for bounded LLM-driven recommendation weight adaptation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from openbiliclaw.recommendation.curator import PoolCurator
from openbiliclaw.recommendation.tools import (
    RAISE_RECOMMENDATION_WEIGHT_TOOL,
    RecommendationToolDispatcher,
)
from openbiliclaw.recommendation.weight_policy import (
    MAX_WEIGHT_LEVEL,
    RecommendationWeightPolicy,
    ScoringWeights,
    effective_scoring_weights,
)
from openbiliclaw.soul.dialogue import CompositeToolDispatcher
from openbiliclaw.storage.database import Database


def _make_db(tmp_path: object) -> Database:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


def _tool_call(
    *,
    request_id: str,
    dimension: str = "freshness",
    user_message: str = "最近推荐的内容整体太旧了",
) -> dict[str, object]:
    return {
        "name": RAISE_RECOMMENDATION_WEIGHT_TOOL,
        "arguments": {
            "dimension": dimension,
            "reason": "用户明确希望整体推荐更新",
            # An arbitrary coefficient is ignored because it is not part of
            # the server-owned tool contract.
            "weight": 99,
        },
        "_request_id": request_id,
        "_user_message": user_message,
    }


def test_effective_weights_raise_one_dimension_and_preserve_total() -> None:
    base = ScoringWeights()
    effective = effective_scoring_weights(
        base,
        RecommendationWeightPolicy(freshness=1),
    )

    assert effective.freshness > base.freshness
    assert sum(effective.to_dict().values()) == pytest.approx(sum(base.to_dict().values()))
    assert effective.relevance < base.relevance


def test_equal_levels_leave_relative_formula_unchanged() -> None:
    base = ScoringWeights()
    effective = effective_scoring_weights(
        base,
        RecommendationWeightPolicy(
            relevance=2,
            freshness=2,
            topic_fatigue=2,
            source_monotony=2,
            serendipity=2,
        ),
    )

    assert effective.to_dict() == pytest.approx(base.to_dict())


def test_database_step_is_idempotent_and_conflict_safe(tmp_path: object) -> None:
    database = _make_db(tmp_path)

    first = database.raise_recommendation_weight_level(
        dimension="freshness",
        request_id="turn-1",
        feedback_excerpt="最近推荐的内容整体太旧了",
        reason="整体时效反馈",
    )
    duplicate = database.raise_recommendation_weight_level(
        dimension="freshness",
        request_id="turn-1",
        feedback_excerpt="重复请求",
        reason="重试",
    )
    conflict = database.raise_recommendation_weight_level(
        dimension="relevance",
        request_id="turn-1",
        feedback_excerpt="冲突请求",
        reason="不应生效",
    )

    assert first["status"] == "applied"
    assert first["new_level"] == 1
    assert duplicate["status"] == "duplicate"
    assert conflict["status"] == "conflict"
    policy = database.get_recommendation_weight_policy()
    assert policy["freshness_level"] == 1
    assert policy["relevance_level"] == 0
    assert policy["policy_revision"] == 1
    receipts = database.list_recommendation_weight_adjustments()
    assert len(receipts) == 1
    assert receipts[0]["feedback_excerpt"] == "最近推荐的内容整体太旧了"


def test_tool_advances_one_step_and_stops_at_server_cap(tmp_path: object) -> None:
    database = _make_db(tmp_path)
    dispatcher = RecommendationToolDispatcher(database)

    messages = [
        dispatcher.dispatch(_tool_call(request_id=f"turn-{index}"))
        for index in range(1, MAX_WEIGHT_LEVEL + 2)
    ]

    assert "第 1/3 阶" in messages[0]
    assert "最高第 3 阶" in messages[-1]
    policy = database.get_recommendation_weight_policy()
    assert policy["freshness_level"] == MAX_WEIGHT_LEVEL
    assert policy["policy_revision"] == MAX_WEIGHT_LEVEL
    assert len(database.list_recommendation_weight_adjustments()) == MAX_WEIGHT_LEVEL + 1


def test_concurrent_feedback_steps_are_atomic(tmp_path: object) -> None:
    database = _make_db(tmp_path)

    def _raise(index: int) -> dict[str, object]:
        return database.raise_recommendation_weight_level(
            dimension="relevance",
            request_id=f"parallel-turn-{index}",
            feedback_excerpt="整体推荐不够贴合我的兴趣",
            reason="用户希望提高整体兴趣相关性",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_raise, range(4)))

    assert sorted(str(result["status"]) for result in results) == [
        "applied",
        "applied",
        "applied",
        "at_limit",
    ]
    policy = database.get_recommendation_weight_policy()
    assert policy["relevance_level"] == MAX_WEIGHT_LEVEL
    assert policy["policy_revision"] == MAX_WEIGHT_LEVEL


def test_tool_requires_server_bound_feedback(tmp_path: object) -> None:
    database = _make_db(tmp_path)
    dispatcher = RecommendationToolDispatcher(database)

    result = dispatcher.dispatch(
        {
            "name": RAISE_RECOMMENDATION_WEIGHT_TOOL,
            "arguments": {"dimension": "freshness", "reason": "内容太旧"},
        }
    )

    assert "缺少可核对的用户反馈" in result
    assert database.get_recommendation_weight_policy()["policy_revision"] == 0


def test_committed_tool_result_survives_diagnostic_read_failure() -> None:
    class CommittedDatabase:
        def raise_recommendation_weight_level(self, **_kwargs: object) -> dict[str, object]:
            return {
                "status": "applied",
                "dimension": "freshness",
                "previous_level": 0,
                "new_level": 1,
            }

        def get_recommendation_weight_policy(self) -> dict[str, object]:
            raise RuntimeError("diagnostic read unavailable")

    result = RecommendationToolDispatcher(CommittedDatabase()).dispatch(
        _tool_call(request_id="turn-committed")
    )

    assert "已把「内容新鲜度」提高到第 1/3 阶" in result
    assert "明细暂时无法读取" in result


def test_curator_context_reads_persisted_policy(tmp_path: object) -> None:
    database = _make_db(tmp_path)
    database.raise_recommendation_weight_level(
        dimension="serendipity",
        request_id="turn-explore",
        feedback_excerpt="推荐整体太保守，我想多看看圈外内容",
        reason="用户希望增加探索内容",
    )

    context = PoolCurator(database).build_context_from_rows([], [])

    assert context.weights is not None
    assert context.weights.serendipity > ScoringWeights().serendipity
    assert sum(context.weights.to_dict().values()) == pytest.approx(1.0)


def test_composite_dispatcher_routes_only_registered_owner(tmp_path: object) -> None:
    database = _make_db(tmp_path)
    recommendation = RecommendationToolDispatcher(database)
    dispatcher = CompositeToolDispatcher({RAISE_RECOMMENDATION_WEIGHT_TOOL: recommendation})

    applied = dispatcher.dispatch(_tool_call(request_id="turn-route"))
    unknown = dispatcher.dispatch({"name": "replace_formula", "arguments": {}})

    assert "第 1/3 阶" in applied
    assert unknown == "未知工具: replace_formula"
