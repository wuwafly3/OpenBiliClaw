"""LLM tool for bounded recommendation-weight adaptation."""

from __future__ import annotations

import logging
from typing import Any

from openbiliclaw.recommendation.weight_policy import (
    MAX_WEIGHT_LEVEL,
    RECOMMENDATION_WEIGHT_DIMENSION_SET,
    RecommendationWeightPolicy,
    ScoringWeights,
    effective_scoring_weights,
)

logger = logging.getLogger(__name__)

RAISE_RECOMMENDATION_WEIGHT_TOOL = "raise_recommendation_weight"
RECOMMENDATION_TOOL_NAMES = frozenset({RAISE_RECOMMENDATION_WEIGHT_TOOL})

RECOMMENDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": RAISE_RECOMMENDATION_WEIGHT_TOOL,
        "description": (
            "当用户明确评价整体推荐排序时，把一个既有打分维度提高一阶。"
            "只在用户表达‘整体不够相关、内容普遍太旧、同主题反复出现、"
            "来源过于单一、推荐太保守’等反馈时调用；不要因单张卡片的 like/dislike 调用。"
            "每次只能升一阶，具体幅度和上限由后端决定，不能修改公式或传入数值权重。"
        ),
        "parameters": {
            "dimension": (
                "只能是 relevance（整体更贴合兴趣）/ freshness（更新）/ "
                "topic_fatigue（减少同主题重复）/ source_monotony（增加来源多样性）/ "
                "serendipity（增加探索和惊喜）"
            ),
            "reason": "用一句话说明这项维度如何对应用户刚刚给出的整体推荐反馈",
        },
    }
]

_DIMENSION_LABELS = {
    "relevance": "兴趣相关性",
    "freshness": "内容新鲜度",
    "topic_fatigue": "主题去重复",
    "source_monotony": "来源多样性",
    "serendipity": "探索与惊喜",
}


class RecommendationToolDispatcher:
    """Validate and execute recommendation policy tool calls."""

    def __init__(self, database: Any) -> None:
        self._db = database

    def dispatch(self, tool_call: dict[str, Any]) -> str:
        """Raise exactly one server-controlled ladder step."""
        name = str(tool_call.get("name", ""))
        if name != RAISE_RECOMMENDATION_WEIGHT_TOOL:
            return f"未知工具: {name}"

        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            return "调权参数必须是对象。"
        dimension = str(args.get("dimension", "")).strip()
        if dimension not in RECOMMENDATION_WEIGHT_DIMENSION_SET:
            return "未知打分维度，未修改推荐权重。"
        reason = " ".join(str(args.get("reason", "")).split())[:240]
        if not reason:
            return "缺少调权理由，未修改推荐权重。"

        request_id = str(tool_call.get("_request_id", "")).strip()
        user_feedback = " ".join(str(tool_call.get("_user_message", "")).split())[:400]
        if not request_id or not user_feedback:
            return "缺少可核对的用户反馈，未修改推荐权重。"

        try:
            result = self._db.raise_recommendation_weight_level(
                dimension=dimension,
                request_id=request_id,
                feedback_excerpt=user_feedback,
                reason=reason,
            )
        except ValueError as exc:
            logger.warning("Rejected recommendation weight tool call: %s", exc)
            return "调权请求不符合安全限制，未修改推荐权重。"
        except Exception:
            logger.exception("Recommendation weight tool failed")
            return "推荐权重暂时无法更新，请稍后再试。"

        status = str(result.get("status", ""))
        if status == "conflict":
            return "这条反馈已经用于另一项调权，未重复修改。"

        try:
            policy = RecommendationWeightPolicy.from_mapping(
                self._db.get_recommendation_weight_policy()
            )
            weights = effective_scoring_weights(ScoringWeights(), policy).to_dict()
            weight_suffix = "当前权重：" + "、".join(
                f"{_DIMENSION_LABELS[key]} {weights[key]:.3f}" for key in _DIMENSION_LABELS
            )
        except Exception:
            # The mutation already committed. A diagnostic read must not turn
            # that success into a failed turn and invite a confusing retry.
            logger.warning("Recommendation weight result read failed", exc_info=True)
            policy = RecommendationWeightPolicy()
            weight_suffix = "当前权重明细暂时无法读取"
        label = _DIMENSION_LABELS[dimension]
        level = int(result.get("new_level", policy.level(dimension)))
        if status == "duplicate":
            return f"这条反馈已处理过，未重复调权。{weight_suffix}。"
        if status == "at_limit":
            return f"「{label}」已在最高第 {MAX_WEIGHT_LEVEL} 阶，未继续提高。{weight_suffix}。"
        return f"已把「{label}」提高到第 {level}/{MAX_WEIGHT_LEVEL} 阶。{weight_suffix}。"
