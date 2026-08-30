"""Bounded, step-based policy for recommendation scoring weights.

The LLM may select which existing scoring component should matter more.  It
cannot provide a coefficient or replace the ranking formula.  Server-owned
steps and normalization keep every change small and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

RECOMMENDATION_WEIGHT_DIMENSIONS = (
    "relevance",
    "freshness",
    "topic_fatigue",
    "source_monotony",
    "serendipity",
)
RECOMMENDATION_WEIGHT_DIMENSION_SET = frozenset(RECOMMENDATION_WEIGHT_DIMENSIONS)

# Initial safety calibration (2026-08-30): one level changes the selected raw
# component by 5%, with three levels allowing at most 15% before normalization.
# This is deliberately smaller than the existing per-item feedback adjustments
# (0.05-0.20). Recalibrate from ranking/feedback A/B evidence after changing the
# evaluator model, embedding model, or scoring components.
WEIGHT_STEP_RATIO = 0.05
MAX_WEIGHT_LEVEL = 3


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the existing composite recommendation score.

    ``freshness`` weights the publication-time bonus, never cache insertion
    time. ``topic_fatigue`` and ``source_monotony`` weight penalties, while
    the other three dimensions weight positive components.
    """

    relevance: float = 0.30
    freshness: float = 0.10
    topic_fatigue: float = 0.25
    source_monotony: float = 0.15
    serendipity: float = 0.20

    def to_dict(self) -> dict[str, float]:
        """Return a deterministic dimension-to-weight mapping."""
        return {
            dimension: float(getattr(self, dimension))
            for dimension in RECOMMENDATION_WEIGHT_DIMENSIONS
        }


@dataclass(frozen=True)
class RecommendationWeightPolicy:
    """Persisted ladder levels for each scoring component."""

    relevance: int = 0
    freshness: int = 0
    topic_fatigue: int = 0
    source_monotony: int = 0
    serendipity: int = 0
    revision: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> RecommendationWeightPolicy:
        """Build a safe policy from a database-shaped mapping."""

        source = value or {}

        def _level(name: str) -> int:
            raw = source.get(f"{name}_level", source.get(name, 0))
            if isinstance(raw, bool):
                return 0
            if not isinstance(raw, int | float | str):
                return 0
            try:
                parsed = int(raw)
            except (TypeError, ValueError, OverflowError):
                return 0
            return max(0, min(MAX_WEIGHT_LEVEL, parsed))

        raw_revision = source.get("policy_revision", source.get("revision", 0))
        if not isinstance(raw_revision, int | float | str) or isinstance(raw_revision, bool):
            revision = 0
        else:
            try:
                revision = max(0, int(raw_revision))
            except (TypeError, ValueError, OverflowError):
                revision = 0
        return cls(
            relevance=_level("relevance"),
            freshness=_level("freshness"),
            topic_fatigue=_level("topic_fatigue"),
            source_monotony=_level("source_monotony"),
            serendipity=_level("serendipity"),
            revision=revision,
        )

    def level(self, dimension: str) -> int:
        """Return one validated dimension's current ladder level."""
        if dimension not in RECOMMENDATION_WEIGHT_DIMENSION_SET:
            raise ValueError(f"unsupported recommendation weight dimension: {dimension}")
        return int(getattr(self, dimension))

    def to_level_dict(self) -> dict[str, int]:
        """Return the persisted level shape without metadata."""
        return {
            dimension: int(getattr(self, dimension))
            for dimension in RECOMMENDATION_WEIGHT_DIMENSIONS
        }


def effective_scoring_weights(
    base: ScoringWeights,
    policy: RecommendationWeightPolicy,
) -> ScoringWeights:
    """Apply ladder multipliers and preserve the base formula's total scale."""

    base_values = base.to_dict()
    raw = {
        dimension: base_values[dimension] * (1.0 + WEIGHT_STEP_RATIO * policy.level(dimension))
        for dimension in RECOMMENDATION_WEIGHT_DIMENSIONS
    }
    base_total = sum(base_values.values())
    raw_total = sum(raw.values())
    if base_total <= 0.0 or raw_total <= 0.0:
        return base
    scale = base_total / raw_total
    return ScoringWeights(
        relevance=raw["relevance"] * scale,
        freshness=raw["freshness"] * scale,
        topic_fatigue=raw["topic_fatigue"] * scale,
        source_monotony=raw["source_monotony"] * scale,
        serendipity=raw["serendipity"] * scale,
    )
