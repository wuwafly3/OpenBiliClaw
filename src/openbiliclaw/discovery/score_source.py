"""Score provenance taxonomy for evaluated discovery candidates.

``discovery_candidates.relevance_score`` mixes genuine LLM judgments with
scores that deterministic post-processing zeroed or synthesized (intra-batch
diversity caps, embedding prefilter, recently-viewed skips, failure
fallbacks). The ml-ranking distillation pipeline (Wave 0) must train only on
scores the teacher actually produced, so every write path stamps
``score_source`` and preserves the pre-cap judgment in ``llm_score_raw``.

This module is a dependency leaf: ``discovery.engine`` and
``storage.database`` both import it, so it must not import either package.
"""

from typing import Final

# The teacher (LLM) produced this score — distillation may train on it.
SCORE_SOURCE_LLM: Final = "llm"
# The teacher scored the item, but an intra-batch franchise/style cap then
# zeroed ``relevance_score``; the original judgment survives in
# ``llm_score_raw``.
SCORE_SOURCE_CAP_FRANCHISE: Final = "cap_franchise"
SCORE_SOURCE_CAP_STYLE: Final = "cap_style"
# Embedding prefilter pseudo-score (``max_similarity * 0.5``) — never saw
# the LLM.
SCORE_SOURCE_PREFILTER: Final = "prefilter"
# Zeroed before evaluation because the user recently viewed the content.
SCORE_SOURCE_VIEWED: Final = "viewed"
# Single-item evaluation raised; the 0.0 fallback is not a judgment.
SCORE_SOURCE_EVAL_ERROR: Final = "eval_error"
# The batch response omitted this member; the item is re-queued, not judged.
SCORE_SOURCE_RESPONSE_MISSING: Final = "response_missing"
# Beyond ``_EVALUATE_BATCH_HARD_CAP``; never sent to the LLM.
SCORE_SOURCE_TRUNCATED: Final = "truncated"

# Sources whose ``llm_score_raw`` is a genuine teacher judgment and may enter
# the distillation training set. The empty string (rows written before this
# taxonomy shipped) is deliberately absent: legacy origin is unknown, never
# guessed.
LLM_JUDGMENT_SCORE_SOURCES: Final[frozenset[str]] = frozenset(
    {SCORE_SOURCE_LLM, SCORE_SOURCE_CAP_FRANCHISE, SCORE_SOURCE_CAP_STYLE}
)
