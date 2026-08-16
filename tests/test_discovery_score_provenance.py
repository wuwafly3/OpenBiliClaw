"""Score provenance marking for discovery evaluation (ml-ranking Wave 0).

``discovery_candidates.relevance_score`` mixes teacher judgments with
deterministically zeroed/synthesized scores. These tests pin the contract the
distillation dataset depends on: every non-LLM score path stamps a distinct
``score_source``, cap zeroing preserves the teacher judgment in
``llm_score_raw``, and the storage layer round-trips both columns with a
teacher-only read allowlist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from openbiliclaw.discovery import engine as discovery_engine_module
from openbiliclaw.discovery.candidate_pool import DiscoveryCandidateWrite
from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent
from openbiliclaw.discovery.score_source import (
    LLM_JUDGMENT_SCORE_SOURCES,
    SCORE_SOURCE_CAP_FRANCHISE,
    SCORE_SOURCE_CAP_STYLE,
    SCORE_SOURCE_LLM,
    SCORE_SOURCE_PREFILTER,
    SCORE_SOURCE_RESPONSE_MISSING,
    SCORE_SOURCE_TRUNCATED,
    SCORE_SOURCE_VIEWED,
)
from openbiliclaw.storage.database import Database

from .test_discovery_engine import (
    _batch_prompt_items,
    _batch_prompt_uses_local_ids,
    _CountingEmbeddingService,
    _DynamicBatchLLMService,
    _prefilter_vectors,
    _RecentViewedDatabase,
    _SlowResponse,
    _split_retry_contents,
    _SplitRetryBatchLLMService,
)
from .test_search_strategy import FakeLLMService, _build_profile

if TYPE_CHECKING:
    from pathlib import Path


class _VariedStyleBatchLLMService(_DynamicBatchLLMService):
    """Same as the parent, but every item gets a distinct style key.

    Keeps the truncation test below out of the intra-batch style cap so
    provenance assertions see the plain LLM path.
    """

    async def complete_structured_task(
        self,
        *,
        system_instruction: str,
        user_input: str,
        **kwargs: object,
    ) -> object:
        self.user_inputs.append(user_input)
        self.max_tokens.append(int(kwargs.get("max_tokens") or 4096))  # type: ignore[arg-type]
        items = _batch_prompt_items(user_input)
        local_ids = _batch_prompt_uses_local_ids(user_input)
        identity_field = "id" if local_ids else "content_id"
        payload: list[dict[str, object]] = []
        for index, item in enumerate(items):
            identity = (
                item.get("id")
                if local_ids
                else item.get("content_id") or item.get("bvid") or str(index)
            )
            payload.append(
                {
                    identity_field: identity,
                    "score": 0.8,
                    "reason": "ok",
                    "style_key": f"style_{index % 20}",
                }
            )
        return _SlowResponse(json.dumps(payload, ensure_ascii=False))


def _llm_judged(franchise: str = "", score: float = 0.8) -> DiscoveredContent:
    content = DiscoveredContent(bvid="", title="", franchise_key=franchise)
    content.relevance_score = score
    content.score_source = SCORE_SOURCE_LLM
    content.llm_score_raw = score
    return content


def test_franchise_cap_marks_provenance_and_preserves_teacher_score() -> None:
    cap = discovery_engine_module._BATCH_FRANCHISE_CAP
    contents = [_llm_judged(franchise="原神", score=0.9 - index * 0.01) for index in range(cap + 2)]
    results = [content.relevance_score for content in contents]

    ContentDiscoveryEngine._apply_intra_batch_caps(contents, results)

    for kept in contents[:cap]:
        assert kept.score_source == SCORE_SOURCE_LLM
        assert kept.relevance_score > 0.0
        assert kept.llm_score_raw == kept.relevance_score
    for index, dropped in enumerate(contents[cap:]):
        assert dropped.relevance_score == 0.0
        assert results[cap + index] == 0.0
        assert dropped.score_source == SCORE_SOURCE_CAP_FRANCHISE
        assert dropped.llm_score_raw == pytest.approx(0.9 - (cap + index) * 0.01)


def test_style_cap_marks_provenance_and_preserves_teacher_score() -> None:
    style_cap = discovery_engine_module._BATCH_STYLE_CAP
    contents = []
    for index in range(style_cap + 1):
        content = _llm_judged(score=0.85 - index * 0.01)
        content.style_key = "deep_dive"
        contents.append(content)
    results = [content.relevance_score for content in contents]

    ContentDiscoveryEngine._apply_intra_batch_caps(contents, results)

    assert contents[-1].relevance_score == 0.0
    assert contents[-1].score_source == SCORE_SOURCE_CAP_STYLE
    assert contents[-1].llm_score_raw == pytest.approx(0.85 - style_cap * 0.01)
    for kept in contents[:style_cap]:
        assert kept.score_source == SCORE_SOURCE_LLM
        assert kept.relevance_score > 0.0


def test_cap_does_not_fabricate_teacher_score_for_non_llm_items() -> None:
    content = _llm_judged(score=0.9)
    content.score_source = SCORE_SOURCE_PREFILTER
    content.llm_score_raw = None
    content.style_key = "deep_dive"
    others = [_llm_judged(score=0.95) for _ in range(discovery_engine_module._BATCH_STYLE_CAP + 1)]
    for other in others:
        other.style_key = "deep_dive"
    batch = others + [content]
    results = [item.relevance_score for item in batch]

    ContentDiscoveryEngine._apply_intra_batch_caps(batch, results)

    assert content.score_source == SCORE_SOURCE_CAP_STYLE
    assert content.llm_score_raw is None


@pytest.mark.asyncio
async def test_batch_llm_scores_are_marked_as_teacher_judgments() -> None:
    engine = ContentDiscoveryEngine(llm_service=_DynamicBatchLLMService())
    contents = [
        DiscoveredContent(
            bvid=f"BV_PROV_LLM_{index}", title=f"候选 {index}", source_strategy="search"
        )
        for index in range(3)
    ]

    scores = await engine.evaluate_content_batch(contents, _build_profile())

    assert scores == [0.8, 0.8, 0.8]
    for content in contents:
        assert content.score_source == SCORE_SOURCE_LLM
        assert content.llm_score_raw == 0.8


@pytest.mark.asyncio
async def test_recently_viewed_candidates_are_marked_not_teacher_judged() -> None:
    llm_service = FakeLLMService(
        json.dumps(
            [{"bvid": "BV_PROV_FRESH", "score": 0.88, "reason": "fresh match"}],
            ensure_ascii=False,
        )
    )
    engine = ContentDiscoveryEngine(
        llm_service=llm_service,
        database=_RecentViewedDatabase({"BV_PROV_VIEWED"}),  # type: ignore[arg-type]
    )
    viewed = DiscoveredContent(bvid="BV_PROV_VIEWED", title="已经看过", source_strategy="trending")
    fresh = DiscoveredContent(bvid="BV_PROV_FRESH", title="新内容", source_strategy="trending")

    scores = await engine.evaluate_content_batch([viewed, fresh], _build_profile())

    assert scores == [0.0, 0.88]
    assert viewed.score_source == SCORE_SOURCE_VIEWED
    assert viewed.llm_score_raw is None
    assert fresh.score_source == SCORE_SOURCE_LLM
    assert fresh.llm_score_raw == 0.88


@pytest.mark.asyncio
async def test_missing_batch_members_are_marked_not_teacher_judged() -> None:
    engine = ContentDiscoveryEngine(
        llm_service=_SplitRetryBatchLLMService(invalid_all_batches=True),
        evaluation_candidate_transport="production",
    )
    contents = _split_retry_contents(16, prefix="PROVMISS")

    scores = await engine.evaluate_content_batch(contents, _build_profile(), batch_size=16)

    assert scores == [0.0] * 16
    for content in contents:
        assert content.score_source == SCORE_SOURCE_RESPONSE_MISSING
        assert content.llm_score_raw is None


@pytest.mark.asyncio
async def test_batch_truncation_overflow_is_marked_not_teacher_judged() -> None:
    engine = ContentDiscoveryEngine(llm_service=_VariedStyleBatchLLMService())
    hard_cap = engine._EVALUATE_BATCH_HARD_CAP
    contents = [
        DiscoveredContent(
            bvid=f"BV_PROV_TRUNC_{index}", title=f"候选 {index}", source_strategy="search"
        )
        for index in range(hard_cap + 2)
    ]

    await engine.evaluate_content_batch(contents, _build_profile(), batch_size=90)

    for overflow in contents[hard_cap:]:
        assert overflow.score_source == SCORE_SOURCE_TRUNCATED
        assert overflow.llm_score_raw is None
    for kept in contents[:hard_cap]:
        assert kept.score_source == SCORE_SOURCE_LLM


@pytest.mark.asyncio
async def test_prefilter_enforce_marks_pseudo_scores_not_teacher_judged(
    tmp_path: Path,
) -> None:
    low_text = "不相关内容 厨房技巧"
    database = Database(tmp_path / "provenance-prefilter.db")
    database.initialize()
    engine = ContentDiscoveryEngine(
        llm_service=_DynamicBatchLLMService(),
        database=database,
        embedding_service=_CountingEmbeddingService(_prefilter_vectors(low_texts=[low_text])),
        eval_prefilter_mode="enforce",
    )
    filtered = DiscoveredContent(
        bvid="BV_PROV_FILTER",
        title="不相关内容",
        description="厨房技巧",
        source_strategy="trending",
    )
    relevant = DiscoveredContent(
        bvid="BV_PROV_KEEP",
        title="匹配内容",
        description="深度纪录片解析",
        source_strategy="trending",
    )

    scores = await engine.evaluate_content_batch(
        [filtered, relevant], _build_profile(), batch_size=2
    )

    assert scores[0] == pytest.approx(0.05)
    assert filtered.score_source == SCORE_SOURCE_PREFILTER
    assert filtered.llm_score_raw is None
    assert relevant.score_source == SCORE_SOURCE_LLM
    assert relevant.llm_score_raw == 0.8

    # The pseudo-score is cached like an eval result; a later cache hit must
    # keep the non-teacher provenance instead of re-labeling it as LLM.
    profile_digest = engine._evaluation_profile_digest(_build_profile())
    negative_digest = engine._negative_examples_digest(None)
    cache_key = engine._batch_eval_cache_key(
        filtered,
        profile_digest=profile_digest,
        negative_digest=negative_digest,
    )
    cached = engine._get_eval_cache_entry(cache_key)
    assert cached is not None
    decoded = discovery_engine_module._decode_eval_cache_entry(cached)
    assert decoded[6] == SCORE_SOURCE_PREFILTER


def _enqueue_and_claim(database: Database, candidate_key: str) -> int:
    inserted = database.enqueue_discovery_candidates(
        [
            DiscoveryCandidateWrite(
                candidate_key=candidate_key,
                source_platform="bilibili",
                source_strategy="search",
                content_id=candidate_key,
                title=f"候选 {candidate_key}",
            )
        ]
    )
    assert inserted == 1
    claimed = database.claim_discovery_candidates_for_eval(limit=10)
    assert len(claimed) == 1
    return int(claimed[0]["id"])


def _evaluation(
    candidate_id: int,
    *,
    score_source: str,
    llm_score_raw: float | None,
    relevance_score: float = 0.0,
    teacher_model: str = "",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "status": "evaluated",
        "relevance_score": relevance_score,
        "relevance_reason": "",
        "score_source": score_source,
        "llm_score_raw": llm_score_raw,
        "teacher_model": teacher_model,
        "temporal_class": "unknown",
        "temporal_confidence": 0.0,
        "temporal_policy_version": "v1",
    }


def test_evaluation_persist_round_trips_score_source_and_raw(tmp_path: Path) -> None:
    database = Database(tmp_path / "provenance-roundtrip.db")
    database.initialize()
    capped_id = _enqueue_and_claim(database, "BV_PROV_CAPPED")
    legacy_id = _enqueue_and_claim(database, "BV_PROV_LEGACY")
    prefilter_id = _enqueue_and_claim(database, "BV_PROV_PREFILTER")

    updated = database.update_discovery_candidate_evaluations(
        [
            _evaluation(
                capped_id,
                score_source=SCORE_SOURCE_CAP_FRANCHISE,
                llm_score_raw=0.83,
            ),
            _evaluation(legacy_id, score_source="", llm_score_raw=None),
            _evaluation(
                prefilter_id,
                score_source=SCORE_SOURCE_PREFILTER,
                llm_score_raw=None,
                relevance_score=0.1,
            ),
        ]
    )
    assert updated == 3

    rows = {
        int(row["id"]): row
        for row in database.conn.execute(
            "SELECT id, score_source, llm_score_raw FROM discovery_candidates"
        ).fetchall()
    }
    assert rows[capped_id]["score_source"] == SCORE_SOURCE_CAP_FRANCHISE
    assert rows[capped_id]["llm_score_raw"] == pytest.approx(0.83)
    assert rows[legacy_id]["score_source"] == ""
    assert rows[legacy_id]["llm_score_raw"] is None
    assert rows[prefilter_id]["score_source"] == SCORE_SOURCE_PREFILTER

    teacher_rows = database.get_teacher_labeled_discovery_candidates()
    assert [int(row["id"]) for row in teacher_rows] == [capped_id]
    assert teacher_rows[0]["llm_score_raw"] == pytest.approx(0.83)
    assert teacher_rows[0]["score_source"] == SCORE_SOURCE_CAP_FRANCHISE


def test_legacy_database_gains_provenance_columns_tolerantly(tmp_path: Path) -> None:
    database = Database(tmp_path / "provenance-migration.db")
    database.initialize()
    for column in ("score_source", "llm_score_raw"):
        database.conn.execute(f"ALTER TABLE discovery_candidates DROP COLUMN {column}")
    remaining = {
        str(row["name"])
        for row in database.conn.execute("PRAGMA table_info(discovery_candidates)").fetchall()
    }
    assert "score_source" not in remaining
    assert "llm_score_raw" not in remaining

    database._ensure_discovery_candidate_columns()

    restored = {
        str(row["name"])
        for row in database.conn.execute("PRAGMA table_info(discovery_candidates)").fetchall()
    }
    assert "score_source" in restored
    assert "llm_score_raw" in restored


def test_llm_judgment_allowlist_covers_caps_but_not_synthetic_sources() -> None:
    assert SCORE_SOURCE_LLM in LLM_JUDGMENT_SCORE_SOURCES
    assert SCORE_SOURCE_CAP_FRANCHISE in LLM_JUDGMENT_SCORE_SOURCES
    assert SCORE_SOURCE_CAP_STYLE in LLM_JUDGMENT_SCORE_SOURCES
    for synthetic in (
        SCORE_SOURCE_PREFILTER,
        SCORE_SOURCE_VIEWED,
        SCORE_SOURCE_RESPONSE_MISSING,
        SCORE_SOURCE_TRUNCATED,
        "",
    ):
        assert synthetic not in LLM_JUDGMENT_SCORE_SOURCES


@dataclass
class _StampedResponse:
    content: str
    model: str = ""
    provider: str = ""


class _ModelStampedBatchLLMService(_VariedStyleBatchLLMService):
    """Echoes the per-call teacher identity like a real LLMResponse does."""

    async def complete_structured_task(
        self,
        *,
        system_instruction: str,
        user_input: str,
        **kwargs: object,
    ) -> object:
        plain = await super().complete_structured_task(
            system_instruction=system_instruction, user_input=user_input, **kwargs
        )
        assert isinstance(plain, _SlowResponse)
        return _StampedResponse(
            content=plain.content,
            model="deepseek-v4-flash",
            provider="deepseek",
        )


@pytest.mark.asyncio
async def test_batch_llm_scores_stamp_teacher_model_identity() -> None:
    engine = ContentDiscoveryEngine(llm_service=_ModelStampedBatchLLMService())
    contents = [
        DiscoveredContent(
            bvid=f"BV_PROV_MODEL_{index}", title=f"候选 {index}", source_strategy="search"
        )
        for index in range(3)
    ]

    await engine.evaluate_content_batch(contents, _build_profile())

    for content in contents:
        assert content.score_source == "llm"
        assert content.teacher_model == "deepseek/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_response_without_identity_stamps_empty_teacher_model() -> None:
    engine = ContentDiscoveryEngine(llm_service=_VariedStyleBatchLLMService())
    contents = [DiscoveredContent(bvid="BV_PROV_NOMODEL", title="候选", source_strategy="search")]

    await engine.evaluate_content_batch(contents, _build_profile())

    assert contents[0].score_source == "llm"
    assert contents[0].teacher_model == ""


@pytest.mark.asyncio
async def test_non_teacher_paths_leave_teacher_model_empty(tmp_path: Path) -> None:
    low_text = "不相关内容 厨房技巧"
    database = Database(tmp_path / "teacher-model-prefilter.db")
    database.initialize()
    engine = ContentDiscoveryEngine(
        llm_service=_ModelStampedBatchLLMService(),
        database=database,
        embedding_service=_CountingEmbeddingService(_prefilter_vectors(low_texts=[low_text])),
        eval_prefilter_mode="enforce",
    )
    filtered = DiscoveredContent(
        bvid="BV_PROV_MODEL_FILTER",
        title="不相关内容",
        description="厨房技巧",
        source_strategy="trending",
    )
    viewed_db = _RecentViewedDatabase({"BV_PROV_MODEL_VIEWED"})
    viewed_engine = ContentDiscoveryEngine(
        llm_service=_ModelStampedBatchLLMService(),
        database=viewed_db,  # type: ignore[arg-type]
    )
    viewed = DiscoveredContent(
        bvid="BV_PROV_MODEL_VIEWED", title="已经看过", source_strategy="trending"
    )

    await engine.evaluate_content_batch([filtered], _build_profile(), batch_size=1)
    await viewed_engine.evaluate_content_batch(
        [
            viewed,
            DiscoveredContent(
                bvid="BV_PROV_MODEL_FRESH", title="新内容", source_strategy="trending"
            ),
        ],
        _build_profile(),
    )

    assert filtered.score_source == "prefilter"
    assert filtered.teacher_model == ""
    assert viewed.score_source == "viewed"
    assert viewed.teacher_model == ""


def test_evaluation_persist_round_trips_teacher_model(tmp_path: Path) -> None:
    database = Database(tmp_path / "teacher-model-roundtrip.db")
    database.initialize()
    candidate_id = _enqueue_and_claim(database, "BV_PROV_TEACHER")

    updated = database.update_discovery_candidate_evaluations(
        [
            _evaluation(
                candidate_id,
                score_source="llm",
                llm_score_raw=0.77,
                teacher_model="deepseek/deepseek-v4-flash",
            )
        ]
    )
    assert updated == 1

    row = database.conn.execute(
        "SELECT score_source, llm_score_raw, teacher_model FROM discovery_candidates WHERE id = ?",
        (candidate_id,),
    ).fetchone()
    assert row["teacher_model"] == "deepseek/deepseek-v4-flash"

    teacher_rows = database.get_teacher_labeled_discovery_candidates()
    assert teacher_rows[0]["teacher_model"] == "deepseek/deepseek-v4-flash"
