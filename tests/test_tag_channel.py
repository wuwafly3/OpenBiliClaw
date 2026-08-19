"""Tests for the Wave 1 cheap tags-only LLM channel."""

from __future__ import annotations

from dataclasses import dataclass, field

from openbiliclaw.discovery.candidate_pipeline import (
    CandidateEvalClaim,
    DiscoveryCandidatePipeline,
)
from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent
from openbiliclaw.discovery.tag_channel import parse_tag_channel_payload
from openbiliclaw.llm.prompts import build_batch_tag_prompt


def test_build_batch_tag_prompt_system_has_no_profile_or_score() -> None:
    messages = build_batch_tag_prompt(
        content_items=[{"bvid": "BV1A", "title": "标题", "description": "简介"}]
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert "画像" not in system
    assert "profile" not in system.lower()
    assert "topic_group" in system
    assert "style_key" in system
    assert "temporal_class" in system
    assert '"results"' in system
    assert "score" not in system.lower()
    assert "franchise" not in system.lower()
    assert "admission" not in system.lower()
    assert "不要" not in system
    assert "<content_batch>" in user
    assert "BV1A" in user
    assert "<profile" not in user


def test_build_batch_tag_prompt_puts_per_item_data_in_user_message() -> None:
    first = build_batch_tag_prompt(content_items=[{"bvid": "BV1A", "title": "甲"}])
    second = build_batch_tag_prompt(content_items=[{"bvid": "BV1B", "title": "乙"}])
    assert first[0]["content"] == second[0]["content"]
    assert first[1]["content"] != second[1]["content"]


def test_parse_tag_channel_payload_coerces_invalid_enums() -> None:
    payload = """
    {"results": [
      {"bvid": "BV1OK", "topic_group": "人工智能",
       "style_key": "deep_focus", "temporal_class": "evergreen"},
      {"bvid": "BV1BAD", "topic_group": "N/A",
       "style_key": "not_a_style", "temporal_class": "whenever"},
      {"bvid": "BV1LEGACY", "topic_group": "游戏",
       "style_key": "deep_dive", "temporal_class": "CURRENT"}
    ]}
    """
    rows = parse_tag_channel_payload(payload)
    assert rows is not None
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["BV1OK"]["topic_group"] == "人工智能"
    assert by_id["BV1OK"]["style_key"] == "deep_focus"
    assert by_id["BV1OK"]["temporal_class"] == "evergreen"
    assert by_id["BV1BAD"]["topic_group"] == ""
    assert by_id["BV1BAD"]["style_key"] == ""
    assert by_id["BV1BAD"]["temporal_class"] == "unknown"
    assert by_id["BV1LEGACY"]["style_key"] == "deep_focus"
    assert by_id["BV1LEGACY"]["temporal_class"] == "current"


def test_parse_tag_channel_payload_returns_none_for_garbage() -> None:
    assert parse_tag_channel_payload("not json") is None


@dataclass
class _TagResponse:
    content: str
    provider: str = "openai"
    model: str = "deepseek-v4-flash"


class _RecordingTagLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    async def complete_structured_task(
        self,
        *,
        system_instruction: str,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        caller: str = "",
        reasoning_effort: str | None = None,
        inject_core_memory: bool = True,
        json_mode: bool = False,
    ) -> object:
        self.calls.append(
            {
                "caller": caller,
                "inject_core_memory": inject_core_memory,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "system_instruction": system_instruction,
                "user_input": user_input,
            }
        )
        return _TagResponse(self.payload)


class _SequencedTagLLM:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, object]] = []

    async def complete_structured_task(
        self,
        *,
        system_instruction: str = "",
        user_input: str = "",
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        caller: str = "",
        reasoning_effort: str | None = None,
        inject_core_memory: bool = True,
        json_mode: bool = False,
    ) -> object:
        self.calls.append({"caller": caller, "user_input": user_input})
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _TagResponse(self.payloads[index])


def _three_item_payload() -> str:
    return """
    {"results": [
      {"bvid": "BV1A", "topic_group": "人工智能",
       "style_key": "deep_focus", "temporal_class": "evergreen"},
      {"bvid": "BV1B", "topic_group": "游戏",
       "style_key": "not_a_style", "temporal_class": "current"},
      {"bvid": "BV1C", "topic_group": "历史",
       "style_key": "story_immersion", "temporal_class": "historical"}
    ]}
    """


def _sample_contents() -> list[DiscoveredContent]:
    return [
        DiscoveredContent(bvid="BV1A", title="甲", relevance_score=0.81, topic_group="教师主题"),
        DiscoveredContent(bvid="BV1B", title="乙", relevance_score=0.44, style_key="deep_focus"),
        DiscoveredContent(
            bvid="BV1C",
            title="丙",
            relevance_score=0.62,
            temporal_class="evergreen",
        ),
    ]


async def test_tag_content_batch_uses_tag_batch_caller_and_does_not_mutate_scores() -> None:
    llm = _RecordingTagLLM(_three_item_payload())
    engine = ContentDiscoveryEngine(llm_service=llm)
    contents = _sample_contents()
    teacher_topics = [item.topic_group for item in contents]
    teacher_styles = [item.style_key for item in contents]
    teacher_temporal = [item.temporal_class for item in contents]
    scores = [item.relevance_score for item in contents]

    tagged = await engine.tag_content_batch(contents)

    assert tagged is contents
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["caller"] == "discovery.tag_batch"
    assert call["inject_core_memory"] is False
    assert call["json_mode"] is True
    assert call["max_tokens"] == 256 + 64 * 3
    assert call["reasoning_effort"] == ""
    assert [item.relevance_score for item in contents] == scores
    assert [item.topic_group for item in contents] == teacher_topics
    assert [item.style_key for item in contents] == teacher_styles
    assert [item.temporal_class for item in contents] == teacher_temporal
    assert contents[0].tag_channel_topic_group == "人工智能"
    assert contents[0].tag_channel_style_key == "deep_focus"
    assert contents[0].tag_channel_source == "llm"
    assert contents[0].tag_channel_model == "openai/deepseek-v4-flash"
    assert contents[1].tag_channel_style_key == ""
    assert contents[1].tag_channel_temporal_class == "current"
    assert contents[2].tag_channel_temporal_class == "historical"


async def test_tag_content_batch_forwards_max_tokens_override() -> None:
    llm = _RecordingTagLLM(_three_item_payload())
    engine = ContentDiscoveryEngine(llm_service=llm)
    await engine.tag_content_batch(_sample_contents()[:1], max_tokens=4096)
    assert llm.calls[0]["max_tokens"] == 4096


def _untagged_contents(count: int) -> list[DiscoveredContent]:
    return [
        DiscoveredContent(bvid=f"BVSC{i:02d}", title=f"条目{i}", source_strategy="search")
        for i in range(count)
    ]


async def test_tag_content_batch_scales_max_tokens_with_chunk_size() -> None:
    llm = _RecordingTagLLM(_three_item_payload())
    engine = ContentDiscoveryEngine(llm_service=llm)

    # A full default 45-item batch must get far more than the old fixed
    # 1024 budget, and a hard-cap 90-item batch must hit the 4096 ceiling.
    await engine.tag_content_batch(_untagged_contents(45), batch_size=45)
    assert llm.calls[0]["max_tokens"] == 256 + 64 * 45

    capped_llm = _RecordingTagLLM(_three_item_payload())
    capped_engine = ContentDiscoveryEngine(llm_service=capped_llm)
    await capped_engine.tag_content_batch(_untagged_contents(90), batch_size=90)
    assert capped_llm.calls[0]["max_tokens"] == 4096


async def test_tag_content_batch_does_not_cache_failed_results() -> None:
    llm = _SequencedTagLLM(["not json", "not json"])
    engine = ContentDiscoveryEngine(llm_service=llm)
    contents = _sample_contents()[:1]

    await engine.tag_content_batch(contents)
    await engine.tag_content_batch(contents)

    assert contents[0].tag_channel_source == ""
    assert len(llm.calls) >= 2


async def test_tag_content_batch_caches_successful_tags() -> None:
    llm = _RecordingTagLLM(_three_item_payload())
    engine = ContentDiscoveryEngine(llm_service=llm)
    first = _sample_contents()
    second = _sample_contents()

    await engine.tag_content_batch(first)
    await engine.tag_content_batch(second)

    assert len(llm.calls) == 1
    assert second[0].tag_channel_topic_group == "人工智能"
    assert second[0].relevance_score == 0.81


async def test_tag_content_batch_retries_missing_identities_once() -> None:
    first = (
        '{"results": [{"bvid": "BV1A", "topic_group": "甲",'
        ' "style_key": "deep_focus", "temporal_class": "evergreen"}]}'
    )
    second = (
        '{"results": [{"bvid": "BV1B", "topic_group": "乙",'
        ' "style_key": "mood_release", "temporal_class": "current"}]}'
    )
    llm = _SequencedTagLLM([first, second])
    engine = ContentDiscoveryEngine(llm_service=llm)
    contents = _sample_contents()[:2]

    await engine.tag_content_batch(contents)

    assert len(llm.calls) == 2
    assert contents[0].tag_channel_topic_group == "甲"
    assert contents[1].tag_channel_topic_group == "乙"


class _TagAndEvalEngine:
    def __init__(self, *, mode: str = "off", fail_tag: bool = False) -> None:
        self.tag_channel_mode = mode
        self.tag_calls = 0
        self.eval_calls = 0
        self.fail_tag = fail_tag
        self.tagged_bvids: list[str] = []
        self.tag_kwargs: list[dict[str, object]] = []

    async def tag_content_batch(
        self,
        items: list[DiscoveredContent],
        **kwargs: object,
    ) -> list[DiscoveredContent]:
        self.tag_calls += 1
        self.tag_kwargs.append(dict(kwargs))
        if self.fail_tag:
            raise RuntimeError("tag-channel down")
        for item in items:
            item.tag_channel_topic_group = "人工智能"
            item.tag_channel_style_key = "deep_focus"
            item.tag_channel_temporal_class = "evergreen"
            item.tag_channel_source = "llm"
            item.tag_channel_model = "test/model"
            self.tagged_bvids.append(item.bvid)
        return items

    async def evaluate_content_batch(
        self,
        items: list[object],
        profile: object,
        **kwargs: object,
    ) -> list[float]:
        self.eval_calls += 1
        return [0.9] * len(items)


@dataclass
class _PersistDB:
    rows: list[dict[str, object]] = field(default_factory=list)

    def update_discovery_candidate_tag_channel(self, rows: list[dict[str, object]]) -> int:
        self.rows.extend(list(rows))
        return len(rows)


def _claim_pair(*, first_tagged: bool = False) -> CandidateEvalClaim:
    first = DiscoveredContent(bvid="BV1A", title="甲")
    second = DiscoveredContent(bvid="BV1B", title="乙")
    if first_tagged:
        first.tag_channel_source = "llm"
        first.tag_channel_topic_group = "已有"
    return CandidateEvalClaim(
        token="tok",
        rows=({"id": 11}, {"id": 12}),
        items=(first, second),
    )


async def test_evaluate_claim_off_makes_zero_tag_calls() -> None:
    engine = _TagAndEvalEngine(mode="off")
    db = _PersistDB()
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    outcome = await pipeline.evaluate_claim(_claim_pair(), object())

    assert engine.tag_calls == 0
    assert engine.eval_calls == 1
    assert db.rows == []
    assert outcome.scores == (0.9, 0.9)


async def test_evaluate_claim_shadow_does_not_tag_on_hot_path() -> None:
    engine = _TagAndEvalEngine(mode="shadow")
    pipeline = DiscoveryCandidatePipeline(
        database=_PersistDB(),
        discovery_engine=engine,  # type: ignore[arg-type]
    )
    await pipeline.evaluate_claim(_claim_pair(), object())
    assert engine.tag_calls == 0
    assert engine.eval_calls == 1


async def test_evaluate_claim_enforce_tags_then_still_evaluates() -> None:
    engine = _TagAndEvalEngine(mode="enforce")
    db = _PersistDB()
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    claim = _claim_pair()
    await pipeline.evaluate_claim(claim, object())

    assert engine.tag_calls == 1
    assert engine.eval_calls == 1
    assert engine.tagged_bvids == ["BV1A", "BV1B"]
    assert [row["candidate_id"] for row in db.rows] == [11, 12]
    assert claim.items[0].tag_channel_source == "llm"
    # The pipeline must not override batch_size: untagged claims can reach
    # the evaluate hard cap (90) and one giant call would blow the tag
    # channel's output token budget.
    assert engine.tag_kwargs == [{}]


async def test_evaluate_claim_enforce_skips_already_tagged_items() -> None:
    engine = _TagAndEvalEngine(mode="enforce")
    db = _PersistDB()
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    await pipeline.evaluate_claim(_claim_pair(first_tagged=True), object())

    assert engine.tagged_bvids == ["BV1B"]
    assert [row["candidate_id"] for row in db.rows] == [12]


async def test_evaluate_claim_enforce_tag_failure_still_evaluates() -> None:
    engine = _TagAndEvalEngine(mode="enforce", fail_tag=True)
    db = _PersistDB()
    pipeline = DiscoveryCandidatePipeline(database=db, discovery_engine=engine)  # type: ignore[arg-type]
    await pipeline.evaluate_claim(_claim_pair(), object())

    assert engine.tag_calls == 1
    assert engine.eval_calls == 1
    assert db.rows == []
