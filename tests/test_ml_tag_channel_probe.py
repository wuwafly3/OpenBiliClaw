"""Unit tests for Wave 1 tag-channel probe helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ml_tag_channel_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "ml_tag_channel_probe.py",
)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_stratified_sample_round_robins_thin_platforms() -> None:
    rows = [{"source_platform": "bilibili", "id": i} for i in range(80)]
    rows.extend({"source_platform": "youtube", "id": 100 + i} for i in range(5))
    rows.extend({"source_platform": "x", "id": 200 + i} for i in range(5))
    sample = probe.stratified_sample(rows, limit=12, seed=19)
    platforms = {str(row["source_platform"]) for row in sample}
    assert platforms == {"bilibili", "youtube", "x"}
    assert len(sample) == 12
    assert sum(1 for row in sample if row["source_platform"] == "youtube") >= 1
    assert sum(1 for row in sample if row["source_platform"] == "x") >= 1


def test_agreement_stats_skip_untagged_and_note_topic() -> None:
    pairs = [
        (
            {
                "platform": "bilibili",
                "topic_group": "强化学习",
                "style_key": "hands_on",
                "temporal_class": "evergreen",
            },
            {
                "source": "llm",
                "topic_group": "强化学习",
                "style_key": "hands_on",
                "temporal_class": "evergreen",
            },
        ),
        (
            {
                "platform": "youtube",
                "topic_group": "吉他",
                "style_key": "story_immersion",
                "temporal_class": "current",
            },
            {
                "source": "llm",
                "topic_group": "木吉他保养",
                "style_key": "story_immersion",
                "temporal_class": "unknown",
            },
        ),
        (
            {
                "platform": "x",
                "topic_group": "新闻",
                "style_key": "social_chat",
                "temporal_class": "breaking",
            },
            {
                "source": "",
                "topic_group": "",
                "style_key": "",
                "temporal_class": "unknown",
            },
        ),
    ]
    stats = probe.agreement_stats(pairs)
    assert stats["requested"] == 3
    assert stats["compared"] == 2
    assert stats["style_exact"] == 2
    assert stats["temporal_exact"] == 1
    assert stats["topic_exact"] == 1
    assert stats["by_platform"]["bilibili"]["n"] == 1
    assert stats["by_platform"]["youtube"]["temporal"] == 0
