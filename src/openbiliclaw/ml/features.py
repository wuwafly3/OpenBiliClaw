"""Deterministic admission features for Wave 1 (spec S1.2 / S1.2a).

Runtime inference may only feed cheap-channel tags (``tag_channel_*``).
Teacher ``topic_group`` / ``style_key`` / ``temporal_*`` / ``franchise_key``
are training-oracle inputs and must not be read on the live path.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from openbiliclaw.discovery.style_keys import VALID_STYLE_KEYS, normalize_style_key
from openbiliclaw.discovery.temporal import TEMPORAL_CLASSES, normalize_temporal_class

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FEATURE_VERSION = "admission-teacher-tags-v1"
TAG_SOURCE_CHEAP: Literal["cheap"] = "cheap"
TAG_SOURCE_TEACHER_ORACLE: Literal["teacher_oracle"] = "teacher_oracle"
ENGAGEMENT_COLUMNS: tuple[str, ...] = (
    "view_count",
    "like_count",
    "favorite_count",
    "collect_count",
    "comment_count",
    "share_count",
    "danmaku_count",
    "reply_count",
    "retweet_count",
    "bookmark_count",
)
TOP_STRATEGIES = 8
TOP_TOPICS = 15
STYLE_NAMES: tuple[str, ...] = tuple(sorted(VALID_STYLE_KEYS)) + ("style_empty",)
TEMPORAL_NAMES: tuple[str, ...] = tuple(sorted(TEMPORAL_CLASSES))
_BANNED_NAME_TOKENS = ("relevance_score", "llm_score_raw", "teacher_score", "franchise")


def feature_record_from_content(
    content: object,
    *,
    tag_source: Literal["cheap", "teacher_oracle"],
) -> dict[str, Any]:
    """Build one encoder record. Cheap tags are the only live-path tag source."""

    if tag_source == TAG_SOURCE_CHEAP:
        style = normalize_style_key(getattr(content, "tag_channel_style_key", "")) or "style_empty"
        temporal = normalize_temporal_class(getattr(content, "tag_channel_temporal_class", ""))
        topic = " ".join(str(getattr(content, "tag_channel_topic_group", "") or "").split())
    elif tag_source == TAG_SOURCE_TEACHER_ORACLE:
        style = normalize_style_key(getattr(content, "style_key", "")) or "style_empty"
        temporal = normalize_temporal_class(getattr(content, "temporal_class", ""))
        topic = " ".join(str(getattr(content, "topic_group", "") or "").split())
    else:
        raise ValueError(f"unsupported tag_source {tag_source!r}")
    record: dict[str, Any] = {
        "platform": str(getattr(content, "source_platform", "") or "unknown") or "unknown",
        "strategy": str(getattr(content, "source_strategy", "") or "unknown") or "unknown",
        "content_type": str(getattr(content, "content_type", "") or "video") or "video",
        "style": style if style in STYLE_NAMES else "style_empty",
        "temporal": temporal,
        "topic": topic or "topic_empty",
        "title_len": len(str(getattr(content, "title", "") or "")),
        "desc_len": len(str(getattr(content, "description", "") or "")),
        "body_len": len(str(getattr(content, "body_text", "") or "")),
        "duration_s": float(getattr(content, "duration", 0) or 0),
        "rating_score": float(getattr(content, "rating_score", 0) or 0),
        "source_rank": float(getattr(content, "source_rank", 0) or 0),
        **{col: float(getattr(content, col, 0) or 0) for col in ENGAGEMENT_COLUMNS},
    }
    return record


def build_vocab(
    records: Sequence[Mapping[str, Any]],
    *,
    strategy_names: list[str] | None = None,
    topic_names: list[str] | None = None,
    context_names: list[str] | None = None,
    platform_names: list[str] | None = None,
    content_type_names: list[str] | None = None,
) -> dict[str, list[str]]:
    platforms = platform_names or sorted({str(r["platform"]) for r in records})
    types = content_type_names or sorted({str(r["content_type"]) for r in records})
    strategies = strategy_names or [
        item for item, _ in Counter(str(r["strategy"]) for r in records).most_common(TOP_STRATEGIES)
    ]
    contexts = context_names or sorted({str(r.get("context", "no_audit")) for r in records})
    topics = topic_names or [
        item for item, _ in Counter(str(r["topic"]) for r in records).most_common(TOP_TOPICS)
    ]
    return {
        "platforms": list(platforms),
        "content_types": list(types),
        "strategies": list(strategies),
        "contexts": list(contexts),
        "topics": list(topics),
    }


def canonical_feature_names(vocab: Mapping[str, Sequence[str]]) -> list[str]:
    names: list[str] = ["sim", "sim_available", "would_filter"]
    for col in ENGAGEMENT_COLUMNS:
        names.extend((f"log1p_{col}", f"has_{col}"))
    names.extend(
        (
            "log1p_duration",
            "has_duration",
            "log1p_title_len",
            "log1p_desc_len",
            "log1p_body_len",
            "rating_score",
            "log1p_source_rank",
        )
    )
    names.extend(f"platform={item}" for item in vocab["platforms"])
    names.extend(f"content_type={item}" for item in vocab["content_types"])
    names.extend(f"strategy={item}" for item in vocab["strategies"])
    names.append("strategy=other")
    names.extend(f"context={item}" for item in vocab["contexts"])
    names.extend(f"style={item}" for item in STYLE_NAMES)
    names.extend(f"temporal={item}" for item in TEMPORAL_NAMES)
    names.extend(f"topic={item}" for item in vocab["topics"])
    names.append("topic=other")
    _assert_no_leak(names)
    return names


def _assert_no_leak(names: Sequence[str]) -> None:
    joined = " ".join(names)
    if any(token in joined for token in _BANNED_NAME_TOKENS):
        raise RuntimeError(f"score/franchise leaked into features: {list(names)}")


def feature_map(record: Mapping[str, Any], vocab: Mapping[str, Sequence[str]]) -> dict[str, float]:
    values: dict[str, float] = {
        "sim": float(record.get("sim", 0.0) or 0.0),
        "sim_available": 1.0 if "sim" in record else 0.0,
        "would_filter": float(record.get("would_filter", 0.0) or 0.0),
    }
    for col in ENGAGEMENT_COLUMNS:
        raw = float(record.get(col, 0.0) or 0.0)
        values[f"log1p_{col}"] = float(np.log1p(max(0.0, raw)))
        values[f"has_{col}"] = 1.0 if raw > 0 else 0.0
    duration = float(record.get("duration_s", 0.0) or 0.0)
    values["log1p_duration"] = float(np.log1p(max(0.0, duration)))
    values["has_duration"] = 1.0 if duration > 0 else 0.0
    values["log1p_title_len"] = float(np.log1p(max(0.0, float(record.get("title_len", 0) or 0))))
    values["log1p_desc_len"] = float(np.log1p(max(0.0, float(record.get("desc_len", 0) or 0))))
    values["log1p_body_len"] = float(np.log1p(max(0.0, float(record.get("body_len", 0) or 0))))
    values["rating_score"] = float(record.get("rating_score", 0.0) or 0.0)
    values["log1p_source_rank"] = float(
        np.log1p(max(0.0, float(record.get("source_rank", 0.0) or 0.0)))
    )
    platform = str(record.get("platform") or "unknown")
    for item in vocab["platforms"]:
        values[f"platform={item}"] = 1.0 if platform == item else 0.0
    content_type = str(record.get("content_type") or "video")
    for item in vocab["content_types"]:
        values[f"content_type={item}"] = 1.0 if content_type == item else 0.0
    strategy = str(record.get("strategy") or "unknown")
    strategies = list(vocab["strategies"])
    for item in strategies:
        values[f"strategy={item}"] = 1.0 if strategy == item else 0.0
    values["strategy=other"] = 1.0 if strategy not in strategies else 0.0
    context = str(record.get("context", "no_audit") or "no_audit")
    for item in vocab["contexts"]:
        values[f"context={item}"] = 1.0 if context == item else 0.0
    style = str(record.get("style") or "style_empty")
    if style not in STYLE_NAMES:
        style = "style_empty"
    for item in STYLE_NAMES:
        values[f"style={item}"] = 1.0 if style == item else 0.0
    temporal = str(record.get("temporal") or "unknown")
    for item in TEMPORAL_NAMES:
        values[f"temporal={item}"] = 1.0 if temporal == item else 0.0
    topic = str(record.get("topic") or "topic_empty")
    topics = list(vocab["topics"])
    for item in topics:
        values[f"topic={item}"] = 1.0 if topic == item else 0.0
    values["topic=other"] = 1.0 if topic not in topics else 0.0
    return values


def encode_features(
    records: Sequence[Mapping[str, Any]],
    *,
    vocab: Mapping[str, Sequence[str]] | None = None,
    feature_names: Sequence[str] | None = None,
    strategy_names: list[str] | None = None,
    topic_names: list[str] | None = None,
    context_names: list[str] | None = None,
    platform_names: list[str] | None = None,
    content_type_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Encode records into the frozen Wave 1 feature matrix.

    Passing ``feature_names`` (from an artifact) pins column order. Teacher
    scores are never columns.
    """

    resolved_vocab = dict(
        vocab
        or build_vocab(
            records,
            strategy_names=strategy_names,
            topic_names=topic_names,
            context_names=context_names,
            platform_names=platform_names,
            content_type_names=content_type_names,
        )
    )
    names = (
        list(feature_names)
        if feature_names is not None
        else canonical_feature_names(resolved_vocab)
    )
    _assert_no_leak(names)
    rows = [
        [feature_map(record, resolved_vocab).get(name, 0.0) for name in names] for record in records
    ]
    return (
        np.asarray(rows, dtype=float),
        names,
        {key: list(value) for key, value in resolved_vocab.items()},
    )
