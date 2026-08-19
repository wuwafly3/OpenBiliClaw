"""Cheap tags-only LLM channel for Wave 1 ML ranking features.

The full ``discovery.evaluate_batch`` prompt already emits topic / style /
temporal as a side-product of scoring. This module owns the cheaper
three-field channel and the parser that writes only ``tag_channel_*``
columns. It never touches teacher scores or teacher tags.
"""

from __future__ import annotations

import logging
from typing import Any

from openbiliclaw.discovery.style_keys import normalize_style_key
from openbiliclaw.discovery.temporal import normalize_temporal_class
from openbiliclaw.llm.json_utils import extract_llm_json_list, validated_text_field

logger = logging.getLogger(__name__)

# Placeholders that schema-defying models use instead of omitting a field.
# Treat as missing (hard-won rule 4) rather than persisting them as topics.
_TAG_PLACEHOLDERS = frozenset(
    {
        "",
        "-",
        "—",
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "unknown",
        "undefined",
        "待定",
        "未知",
        "无",
        "空",
    }
)
_TOPIC_GROUP_MAX_CHARS = 32

TAG_CHANNEL_SOURCE_LLM = "llm"


def parse_tag_channel_payload(content: str) -> list[dict[str, str]] | None:
    """Parse a tags-only JSON payload into validated row dicts.

    Returns ``None`` when the envelope cannot be read as a list of objects.
    Per-item invalid enums coerce to empty / ``unknown``; non-string identity
    or topic fields skip that item. Callers retry missing identities.
    """

    rows = extract_llm_json_list(content)
    if rows is None:
        return None
    parsed: list[dict[str, str]] = []
    for row in rows:
        item_id = _item_identity(row)
        if not item_id:
            logger.warning("Dropping tag-channel row without bvid/content_id")
            continue
        topic_raw = validated_text_field(
            row.get("topic_group", ""),
            field="topic_group",
            content_key=item_id,
        )
        if topic_raw is None:
            logger.warning("Coercing non-string topic_group to empty for %s", item_id)
            topic_group = ""
        else:
            topic_group = _normalize_topic_group(topic_raw, content_key=item_id)
        raw_style = row.get("style_key", "")
        style_key = normalize_style_key(raw_style)
        if str(raw_style or "").strip() and not style_key:
            logger.warning(
                "Coercing unknown style_key %r to empty for %s",
                raw_style,
                item_id,
            )
        temporal_class = normalize_temporal_class(row.get("temporal_class", ""))
        parsed.append(
            {
                "item_id": item_id,
                "topic_group": topic_group,
                "style_key": style_key,
                "temporal_class": temporal_class,
            }
        )
    return parsed


def _item_identity(row: dict[str, Any]) -> str:
    bvid = validated_text_field(row.get("bvid", ""), field="bvid", content_key="")
    if bvid:
        return bvid
    content_id = validated_text_field(
        row.get("content_id", ""),
        field="content_id",
        content_key="",
    )
    return content_id or ""


def _normalize_topic_group(value: str, *, content_key: str) -> str:
    collapsed = " ".join(value.split())
    if collapsed.lower() in _TAG_PLACEHOLDERS or collapsed in _TAG_PLACEHOLDERS:
        return ""
    if len(collapsed) > _TOPIC_GROUP_MAX_CHARS:
        logger.warning(
            "Truncating topic_group for %s from %s to %s chars",
            content_key,
            len(collapsed),
            _TOPIC_GROUP_MAX_CHARS,
        )
        return collapsed[:_TOPIC_GROUP_MAX_CHARS]
    return collapsed
