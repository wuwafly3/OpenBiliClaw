"""Evaluation-time profile + negative-exemplar snapshot.

Wave 1 needs the compact eval-visible profile (and the negative titles the
teacher actually saw) frozen at labeling time. ``profile_digest`` is the
same short hash the eval cache already uses; the snapshot table stores the
payload so a later self-consistency probe can replay that prompt instead of
today's ``soul.json``.

The compact summary never includes ``personality_portrait``. Negative
exemplars may include disliked titles — the same strings already live in
the local event log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openbiliclaw.llm.prompt_cache import stable_json_digest

SNAPSHOT_SCHEMA_VERSION = 1


def profile_digest_payload(
    profile_summary: dict[str, Any],
    recall_pool: list[tuple[str, str, float]] | list[list[Any]],
) -> dict[str, Any]:
    """Payload hashed by ``ContentDiscoveryEngine._evaluation_profile_digest``."""

    return {"summary": profile_summary, "recall_pool": recall_pool}


def compute_profile_digest(
    profile_summary: dict[str, Any],
    recall_pool: list[tuple[str, str, float]] | list[list[Any]],
) -> str:
    return stable_json_digest(profile_digest_payload(profile_summary, recall_pool))


def compute_negative_digest(negative_examples: list[dict[str, Any]] | None) -> str:
    return stable_json_digest(negative_examples or [])


def parse_recall_pool(raw: object) -> list[tuple[str, str, float]]:
    if not isinstance(raw, list):
        return []
    parsed: list[tuple[str, str, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                parsed.append((str(item[0]), str(item[1]), float(item[2])))
            except (TypeError, ValueError):
                continue
    return parsed


def parse_negative_examples(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    examples: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        examples.append(
            {
                "title": str(item.get("title") or ""),
                "reason": str(item.get("reason") or ""),
                "age_days": item.get("age_days"),
            }
        )
    return examples


def dumps_snapshot_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads_snapshot_json(text: str) -> object:
    if not str(text or "").strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class EvaluationContextSnapshot:
    """Frozen eval-visible context for one teacher labeling call."""

    profile_digest: str
    negative_digest: str
    profile_summary: dict[str, Any]
    recall_pool: list[tuple[str, str, float]] = field(default_factory=list)
    negative_examples: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def recomputed_profile_digest(self) -> str:
        return compute_profile_digest(self.profile_summary, self.recall_pool)

    def recomputed_negative_digest(self) -> str:
        return compute_negative_digest(self.negative_examples)

    def digests_match(self) -> bool:
        return (
            self.profile_digest == self.recomputed_profile_digest()
            and self.negative_digest == self.recomputed_negative_digest()
        )
