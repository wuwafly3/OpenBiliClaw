"""Unit tests for the dataset export row policy (ml-ranking Wave 0 step 3).

The script lives in ``scripts/`` (not a package), so it is loaded via
importlib. Pure decision functions and an in-memory sqlite snapshot of
``build_export_rows`` are covered here.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "export_ranking_dataset",
    Path(__file__).resolve().parents[1] / "scripts" / "export_ranking_dataset.py",
)
assert _SPEC is not None and _SPEC.loader is not None
export_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_module)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relevance_score": 0.0,
        "score_source": "",
        "llm_score_raw": None,
    }
    row.update(overrides)
    return row


def test_provenance_rows_use_llm_score_raw() -> None:
    resolved = export_module.resolve_teacher_label(
        _row(score_source="llm", llm_score_raw=0.83, relevance_score=0.83), None
    )
    assert resolved == (0.83, "provenance")

    # Cap-zeroed rows keep their pre-cap teacher judgment.
    resolved = export_module.resolve_teacher_label(
        _row(score_source="cap_franchise", llm_score_raw=0.76, relevance_score=0.0), None
    )
    assert resolved == (0.76, "provenance")


def test_provenance_row_without_raw_score_is_dropped() -> None:
    assert export_module.resolve_teacher_label(_row(score_source="llm"), None) is None


def test_legacy_nonzero_rows_are_teacher_judgments() -> None:
    resolved = export_module.resolve_teacher_label(_row(relevance_score=0.42), None)
    assert resolved == (0.42, "legacy_nonzero")


def test_legacy_zero_rows_recover_from_audit_or_drop() -> None:
    resolved = export_module.resolve_teacher_label(_row(relevance_score=0.0), 0.55)
    assert resolved == (0.55, "legacy_audit_recovered")
    assert export_module.resolve_teacher_label(_row(relevance_score=0.0), None) is None


def test_synthetic_sources_are_dropped() -> None:
    for source in ("prefilter", "viewed", "truncated", "response_missing", "eval_error"):
        assert export_module.resolve_teacher_label(_row(score_source=source), 0.9) is None


def test_admission_threshold_uses_explore_floor() -> None:
    assert export_module.effective_admission_threshold("search") == 0.60
    assert export_module.effective_admission_threshold("explore") == 0.58
    assert export_module.effective_admission_threshold("Explore") == 0.58
    assert export_module.effective_admission_threshold(" explore ") == 0.58


def test_admission_threshold_rejects_explore_prefixes() -> None:
    assert export_module.effective_admission_threshold("explore_deep") == 0.60
    assert export_module.effective_admission_threshold("explore-backfill") == 0.60
    assert export_module.effective_admission_threshold("") == 0.60


def test_admission_threshold_matches_production_policy_floor() -> None:
    from openbiliclaw.discovery.admission import (
        effective_admission_threshold as production_threshold,
    )

    for strategy in (
        "search",
        "explore",
        "Explore",
        " explore ",
        "explore_deep",
        "explore-backfill",
        "trending",
        "",
        None,
    ):
        assert export_module.effective_admission_threshold(strategy) == production_threshold(
            strategy
        )


def test_allowlist_matches_score_source_module() -> None:
    from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES

    assert export_module.LLM_JUDGMENT_SCORE_SOURCES == LLM_JUDGMENT_SCORE_SOURCES


def _memory_export_db(rows: list[dict[str, Any]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    engagement = ", ".join(
        f"{column} INTEGER DEFAULT 0" for column in export_module.ENGAGEMENT_COLUMNS
    )
    conn.execute(
        f"""
        CREATE TABLE discovery_candidates (
            id INTEGER PRIMARY KEY,
            candidate_key TEXT,
            status TEXT,
            source_platform TEXT,
            source_strategy TEXT,
            content_type TEXT,
            candidate_tier TEXT,
            title TEXT,
            description TEXT,
            body_text TEXT,
            published_at TEXT,
            evaluated_at TEXT,
            duration INTEGER DEFAULT 0,
            relevance_score REAL DEFAULT 0,
            score_source TEXT DEFAULT '',
            llm_score_raw REAL,
            teacher_model TEXT,
            style_key TEXT,
            temporal_class TEXT,
            topic_group TEXT,
            franchise_key TEXT,
            {engagement}
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE evaluator_prefilter_shadow_audit (
            id INTEGER PRIMARY KEY,
            candidate_hash TEXT,
            similarity REAL,
            llm_score REAL,
            would_filter INTEGER,
            context_class TEXT,
            profile_digest TEXT,
            created_at TEXT
        )
        """
    )
    for index, row in enumerate(rows, start=1):
        payload = {
            "id": index,
            "candidate_key": f"bilibili:BV{index}",
            "status": "evaluated",
            "source_platform": "bilibili",
            "source_strategy": "search",
            "content_type": "video",
            "candidate_tier": "primary",
            "title": "t",
            "description": "",
            "body_text": "",
            "published_at": "",
            "evaluated_at": "2026-08-18T00:00:00+00:00",
            "duration": 0,
            "relevance_score": 0.0,
            "score_source": "",
            "llm_score_raw": None,
            "teacher_model": "deepseek/deepseek-v4-flash",
            "style_key": "",
            "temporal_class": "",
            "topic_group": "",
            "franchise_key": "",
        }
        payload.update(row)
        placeholders = ", ".join(f":{key}" for key in payload)
        conn.execute(
            f"INSERT INTO discovery_candidates ({', '.join(payload)}) VALUES ({placeholders})",
            payload,
        )
    conn.commit()
    return conn


def test_export_drops_prefilter_rows_even_when_score_is_nonzero() -> None:
    conn = _memory_export_db(
        [
            {
                "candidate_key": "bilibili:BVLLM",
                "score_source": "llm",
                "llm_score_raw": 0.72,
                "relevance_score": 0.72,
            },
            {
                "candidate_key": "bilibili:BVPRE",
                "score_source": "prefilter",
                "llm_score_raw": None,
                "relevance_score": 0.41,
            },
        ]
    )
    exported, policy_counts = export_module.build_export_rows(conn)
    conn.close()
    assert [row["candidate_key"] for row in exported] == ["bilibili:BVLLM"]
    assert policy_counts["provenance"] == 1
    assert policy_counts["dropped"] == 1


def test_export_labels_explore_backfill_at_default_threshold() -> None:
    conn = _memory_export_db(
        [
            {
                "candidate_key": "bilibili:BVEXP",
                "source_strategy": "explore",
                "score_source": "llm",
                "llm_score_raw": 0.59,
                "relevance_score": 0.59,
            },
            {
                "candidate_key": "bilibili:BVFILL",
                "source_strategy": "explore-backfill",
                "score_source": "llm",
                "llm_score_raw": 0.59,
                "relevance_score": 0.59,
            },
        ]
    )
    exported, _policy_counts = export_module.build_export_rows(conn)
    conn.close()
    by_key = {row["candidate_key"]: row for row in exported}
    assert by_key["bilibili:BVEXP"]["admission_threshold"] == 0.58
    assert by_key["bilibili:BVEXP"]["y"] == 1
    assert by_key["bilibili:BVFILL"]["admission_threshold"] == 0.60
    assert by_key["bilibili:BVFILL"]["y"] == 0
