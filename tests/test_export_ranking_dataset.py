"""Unit tests for the dataset export row policy (ml-ranking Wave 0 step 3).

The script lives in ``scripts/`` (not a package), so it is loaded via
importlib; only the pure decision functions are tested here — the DB plumbing
is verified by running the script against the real database.
"""

from __future__ import annotations

import importlib.util
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
    assert export_module.effective_admission_threshold("explore_deep") == 0.58


def test_allowlist_matches_score_source_module() -> None:
    from openbiliclaw.discovery.score_source import LLM_JUDGMENT_SCORE_SOURCES

    assert export_module.LLM_JUDGMENT_SCORE_SOURCES == LLM_JUDGMENT_SCORE_SOURCES
