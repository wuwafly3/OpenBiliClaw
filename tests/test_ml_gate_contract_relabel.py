"""Unit tests for gate-contract relabel helpers."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

from openbiliclaw.discovery.eval_context import (
    EvaluationContextSnapshot,
    compute_negative_digest,
    compute_profile_digest,
    rewrite_snapshot_to_gate_contract,
)

_SPEC = importlib.util.spec_from_file_location(
    "ml_gate_contract_relabel",
    Path(__file__).resolve().parents[1] / "scripts" / "ml_gate_contract_relabel.py",
)
assert _SPEC is not None and _SPEC.loader is not None
relabel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(relabel)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ml_gate_contract_relabel.py"
_BILI_HOME = "哔哩哔哩 (゜-゜)つロ 干杯~-bilibili"


class _FakeDatabase:
    def __init__(self, rows: list[dict[str, Any]], snapshots: dict[tuple[str, str], Any]) -> None:
        self._rows = rows
        self._snapshots = snapshots

    def get_evaluation_context_snapshot(
        self,
        *,
        profile_digest: str,
        negative_digest: str,
    ) -> Any:
        return self._snapshots.get((profile_digest, negative_digest))


def _old_snapshot() -> EvaluationContextSnapshot:
    summary = {
        "interests": [{"name": "吉他", "weight": 0.9}],
        "recent_awareness": [{"observation": "最近在看系统课"}],
    }
    recall_pool = [("长尾兴趣", "音乐", 0.2)]
    negatives = [
        {"title": _BILI_HOME, "reason": "explicit_negative", "age_days": 0},
        {"title": "ChatGLM", "reason": "explicit_negative", "age_days": 1},
    ]
    return EvaluationContextSnapshot(
        profile_digest=compute_profile_digest(summary, recall_pool),
        negative_digest=compute_negative_digest(negatives),
        profile_summary=summary,
        recall_pool=recall_pool,
        negative_examples=negatives,
    )


def _new_snapshot() -> EvaluationContextSnapshot:
    old = _old_snapshot()
    return rewrite_snapshot_to_gate_contract(old).snapshot


def test_script_source_never_updates_candidate_scores() -> None:
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            sql = node.value.lower()
            if "update discovery_candidates" in sql:
                raise AssertionError("relabel script must not UPDATE discovery_candidates")
            if (
                "update" in sql
                and "discovery_candidates" in sql
                and any(column in sql for column in relabel.FORBIDDEN_CANDIDATE_UPDATES)
            ):
                raise AssertionError(sql)


def test_output_row_keeps_old_and_new_scores() -> None:
    rewrite = rewrite_snapshot_to_gate_contract(_old_snapshot())
    row = relabel.output_row(
        {
            "id": 42,
            "source_platform": "bilibili",
            "source_strategy": "search",
            "teacher_score": 0.72,
            "teacher_model": "openai/deepseek-v4-flash",
        },
        rewrite,
        new_score=0.41,
        new_teacher_model="openai/deepseek-v4-flash",
        new_score_source="llm",
    )
    assert row["candidate_id"] == 42
    assert row["old_llm_score_raw"] == 0.72
    assert row["new_llm_score_raw"] == 0.41
    assert row["old_y"] == 1
    assert row["new_y"] == 0
    assert row["old_profile_digest"] != row["new_profile_digest"]
    assert row["had_recent"] is True
    assert row["dropped_shell"] == 1


def test_collect_relabel_cohort_keeps_old_contract_only(monkeypatch: Any) -> None:
    old = _old_snapshot()
    new = _new_snapshot()
    rows = [
        {
            "id": 1,
            "teacher_model": "openai/deepseek-v4-flash",
            "teacher_score": 0.8,
            "profile_digest": old.profile_digest,
            "negative_digest": old.negative_digest,
            "source_platform": "bilibili",
            "source_strategy": "search",
        },
        {
            "id": 2,
            "teacher_model": "openai_compatible/deepseek-v4-flash",
            "teacher_score": 0.8,
            "profile_digest": old.profile_digest,
            "negative_digest": old.negative_digest,
            "source_platform": "bilibili",
            "source_strategy": "search",
        },
        {
            "id": 3,
            "teacher_model": "openai/deepseek-v4-flash",
            "teacher_score": 0.2,
            "profile_digest": new.profile_digest,
            "negative_digest": new.negative_digest,
            "source_platform": "youtube",
            "source_strategy": "yt_search",
        },
        {
            "id": 4,
            "teacher_model": "openai/deepseek-v4-flash",
            "teacher_score": 0.3,
            "profile_digest": "",
            "negative_digest": "",
            "source_platform": "twitter",
            "source_strategy": "x-search",
        },
    ]
    snapshots = {
        (old.profile_digest, old.negative_digest): old,
        (new.profile_digest, new.negative_digest): new,
    }
    fake = _FakeDatabase(rows, snapshots)
    monkeypatch.setattr(relabel, "load_teacher_allowlist_rows", lambda _db: rows)
    groups, counts = relabel.collect_relabel_cohort(
        fake,  # type: ignore[arg-type]
        provider_type="openai",
        model="deepseek-v4-flash",
    )
    kept_ids = [int(row["id"]) for _rewrite, group in groups for row in group]
    assert kept_ids == [1]
    assert counts["kept"] == 1
    assert counts["dropped_provider_mismatch"] == 1
    assert counts["already_new_contract"] == 1
    assert counts["dropped_empty_digest"] == 1
    assert counts["groups"] == 1


def test_collect_relabel_cohort_resume_skips_written_ids(monkeypatch: Any) -> None:
    old = _old_snapshot()
    rows = [
        {
            "id": 11,
            "teacher_model": "openai/deepseek-v4-flash",
            "teacher_score": 0.8,
            "profile_digest": old.profile_digest,
            "negative_digest": old.negative_digest,
            "source_platform": "bilibili",
            "source_strategy": "search",
        },
        {
            "id": 12,
            "teacher_model": "openai/deepseek-v4-flash",
            "teacher_score": 0.4,
            "profile_digest": old.profile_digest,
            "negative_digest": old.negative_digest,
            "source_platform": "xiaohongshu",
            "source_strategy": "xhs-extension-search",
        },
    ]
    fake = _FakeDatabase(rows, {(old.profile_digest, old.negative_digest): old})
    monkeypatch.setattr(relabel, "load_teacher_allowlist_rows", lambda _db: rows)
    groups, counts = relabel.collect_relabel_cohort(
        fake,  # type: ignore[arg-type]
        provider_type="openai",
        model="deepseek-v4-flash",
        resume_ids={11},
    )
    kept_ids = [int(row["id"]) for _rewrite, group in groups for row in group]
    assert kept_ids == [12]
    assert counts["skipped_resume"] == 1
    assert counts["kept"] == 1
