"""Tests for Database.update_discovery_candidate_temporal_relabel."""

from pathlib import Path
from typing import Any

import pytest

from openbiliclaw.discovery.candidate_pool import DiscoveryCandidateWrite
from openbiliclaw.storage.database import Database

_RELABEL_EVALUATED_AT = "2026-08-28T00:00:00Z"


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    database.initialize()
    yield database
    database.close()


def _seed_candidate(db: Database, title: str = "候选标题") -> int:
    db.enqueue_discovery_candidates(
        [
            DiscoveryCandidateWrite(
                candidate_key=f"bilibili:BV1TEST-{title}",
                source_platform="bilibili",
                source_strategy="search",
                content_id=f"BV1TEST-{title}",
                title=title,
                description="一个通用的测试候选描述",
            )
        ]
    )
    row = db.conn.execute(
        "SELECT id FROM discovery_candidates ORDER BY id DESC LIMIT 1"
    ).fetchone()
    candidate_id = int(row["id"])
    # Simulate a durable teacher-labeled row: cached, LLM-judged, historical.
    db.conn.execute(
        """
        UPDATE discovery_candidates
        SET status = 'cached', score_source = 'llm', llm_score_raw = 0.66,
            relevance_score = 0.66,
            temporal_class = 'historical', temporal_confidence = 0.9,
            temporal_reason = '对过去事件的回顾', temporal_policy_version = 'v2',
            temporal_validity_mode = 'none', temporal_valid_until = '',
            temporal_scope = 'none', temporal_state = 'unknown',
            temporal_evidence = '', temporal_next_review_at = '',
            temporal_evaluated_at = '2026-08-01T00:00:00Z',
            temporal_evidence_complete = 1
        WHERE id = ?
        """,
        (candidate_id,),
    )
    db.conn.commit()
    return candidate_id


def _incoming(**overrides: Any) -> dict[str, Any]:
    # A valid v2 versioned result: freshness_only + hook + verbatim evidence.
    fields: dict[str, Any] = {
        "candidate_id": 0,
        "temporal_class": "versioned",
        "temporal_confidence": 0.72,
        "temporal_reason": "指涉仍在迭代的硬件产品",
        "temporal_policy_version": "v2",
        "temporal_validity_mode": "freshness_only",
        "temporal_valid_until": "",
        "temporal_scope": "hook",
        "temporal_state": "unknown",
        "temporal_evidence": "候选标题",
        "temporal_next_review_at": "",
        "temporal_evaluated_at": _RELABEL_EVALUATED_AT,
        "temporal_evidence_complete": True,
    }
    fields.update(overrides)
    return fields


def _row(db: Database, candidate_id: int) -> dict[str, Any]:
    return dict(
        db.conn.execute(
            "SELECT * FROM discovery_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
    )


def test_relabel_replaces_classified_temporal_annotation_and_touches_only_temporal(
    db: Database,
) -> None:
    candidate_id = _seed_candidate(db)
    before = _row(db, candidate_id)

    updated = db.update_discovery_candidate_temporal_relabel(
        [_incoming(candidate_id=candidate_id)]
    )
    assert updated == 1
    after = _row(db, candidate_id)

    assert after["temporal_class"] == "versioned"
    assert after["temporal_confidence"] == 0.72
    assert after["temporal_policy_version"] == "v2"
    # Non-temporal columns are untouched.
    assert after["status"] == before["status"] == "cached"
    assert after["relevance_score"] == before["relevance_score"] == 0.66
    assert after["llm_score_raw"] == before["llm_score_raw"] == 0.66
    assert after["score_source"] == before["score_source"] == "llm"
    assert after["title"] == before["title"]


def test_relabel_preserves_old_annotation_for_unknown_incoming(db: Database) -> None:
    candidate_id = _seed_candidate(db)

    updated = db.update_discovery_candidate_temporal_relabel(
        [_incoming(candidate_id=candidate_id, temporal_class="unknown")]
    )
    assert updated == 1
    after = _row(db, candidate_id)
    # The neutral re-label must not erase the older classification.
    assert after["temporal_class"] == "historical"


def test_relabel_preserves_old_annotation_for_low_confidence_incoming(
    db: Database,
) -> None:
    candidate_id = _seed_candidate(db)

    updated = db.update_discovery_candidate_temporal_relabel(
        [_incoming(candidate_id=candidate_id, temporal_confidence=0.4)]
    )
    assert updated == 1
    after = _row(db, candidate_id)
    assert after["temporal_class"] == "historical"


def test_relabel_preserves_old_annotation_for_invalid_class_mode_pair(
    db: Database,
) -> None:
    candidate_id = _seed_candidate(db)

    # versioned + mode=none is an invalid v2 pair -> neutralized -> keep old.
    updated = db.update_discovery_candidate_temporal_relabel(
        [
            _incoming(
                candidate_id=candidate_id,
                temporal_validity_mode="none",
                temporal_scope="none",
                temporal_evidence="",
            )
        ]
    )
    assert updated == 1
    after = _row(db, candidate_id)
    assert after["temporal_class"] == "historical"


def test_relabel_updates_real_utc_evaluated_at_and_policy_version(db: Database) -> None:
    candidate_id = _seed_candidate(db)

    db.update_discovery_candidate_temporal_relabel(
        [_incoming(candidate_id=candidate_id)]
    )
    after = _row(db, candidate_id)
    assert after["temporal_evaluated_at"] == _RELABEL_EVALUATED_AT
    assert after["temporal_evidence_complete"] == 1
    assert after["temporal_policy_version"] == "v2"


def test_relabel_skips_unknown_candidate_id(db: Database) -> None:
    updated = db.update_discovery_candidate_temporal_relabel(
        [_incoming(candidate_id=0)]
    )
    assert updated == 0
