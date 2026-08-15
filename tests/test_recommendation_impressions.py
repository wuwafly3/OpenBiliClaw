"""Wave 0 impression-ledger tests for the ML ranking work.

The ledger is the (exposure, no-interaction) negative-sample source for
supervised ranking. These tests pin the two properties that make it usable as
training data — repeat exposure must not inflate row count, and logging must not
disturb the unread/notification semantics that ``recommendations.presented``
owns — plus the four-surface coverage contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbiliclaw.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path


def _db(tmp_path: Path, name: str = "impressions.db") -> Database:
    db = Database(tmp_path / name)
    db.initialize()
    return db


def _seed_recommendation(db: Database, bvid: str, *, confidence: float = 0.8) -> int:
    db.cache_content(bvid, title=f"title-{bvid}", source="search", relevance_score=confidence)
    return db.insert_recommendation(bvid, confidence=confidence, expression="文案", topic="主题")


class TestImpressionLedger:
    def test_records_one_row_per_recommendation_and_surface(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        first = _seed_recommendation(db, "BV1")
        second = _seed_recommendation(db, "BV2")

        db.record_recommendation_impressions(
            [
                {"recommendation_id": first, "surface": "extension", "position": 0},
                {"recommendation_id": second, "surface": "extension", "position": 1},
            ]
        )

        assert db.count_recommendation_impressions() == 2
        db.close()

    def test_repeat_exposure_bumps_count_without_new_rows(self, tmp_path: Path) -> None:
        """A polling client must not be able to inflate the sample count."""
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1")
        payload = [{"recommendation_id": rec_id, "surface": "extension", "position": 3}]

        db.record_recommendation_impressions(payload)
        db.record_recommendation_impressions(payload)
        db.record_recommendation_impressions(payload)

        assert db.count_recommendation_impressions() == 1
        row = db.conn.execute(
            "SELECT impression_count, position FROM recommendation_impressions"
        ).fetchone()
        assert row["impression_count"] == 3
        db.close()

    def test_keeps_best_observed_position(self, tmp_path: Path) -> None:
        """Rank is a training feature; a later worse rank must not erase the best."""
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1")

        for position in (7, 2, 9):
            db.record_recommendation_impressions(
                [{"recommendation_id": rec_id, "surface": "extension", "position": position}]
            )

        row = db.conn.execute("SELECT position FROM recommendation_impressions").fetchone()
        assert row["position"] == 2
        db.close()

    def test_same_card_on_two_surfaces_is_two_rows(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1")

        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "extension", "position": 0}]
        )
        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "mobile_web", "position": 0}]
        )

        assert db.count_recommendation_impressions() == 2
        surfaces = {
            str(row["surface"])
            for row in db.conn.execute("SELECT surface FROM recommendation_impressions")
        }
        assert surfaces == {"extension", "mobile_web"}
        db.close()

    @pytest.mark.parametrize("surface", ["extension", "desktop_web", "mobile_web", "cli"])
    def test_accepts_every_user_facing_surface(self, tmp_path: Path, surface: str) -> None:
        """Four-surface contract: each real surface must be a valid ledger writer."""
        db = _db(tmp_path, f"surface-{surface}.db")
        rec_id = _seed_recommendation(db, "BV1")

        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": surface, "position": 0}]
        )

        assert db.count_recommendation_impressions() == 1
        db.close()

    def test_rejects_unknown_surface_label(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1")

        with pytest.raises(ValueError, match="surface"):
            db.record_recommendation_impressions(
                [{"recommendation_id": rec_id, "surface": "telepathy", "position": 0}]
            )

        assert db.count_recommendation_impressions() == 0
        db.close()

    def test_rejects_non_positive_recommendation_id(self, tmp_path: Path) -> None:
        db = _db(tmp_path)

        with pytest.raises(ValueError, match="recommendation_id"):
            db.record_recommendation_impressions(
                [{"recommendation_id": 0, "surface": "extension", "position": 0}]
            )
        db.close()

    def test_score_at_exposure_is_readable_by_joining_recommendations(
        self,
        tmp_path: Path,
    ) -> None:
        """The ledger stores no score; the trainer joins the immutable one."""
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1", confidence=0.77)

        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "extension", "position": 0}]
        )

        row = db.conn.execute(
            """
            SELECT r.confidence
            FROM recommendation_impressions AS i
            JOIN recommendations AS r ON r.id = i.recommendation_id
            """
        ).fetchone()
        assert row["confidence"] == pytest.approx(0.77)
        db.close()

    def test_empty_payload_is_a_noop(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        assert db.record_recommendation_impressions([]) == 0
        assert db.count_recommendation_impressions() == 0
        db.close()


class TestImpressionLedgerIsolation:
    """The ledger must not disturb unread / notification semantics."""

    def test_logging_does_not_mark_rows_presented(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1")
        unread_before = db.count_unread_recommendations()

        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "extension", "position": 0}]
        )

        row = db.conn.execute(
            "SELECT presented, presented_at FROM recommendations WHERE id = ?",
            (rec_id,),
        ).fetchone()
        assert row["presented"] == 0
        assert row["presented_at"] is None
        assert db.count_unread_recommendations() == unread_before
        db.close()

    def test_logging_keeps_notification_candidate_eligible(self, tmp_path: Path) -> None:
        """Writing ``presented`` here would permanently silence proactive pushes."""
        db = _db(tmp_path)
        rec_id = _seed_recommendation(db, "BV1", confidence=0.95)
        assert db.get_notification_candidate(min_confidence=0.82) is not None

        db.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "extension", "position": 0}]
        )

        assert db.get_notification_candidate(min_confidence=0.82) is not None
        db.close()


class TestImpressionLedgerMigration:
    def test_created_on_pre_migration_database(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        db = Database(db_path)
        db.initialize()
        db.conn.execute("DROP TABLE recommendation_impressions")
        db.conn.commit()
        db.close()

        migrated = Database(db_path)
        migrated.initialize()

        assert migrated.count_recommendation_impressions() == 0
        rec_id = _seed_recommendation(migrated, "BV1")
        migrated.record_recommendation_impressions(
            [{"recommendation_id": rec_id, "surface": "cli", "position": 0}]
        )
        assert migrated.count_recommendation_impressions() == 1
        migrated.close()
