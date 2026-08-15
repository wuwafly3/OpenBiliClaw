"""Wave 0 collection status for the ML ranking work.

Usage:
    python scripts/ml_impression_status.py [path/to/openbiliclaw.db]

Read-only. Reports progress against the two Wave 0 gates from
``docs/plans/2026-08-15-ml-ranking-spec.md``:

  * (exposure, no-interaction) negative samples >= 300
  * engagement positives >= 200 (the Wave 2 shadow gate)

It also prints per-surface coverage, because a ledger fed by only one of the
four surfaces would train a model on a biased slice of what the user actually
saw. Writes nothing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")

NEGATIVE_SAMPLE_GATE = 300
POSITIVE_SAMPLE_GATE = 200
SURFACES = ("extension", "desktop_web", "mobile_web", "cli", "unknown")


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def scalar(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error as exc:
        print(f"  ! {exc}")
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def bar(done: int, gate: int, width: int = 30) -> str:
    filled = min(width, int(width * done / gate)) if gate else width
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{gate}"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not path.exists():
        print(f"database not found: {path}")
        return 1
    conn = connect_readonly(path)

    ledger_exists = scalar(
        conn,
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name='recommendation_impressions'",
    )
    if not ledger_exists:
        print("recommendation_impressions table absent — start the daemon once to migrate.")
        conn.close()
        return 1

    section("impression ledger")
    total = scalar(conn, "SELECT COUNT(*) FROM recommendation_impressions")
    writes = scalar(
        conn, "SELECT COALESCE(SUM(impression_count), 0) FROM recommendation_impressions"
    )
    print(f"  distinct (recommendation, surface) rows      {total}")
    print(f"  total logged exposures                       {writes}")
    for row in conn.execute(
        "SELECT MIN(first_impression_at) lo, MAX(last_impression_at) hi "
        "FROM recommendation_impressions"
    ):
        print(f"  window                                       {row['lo']}  ->  {row['hi']}")

    section("per-surface coverage")
    counts = {
        str(row["surface"]): int(row["n"])
        for row in conn.execute(
            "SELECT surface, COUNT(*) AS n FROM recommendation_impressions GROUP BY surface"
        )
    }
    for surface in SURFACES:
        n = counts.get(surface, 0)
        flag = "" if n else "   <- no data yet"
        print(f"  {surface:14s} {n}{flag}")

    section("label construction")
    # A negative is an exposure that never drew any feedback. Interaction is
    # read from the recommendations row the ledger points at, so this is exactly
    # the join a trainer would do.
    negatives = scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM recommendation_impressions AS i
        JOIN recommendations AS r ON r.id = i.recommendation_id
        WHERE r.feedback_type IS NULL OR r.feedback_type = ''
        """,
    )
    positives = scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM recommendation_impressions AS i
        JOIN recommendations AS r ON r.id = i.recommendation_id
        WHERE r.feedback_type IN ('like', 'comment')
        """,
    )
    strong_negatives = scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM recommendation_impressions AS i
        JOIN recommendations AS r ON r.id = i.recommendation_id
        WHERE r.feedback_type = 'dislike'
        """,
    )
    print(f"  exposure, no interaction (negative)          {negatives}")
    print(f"  exposure + like/comment (positive)           {positives}")
    print(f"  exposure + dislike (strong negative)         {strong_negatives}")

    section("Wave 0 / Wave 2 gates")
    print(f"  W0 negatives >= {NEGATIVE_SAMPLE_GATE:<4} {bar(negatives, NEGATIVE_SAMPLE_GATE)}")
    print(f"  W2 positives >= {POSITIVE_SAMPLE_GATE:<4} {bar(positives, POSITIVE_SAMPLE_GATE)}")
    missing = [
        surface for surface in ("extension", "desktop_web", "mobile_web") if not counts.get(surface)
    ]
    if missing:
        print(f"  ! no exposures yet from: {', '.join(missing)}")
    if negatives < NEGATIVE_SAMPLE_GATE:
        print("  -> keep collecting; Wave 1 training set is not ready")
    else:
        print("  -> Wave 0 negative-sample gate met")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
