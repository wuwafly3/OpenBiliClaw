"""Read-only inventory of training-signal availability for the ML ranking work.

Usage:
    python scripts/ml_data_probe.py [path/to/openbiliclaw.db]

Opens the database read-only (SQLite URI mode) and prints row counts plus
label-distribution summaries relevant to supervised ranking. Writes nothing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def q(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return list(conn.execute(sql))
    except sqlite3.Error as exc:  # pragma: no cover - diagnostic script
        print(f"  ! query failed: {exc}")
        return []


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not path.exists():
        print(f"database not found: {path}")
        return 1
    conn = connect_readonly(path)

    section("tables and row counts")
    tables = [
        r["name"]
        for r in q(conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]
    for name in tables:
        rows = q(conn, f'SELECT COUNT(*) AS n FROM "{name}"')
        n = rows[0]["n"] if rows else "?"
        print(f"  {name:45s} {n}")

    section("content_cache: label-bearing columns")
    for sql, label in [
        ("SELECT COUNT(*) AS n FROM content_cache", "total rows"),
        (
            "SELECT COUNT(*) AS n FROM content_cache WHERE relevance_score > 0",
            "has relevance_score",
        ),
        (
            "SELECT COUNT(*) AS n FROM content_cache WHERE feedback_type IS NOT NULL AND feedback_type != ''",
            "has feedback_type",
        ),
        (
            "SELECT COUNT(*) AS n FROM content_cache WHERE recommended_at IS NOT NULL",
            "was recommended",
        ),
        (
            "SELECT COUNT(*) AS n FROM content_cache WHERE pool_status = 'fresh'",
            "pool_status=fresh",
        ),
    ]:
        rows = q(conn, sql)
        print(f"  {label:45s} {rows[0]['n'] if rows else '?'}")

    section("content_cache: feedback_type distribution")
    for r in q(
        conn,
        "SELECT COALESCE(NULLIF(feedback_type,''),'<none>') AS ft, COUNT(*) AS n "
        "FROM content_cache GROUP BY ft ORDER BY n DESC",
    ):
        print(f"  {r['ft']:45s} {r['n']}")

    section("content_cache: relevance_score histogram (0.05 buckets)")
    for r in q(
        conn,
        "SELECT CAST(relevance_score*20 AS INT)/20.0 AS bucket, COUNT(*) AS n "
        "FROM content_cache GROUP BY bucket ORDER BY bucket",
    ):
        print(f"  {r['bucket']:<10} {r['n']}")

    section("content_cache: platform x scored coverage")
    for r in q(
        conn,
        "SELECT source_platform, COUNT(*) AS n, "
        "SUM(CASE WHEN relevance_score>0 THEN 1 ELSE 0 END) AS scored, "
        "ROUND(AVG(NULLIF(relevance_score,0)),4) AS avg_score "
        "FROM content_cache GROUP BY source_platform ORDER BY n DESC",
    ):
        print(
            f"  {str(r['source_platform']):20s} n={r['n']:<7} scored={r['scored']:<7} avg={r['avg_score']}"
        )

    section("recommendations: feedback distribution")
    for r in q(
        conn,
        "SELECT COALESCE(NULLIF(feedback_type,''),'<none>') AS ft, COUNT(*) AS n, "
        "SUM(presented) AS presented FROM recommendations GROUP BY ft ORDER BY n DESC",
    ):
        print(f"  {r['ft']:30s} n={r['n']:<7} presented={r['presented']}")

    section("events: type x inferred_satisfaction")
    for r in q(
        conn,
        "SELECT event_type, COALESCE(inferred_satisfaction,'<null>') AS sat, COUNT(*) AS n "
        "FROM events GROUP BY event_type, sat ORDER BY n DESC LIMIT 40",
    ):
        print(f"  {r['event_type']:22s} {r['sat']:12s} {r['n']}")

    section("events: time span")
    for r in q(conn, "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM events"):
        print(f"  {r['lo']}  ->  {r['hi']}")

    section("discovery_candidates: status distribution")
    for r in q(
        conn,
        "SELECT status, COUNT(*) AS n FROM discovery_candidates GROUP BY status ORDER BY n DESC",
    ):
        print(f"  {r['status']:30s} {r['n']}")

    section("evaluator_prefilter_shadow_audit: llm_score join availability")
    for sql, label in [
        ("SELECT COUNT(*) AS n FROM evaluator_prefilter_shadow_audit", "total audit rows"),
        (
            "SELECT COUNT(*) AS n FROM evaluator_prefilter_shadow_audit WHERE llm_score IS NOT NULL",
            "with llm_score",
        ),
        (
            "SELECT COUNT(*) AS n FROM evaluator_prefilter_shadow_audit WHERE similarity IS NOT NULL",
            "with similarity",
        ),
    ]:
        rows = q(conn, sql)
        print(f"  {label:45s} {rows[0]['n'] if rows else '?'}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
