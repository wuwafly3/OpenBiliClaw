"""Diagnose the LLM-produced relevance_score on the live database.

Usage:
    python scripts/ml_llm_score_diag.py [path/to/openbiliclaw.db]

Read-only. Shows what the evaluator's score actually looks like so the ML
ranking work can see what it would be distilling:

  * the score histogram on the uncensored population
    (discovery_candidates, which keeps rejected rows) vs the admission-censored
    content_cache;
  * score behavior across source / platform / topic / temporal_class;
  * round score values that hint at default or template behavior;
  * extreme and boundary cases with title + author + reason, so we can judge
    whether the score matches the content a human would expect.

Writes nothing. No API calls, no profile reads beyond what the DB already
stores, no network.
"""

from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

# The default Windows console codec is GBK, which cannot encode the Unicode
# histogram bars or Chinese titles. Rebind stdout to UTF-8 so this diagnostic
# renders correctly from cmd/PowerShell.
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DEFAULT_DB = Path("E:/otherproject/OpenBiliClaw/data/openbiliclaw.db")

UNICODE_BAR = "█"
UNICODE_EMPTY = "·"


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return list(conn.execute(sql))
    except sqlite3.Error as exc:  # pragma: no cover - diagnostic
        print(f"  ! query failed: {exc}")
        return []


def _sql_escape(value: object) -> str:
    """Build a quoted SQL literal for a known-safe string value."""
    return "'" + str(value).replace("'", "''") + "'"


def histogram(conn: sqlite3.Connection, sql: str, *, max_bar: int = 40) -> list[sqlite3.Row]:
    rows = _rows(conn, sql)
    total = sum(int(row["n"]) for row in rows) or 1
    for row in rows:
        width = int(max_bar * int(row["n"]) / total) or (1 if row["n"] else 0)
        print(
            f"  {row['bucket']:<6} {int(row['n']):>5}  {UNICODE_BAR * width}{UNICODE_EMPTY * max(max_bar - width, 0)}"
        )
    return rows


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def show_cases(
    conn: sqlite3.Connection,
    *,
    where: str,
    label: str,
    limit: int = 8,
) -> None:
    print(f"--- {label} ---")
    rows = _rows(
        conn,
        f"""
        SELECT relevance_score, source, source_platform, topic_group, style_key,
               temporal_class, title, up_name, relevance_reason, duration,
               published_at, last_scored_at
        FROM content_cache
        WHERE {where}
        ORDER BY relevance_score DESC
        LIMIT {max(1, int(limit))}
        """,
    )
    for row in rows:
        print(
            f"  {row['relevance_score']:.2f} | {row['source_platform'] or '?':<12} | {row['source'] or '?':<22} | {row['topic_group'] or '?':<18} | {row['style_key'] or '?':<16} | {row['temporal_class'] or '?'}"
        )
        print(f"      title : {row['title']}")
        print(f"      up    : {row['up_name']}")
        if row["relevance_reason"]:
            print(f"      reason: {row['relevance_reason']}")
        print()


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not path.exists():
        print(f"database not found: {path}")
        return 1
    conn = connect_readonly(path)

    section("1. relevance_score on content_cache (admission-censored: 0.60 floor)")
    histogram(
        conn,
        "SELECT CAST(relevance_score*20 AS INT)/20.0 AS bucket, COUNT(*) AS n "
        "FROM content_cache GROUP BY bucket ORDER BY bucket",
    )
    stats = _rows(
        conn,
        "SELECT COUNT(*) AS n, ROUND(AVG(relevance_score),4) AS avg, "
        "ROUND(MIN(relevance_score),4) AS min, ROUND(MAX(relevance_score),4) AS max, "
        "ROUND(SUM(CASE WHEN relevance_score < 0.70 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) "
        "AS below_070 "
        "FROM content_cache",
    )
    row = stats[0]
    print(f"  n={row['n']}  avg={row['avg']}  min={row['min']}  max={row['max']}")

    section("2. relevance_score on discovery_candidates (uncensored, includes rejected)")
    histogram(
        conn,
        "SELECT CAST(relevance_score*10 AS INT)/10.0 AS bucket, COUNT(*) AS n "
        "FROM discovery_candidates WHERE relevance_score > 0 GROUP BY bucket ORDER BY bucket",
    )
    for r in _rows(
        conn,
        """
        SELECT status, COUNT(*) AS n, ROUND(MIN(relevance_score),3) AS lo,
               ROUND(AVG(relevance_score),3) AS avg, ROUND(MAX(relevance_score),3) AS hi
        FROM discovery_candidates WHERE relevance_score > 0
        GROUP BY status ORDER BY n DESC
        """,
    ):
        print(f"  {r['status']:28s} n={r['n']:<5} range=[{r['lo']}, {r['hi']}] avg={r['avg']}")

    section("3. score digit patterns (evaluator digit preference / quantization)")
    rounds = _rows(
        conn,
        """
        SELECT CAST(relevance_score * 100 AS INTEGER) AS cents, COUNT(*) AS n
        FROM content_cache
        WHERE relevance_score > 0
        GROUP BY cents ORDER BY n DESC
        """,
    )
    for r in rounds[:12]:
        print(f"  {r['cents'] / 100:.2f}  x{r['n']}")
    top5 = sum(int(r["n"]) for r in rounds[:5])
    print(f"  (top-5 values cover {top5} of {sum(int(r['n']) for r in rounds)} rows)")

    section("4. score by source_strategy")
    for r in _rows(
        conn,
        """
        SELECT source, COUNT(*) AS n, ROUND(AVG(relevance_score),4) AS avg,
               ROUND(MIN(relevance_score),3) AS lo, ROUND(MAX(relevance_score),3) AS hi
        FROM content_cache GROUP BY source ORDER BY n DESC
        """,
    ):
        print(
            f"  {r['source'] or '?':<22} n={r['n']:<5} avg={r['avg']} range=[{r['lo']}, {r['hi']}]"
        )

    section("5. score by platform")
    for r in _rows(
        conn,
        """
        SELECT source_platform, COUNT(*) AS n, ROUND(AVG(relevance_score),4) AS avg,
               ROUND(MIN(relevance_score),3) AS lo, ROUND(MAX(relevance_score),3) AS hi
        FROM content_cache GROUP BY source_platform ORDER BY n DESC
        """,
    ):
        print(
            f"  {r['source_platform'] or '?':<14} n={r['n']:<5} avg={r['avg']} range=[{r['lo']}, {r['hi']}]"
        )

    section("6. score by temporal_class")
    for r in _rows(
        conn,
        """
        SELECT temporal_class, COUNT(*) AS n, ROUND(AVG(relevance_score),4) AS avg,
               ROUND(MIN(relevance_score),3) AS lo, ROUND(MAX(relevance_score),3) AS hi
        FROM content_cache GROUP BY temporal_class ORDER BY n DESC
        """,
    ):
        print(
            f"  {r['temporal_class'] or '?':<14} n={r['n']:<5} avg={r['avg']} range=[{r['lo']}, {r['hi']}]"
        )

    section("7. top topic_groups by average score")
    for r in _rows(
        conn,
        """
        SELECT topic_group, COUNT(*) AS n, ROUND(AVG(relevance_score),4) AS avg
        FROM content_cache WHERE topic_group != '' GROUP BY topic_group
        ORDER BY n DESC LIMIT 15
        """,
    ):
        print(f"  {r['topic_group'][:30]:<32} n={r['n']:<4} avg={r['avg']}")

    section("8. cases: top of the pool (score >= 0.90)")
    show_cases(conn, where="relevance_score >= 0.90", label="high scorers")

    section("9. cases: score in the 0.60-0.65 band (barely admitted)")
    show_cases(conn, where="relevance_score BETWEEN 0.60 AND 0.65", label="borderline admitted")

    section("10. cases: the rejected (uncensored) population")
    for r in _rows(
        conn,
        """
        SELECT relevance_score, source_strategy, source_platform, topic_group,
               style_key, temporal_class, title, author_name, eval_error
        FROM discovery_candidates
        WHERE status = 'rejected_low_score' AND relevance_score > 0
        ORDER BY relevance_score DESC LIMIT 6
        """,
    ):
        print(
            f"  {r['relevance_score']:.2f} | {r['source_platform'] or '?':<12} | {r['source_strategy'] or '?':<22} | {r['topic_group'] or '?':<18} | {r['style_key'] or '?':<16} | {r['temporal_class'] or '?'}"
        )
        print(f"      title : {r['title']}")
        print()

    section("11. like/dislike feedback rows: what did the LLM score them?")
    for r in _rows(
        conn,
        """
        SELECT feedback_type, relevance_score, title, up_name, source_platform
        FROM content_cache
        WHERE feedback_type IN ('like', 'dislike')
        ORDER BY feedback_type, relevance_score DESC
        """,
    ):
        print(
            f"  {r['feedback_type']:8s} {r['relevance_score']:.2f} | {r['source_platform'] or '?':<12} | {r['title']} | {r['up_name']}"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
