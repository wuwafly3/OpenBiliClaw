"""Measure the temporal_class x style_key joint distribution in teacher labels.

Tests the hypothesis that temporal classes correlate with viewing-mode styles
(breaking->quick_scan, story_immersion->historical, ...), which GLiNER2.5's
joint multi-task classification could exploit as a shared-representation prior.
"""
import sqlite3
from collections import defaultdict

DB = "file:E:/otherproject/OpenBiliClaw/data/openbiliclaw.db?mode=ro"

con = sqlite3.connect(DB, uri=True)
cur = con.cursor()

rows = cur.execute(
    """
    SELECT temporal_class, style_key, COUNT(*)
    FROM discovery_candidates
    WHERE score_source IN ('llm','cap_franchise','cap_style')
      AND llm_score_raw IS NOT NULL
      AND temporal_class IN ('breaking','current','versioned','evergreen','historical')
      AND style_key IS NOT NULL AND style_key != ''
    GROUP BY temporal_class, style_key
    """
).fetchall()

joint: dict[tuple[str, str], int] = {(t, s): n for t, s, n in rows}
temporal_total: dict[str, int] = defaultdict(int)
style_total: dict[str, int] = defaultdict(int)
for t, s, n in rows:
    temporal_total[t] += n
    style_total[s] += n

# Style ordering to keep the matrix readable.
styles = [
    "deep_focus", "quick_scan", "hands_on", "decision_support", "story_immersion",
    "opinion_sparring", "social_chat", "daily_wander", "mood_release",
    "aesthetic_browse", "ambient_companion", "live_pulse", "curiosity_spark",
]
temporals = ["breaking", "current", "versioned", "evergreen", "historical"]

print("=== temporal x style joint counts ===")
header = "          " + "".join(f"{s[:11]:>11}" for s in styles)
print(header)
for t in temporals:
    cells = "".join(f"{joint.get((t, s), 0):>11}" for s in styles)
    print(f"{t:10s}{cells}   (total={temporal_total[t]})")

print("\n=== style distribution WITHIN each temporal class (row %) ===")
for t in temporals:
    tot = temporal_total[t]
    if tot == 0:
        continue
    dist = sorted(
        ((joint.get((t, s), 0) / tot) * 100 for s in styles),
        reverse=True,
    )
    top = sorted(
        ((s, joint.get((t, s), 0) / tot * 100) for s in styles),
        key=lambda kv: kv[1],
        reverse=True,
    )[:4]
    top_str = ", ".join(f"{s}:{p:.0f}%" for s, p in top)
    print(f"{t:10s} n={tot:6d}  top styles: {top_str}")

print("\n=== temporal distribution WITHIN each style (column %) ===")
for s in styles:
    tot = style_total[s]
    if tot == 0:
        continue
    top = sorted(
        ((t, joint.get((t, s), 0) / tot * 100) for t in temporals),
        key=lambda kv: kv[1],
        reverse=True,
    )[:4]
    top_str = ", ".join(f"{t}:{p:.0f}%" for t, p in top)
    print(f"{s:16s} n={tot:6d}  top temporals: {top_str}")

# Chi-square-ish: strongest associations (pointwise mutual information proxy)
print("\n=== strongest (temporal, style) associations (count share of total) ===")
total = sum(rows_r[2] for rows_r in rows)
strong = sorted(
    ((t, s, n / total * 100) for (t, s), n in joint.items()),
    key=lambda kv: kv[2],
    reverse=True,
)[:15]
for t, s, p in strong:
    print(f"  {t:10s} x {s:16s} {p:5.1f}% of all")
con.close()
