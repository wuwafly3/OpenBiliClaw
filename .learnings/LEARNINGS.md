# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260820-001] correction

**Logged**: 2026-08-20T18:55:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
Teacher labels collected on a daemon that is not running the snapshot writer cannot be used for snapshot-backed self-consistency; keep Wave 1 on the live `feat/ml-ranking` checkout instead of a side worktree.

### Details
A separate `OpenBiliClaw-ml-wave1` worktree held the eval-context snapshot code while the live writer stayed on older `feat/ml-ranking`. New teacher rows therefore had empty `profile_digest`, and a 90-row agreement rerun (¥2.84) still mixed profile drift. The user asked to merge Wave 1 into the live branch and delete the extra tree.

### Suggested Action
Ship writer-path instrumentation on the checkout that actually runs `serve-api`. Do not collect a labeling cohort against a daemon missing that code. Do not keep a second worktree for code that must stamp the live DB.

### Metadata
- Source: user_feedback
- Related Files: src/openbiliclaw/discovery/eval_context.py, scripts/ml_teacher_self_consistency_probe.py
- Tags: worktree, teacher-labels, eval-context-snapshot
