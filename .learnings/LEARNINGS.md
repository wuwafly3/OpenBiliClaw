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

## [LRN-20260821-001] insight

**Logged**: 2026-08-21T20:01:00+08:00
**Priority**: high
**Status**: resolved
**Area**: discovery

### Summary
The 13:07–13:11 awareness-window flip-flop was concurrent eval vs mid-write, not the 12h cognition beat. Dropping recent from the gate closed that digest hole; each candidate-eval worker still reloads `get_profile()`, so INTEREST/negatives can still split one drain tick.

### Details
`CandidateEvalLoop._evaluate_worker` calls `profile_provider()` per claim. ContextVar freeze is per `evaluate_content_batch`, not per tick. Cognition `soul_layer.data.clear(); save()` has no reader lock.

### Suggested Action
Implemented 8-21 spec D5 / plan Task 7: `_fill_open_slots` freezes one `EvaluationContextSnapshot` for every worker in that fill. Next fill may load a new profile. Soul-layer reader lock is still out of scope.

### Metadata
- Source: user_feedback
- Related Files: src/openbiliclaw/runtime/candidate_eval.py, src/openbiliclaw/discovery/engine.py, src/openbiliclaw/soul/cognition_cycle.py
- Tags: eval-context-snapshot, concurrency, profile-drift

## [LRN-20260821-002] correction

**Logged**: 2026-08-21T20:01:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: soul

### Summary
Default tab titles such as `哔哩哔哩 (゜-゜)つロ 干杯~-bilibili` in teacher negatives are collection noise and must be dropped from the exemplar list. List only complete website titles; do not denylist product names like `ChatGLM`.

### Details
Extension dislike events use `document.title`. Homepage/shell pages therefore poison `<negative_examples>`. Filter at `recent_negative_exemplars`; keep the event rows.

### Suggested Action
Implemented 8-21 spec D6 / plan Task 6: complete website titles only; the Bilibili cheers prefix matches by stripping the trailing site brand. `ChatGLM` is not a shell. `negative_digest` follows the filtered list.

### Metadata
- Source: user_feedback
- Related Files: src/openbiliclaw/soul/negative_exemplars.py, extension/src/content/bilibili.ts
- Tags: negative-exemplars, teacher-prompt, noise

