# Eval Context Snapshot — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans (execute this plan task-by-task).
> **Spec:** [`2026-08-19-ml-eval-context-snapshot-spec.md`](./2026-08-19-ml-eval-context-snapshot-spec.md)
> **Status:** Wave A implemented in `feat/ml-ranking-wave1` (helper, schema, engine bind/stamp, pipeline persist, export, probe grouping). Docs-first commit was skipped because this work landed on an already-dirty Wave 1 checkout; do not mix a docs-only commit into uncommitted inference/probe code unless the user asks.
> **Execution order:** Task 1 → 2 → 3 → 4 → 5 → 6 (docs). No live 90-row LLM probe in this slice.
> **Tech:** Python 3.11, `uv run pytest <file> -q`, `uv run ruff check src/ tests/ scripts/ml_teacher_self_consistency_probe.py`, `uv run ruff format src tests scripts`, `uv run mypy src/`

**Invariants that MUST hold — re-read before each task:**

- `profile_digest` is the same `stable_json_digest({"summary", "recall_pool"})` the eval cache already uses.
- Snapshots store the compact eval slice only; never `personality_portrait`.
- Only teacher-allowlist `score_source` values receive digests; synthetic paths stay `''`.
- Legacy empty digest means unknown. Do not backfill today's profile onto old scores.
- Snapshot upsert is fail-open.
- Mixed `(profile_digest, negative_digest)` pairs never share one eval batch.
- Popup / desktop web / mobile web / CLI recommend UIs are out of scope.

### Task 1: Helper module + digest round-trip

**Files:** `src/openbiliclaw/discovery/eval_context.py`; `tests/test_eval_context.py`.

**Interfaces:** Consumes: compact summary, recall pool, negative example dicts. Produces: `EvaluationContextSnapshot` with `digests_match()`.

**Steps:**

- [x] Write failing tests for payload-shape digest equality (tuple vs list) and corrupt mismatch.
- [x] Add `compute_profile_digest` / `compute_negative_digest` / JSON helpers.
- [x] Confirm JSON round-trip still hashes.

**Acceptance:**

- Numeric gate: tuple and list recall pools produce an identical 24-char digest.
- Reproduce with `uv run pytest tests/test_eval_context.py -q`.

### Task 2: Schema + persist API

**Files:** `src/openbiliclaw/storage/database.py`; `tests/test_eval_context.py`; `tests/test_discovery_score_provenance.py`.

**Interfaces:** Consumes: snapshot + evaluation dict digest keys. Produces: upsert/get; teacher SELECT includes digest columns.

**Steps:**

- [x] Additive `profile_digest` / `negative_digest` on `discovery_candidates` (`DEFAULT ''`).
- [x] `evaluation_context_snapshots` PK `(profile_digest, negative_digest)`.
- [x] Reject empty/mismatched upserts; get returns None on hash mismatch.
- [x] Persist round-trip test next to `test_evaluation_persist_round_trips_teacher_model`.

**Acceptance:**

- Numeric gate: 1 upsert + get with `digests_match()`; corrupt JSON → `None`; omitted eval keys persist as `''`.
- Reproduce with `uv run pytest tests/test_eval_context.py tests/test_discovery_score_provenance.py::test_evaluation_persist_round_trips_profile_and_negative_digest -q`.

### Task 3: Engine bind / stamp

**Files:** `src/openbiliclaw/discovery/engine.py`; `tests/test_discovery_score_provenance.py`.

**Interfaces:** Consumes: live `SoulProfile` or bound `EvaluationContextSnapshot`. Produces: stamped in-memory digests + fail-open snapshot upsert.

**Steps:**

- [x] `_bind_evaluation_context` / `_unbind_evaluation_context` around single and batch eval.
- [x] Override wins for summary, digest, recall pool, and negatives.
- [x] `_build_live_evaluation_context` captures negatives through `_get_negative_exemplars` while override is unset (replay/test stubs still freeze).
- [x] Stamp after intra-batch caps; non-teacher paths stay empty.
- [x] Snapshot write fail-open.

**Acceptance:**

- Numeric gate: mocked batch of 1 LLM item → non-empty matching digests + snapshot row; viewed/prefilter digests stay `''`; mutated live profile does not appear in a bound snapshot prompt.
- Reproduce with `uv run pytest tests/test_discovery_score_provenance.py -q`.

### Task 4: Pipeline persist + export columns

**Files:** `src/openbiliclaw/discovery/candidate_pipeline.py`; `scripts/export_ranking_dataset.py`.

**Interfaces:** Consumes: `DiscoveredContent.profile_digest` / `negative_digest`. Produces: DB columns and JSONL fields (skipped if the live schema lacks them).

**Steps:**

- [x] Pass digest fields into `persist_claimed_discovery_candidate_evaluations`.
- [x] Add columns to `CANDIDATE_COLUMNS` and the export record.

**Acceptance:**

- Numeric gate: persist dict includes both keys; export script still runs on a fixture DB missing the columns (filter-to-existing).
- Reproduce with `uv run pytest tests/test_export_ranking_dataset.py tests/test_discovery_score_provenance.py::test_evaluation_persist_round_trips_profile_and_negative_digest -q`.

### Task 5: Probe grouping / override / `--require-snapshot`

**Files:** `scripts/ml_teacher_self_consistency_probe.py`; `tests/test_ml_teacher_self_consistency_probe.py`.

**Interfaces:** Consumes: teacher rows with digest columns. Produces: grouped replay; snapshot bind; legacy live fallback.

**Steps:**

- [x] SELECT digest columns; group by pair; resolve snapshot.
- [x] Bind override for the group; do not persist candidate scores or backfill digests.
- [x] `--require-snapshot` skips legacy/missing rows.
- [x] Unit tests for grouping and require-snapshot selection.

**Acceptance:**

- Numeric gate: mixed digest sample splits into distinct groups; require-snapshot keeps only rows with a verified snapshot.
- Reproduce with `uv run pytest tests/test_ml_teacher_self_consistency_probe.py -q`.

### Task 6: Docs

**Files:** module docs, changelog, architecture/spec SQLite line, probe writeup, this pair.

**Steps:**

- [x] Write this spec/plan pair.
- [x] Sync `docs/modules/discovery.md`, `storage.md`, `ml.md`, `docs/changelog.md`, `docs/architecture.md`, `docs/spec.md`, probe notes, ranking plan pointer.

**Acceptance:**

- Numeric gate: none (doc sync). Four-surface exclusion stated. No config/CLI/README highlights.

## Verification after merge

Observe new teacher rows in the live DB after the Wave 1 daemon runs this code:
`SELECT COUNT(*) FROM discovery_candidates WHERE profile_digest != ''` and
`SELECT COUNT(*) FROM evaluation_context_snapshots`. Rollback is additive-column
compatible (empty digest = legacy). Do not enable `relevance_scorer=ml` from this
slice. Do not run a live 90-row LLM probe unless asked.

## Explicitly out of scope

- Step 3 relative cosine features / GroupKFold by digest
- Step 5 stable/volatile teacher prompt split
- Skipping `evaluate_batch` for predicted y=0
- Recalibrating C1–C7
- Enabling live `shadow`/`ml`
- Mixing `openai_compatible` labels
- Committing / pushing unless the user asks
