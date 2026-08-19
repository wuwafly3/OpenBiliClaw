# Wave 1 Tags Channel — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans (execute this plan task-by-task).
> **Spec:** [`2026-08-19-ml-wave1-tags-channel-spec.md`](./2026-08-19-ml-wave1-tags-channel-spec.md)
> **Status:** executable; parent Wave 1 remains [`2026-08-15-ml-ranking-plan.md`](./2026-08-15-ml-ranking-plan.md) item 0
> **Execution order:** Task 1 → 2 → 3 → 4 → 5 → 6 (docs). Task 7 (live probe) is measurement-only.
> **Tech:** Python 3.11, `uv run pytest <file> -q`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`

**Invariants that MUST hold — re-read before each task:**

- System prompt of the tags builder is a byte-static module constant; per-call data stays in the user message.
- Cheap tags write only `tag_channel_*` columns; teacher `topic_group` / `style_key` / `temporal_*` / `score_source` / `llm_score_raw` are untouched.
- The channel never writes a relevance score or changes admission.
- `[discovery].tag_channel_mode` defaults to `off`; illegal values reject at save time.
- Invalid enums coerce to `""` / `"unknown"` with WARNING, never persist placeholders.
- Popup / desktop web / mobile web / CLI recommend UIs are out of scope.

### Task 1: Static tags prompt builder + parse helpers

**Files:** Add `tests/test_llm_prompts.py` inputs; modify `src/openbiliclaw/llm/prompts.py`; add `tests/test_tag_channel.py` (parser cases).

**Interfaces:** Consumes: candidate dicts (title/description/body/platform/type). Produces: `(system, user)` messages; parsed `{bvid, topic_group, style_key, temporal_class}`.

**Steps:**

- [ ] Write a failing test that two different candidate batches yield an identical system message from `build_batch_tag_prompt`, and that the system text contains no profile / score / franchise language.
- [ ] Add `(build_batch_tag_prompt, args1, args2)` to `_builder_test_inputs()` in `tests/test_llm_prompts.py`.
- [ ] Run `uv run pytest tests/test_llm_prompts.py::test_prompt_builder_system_messages_are_call_invariant tests/test_tag_channel.py -q` and confirm FAIL for the missing builder.
- [ ] Add `_BATCH_TAG_SYSTEM_PROMPT` + `build_batch_tag_prompt` + `parse_tag_channel_payload` (validate with `normalize_style_key` / `normalize_temporal_class`).
- [ ] Rerun the focused tests and confirm PASS.
- [ ] `uv run ruff check src/openbiliclaw/llm/prompts.py tests/test_llm_prompts.py tests/test_tag_channel.py`

**Acceptance:**

- Numeric gate: system message equality across two batches (byte-identical).
- Reproduce with `uv run pytest tests/test_llm_prompts.py::test_prompt_builder_system_messages_are_call_invariant tests/test_tag_channel.py -q`.

### Task 2: Schema + persist API

**Files:** `src/openbiliclaw/storage/database.py` (CREATE + `_ensure_discovery_candidate_columns` + `update_discovery_candidate_tag_channel`); `tests/test_storage.py`.

**Interfaces:** Consumes: list of `{candidate_id, tag_channel_*}`. Produces: updated row count; teacher columns unchanged.

**Steps:**

- [ ] Write a failing test: seed a teacher-labeled row, write cheap tags, assert teacher `topic_group`/`style_key`/`temporal_class`/`score_source`/`llm_score_raw` unchanged and `tag_channel_*` populated.
- [ ] Run `uv run pytest tests/test_storage.py::TestDatabase::test_update_discovery_candidate_tag_channel_does_not_touch_teacher_columns -q` and confirm FAIL.
- [ ] Add columns + `_ensure_*` + persist method that SETs only `tag_channel_*`.
- [ ] Rerun the test PASS.
- [ ] `uv run ruff check src/openbiliclaw/storage/database.py tests/test_storage.py`

**Acceptance:**

- Numeric gate: 0 teacher-column diffs after a cheap-tag write on a fixture row.
- Reproduce with the focused pytest name above.

### Task 3: Engine `tag_content_batch`

**Files:** `src/openbiliclaw/discovery/engine.py`; `src/openbiliclaw/llm/concurrency.py`; `src/openbiliclaw/llm/service.py` (priority map if needed); `tests/test_tag_channel.py`.

**Interfaces:** Consumes: `list[DiscoveredContent]`. Produces: tags + `tag_channel_model` from the response; caller `discovery.tag_batch`.

**Steps:**

- [ ] Write a failing test with a fake LLM that returns JSON tags and asserts caller name, `inject_core_memory` false, no `relevance_score` mutation, enum coercion.
- [ ] Run focused pytest; confirm FAIL.
- [ ] Implement `tag_content_batch`; register caller in `_EVALUATION_CALLERS`; add `discovery.tag` to route prefixes if missing.
- [ ] Never cache empty/failed tag results.
- [ ] Rerun PASS; ruff + mypy on touched files.

**Acceptance:**

- Numeric gate: mocked batch of 3 items → 3 tagged, 0 score changes; invalid style → `""`.
- Reproduce with `uv run pytest tests/test_tag_channel.py -q`.

### Task 4: Config `tag_channel_mode`

**Files:** `src/openbiliclaw/config.py`; `config.example.toml`; `src/openbiliclaw/api/app.py` (GET/PUT limits if needed); tests for config round-trip.

**Interfaces:** Consumes: TOML / PUT body. Produces: validated `off|shadow|enforce`; 422 otherwise.

**Steps:**

- [ ] Write a failing test that default is `off` and `"always"` raises/rejects.
- [ ] Add field + normalize + PUT validation mirroring `eval_prefilter_mode`.
- [ ] Wire through RuntimeContext (read-only for now; enforce hook is Task 5).
- [ ] Rerun config tests; ruff.

**Acceptance:**

- Numeric gate: illegal value rejected at save; default `off` on blank config.
- Reproduce with the new config test.

### Task 5: Enforce/shadow hook (default still off)

**Files:** `src/openbiliclaw/discovery/candidate_pipeline.py`; tests.

**Interfaces:** Consumes: `tag_channel_mode` from config. Produces: optional tag persist before `evaluate_claim` when `enforce`; no extra calls when `off`.

**Steps:**

- [ ] Write tests: `off` → tag LLM not called; `enforce` → tag called then eval still called; tag failure still evaluates.
- [ ] Implement skip-if-already-tagged (`tag_channel_source == 'llm'`).
- [ ] Rerun focused tests.

**Acceptance:**

- Numeric gate: `off` extra LLM calls = 0; `enforce` extra calls = 1 per untagged claim (mocked).
- Reproduce with pipeline tests.

### Task 6: Docs

**Files:** `docs/modules/discovery.md`, `docs/modules/config.md`, `docs/modules/storage.md`, `docs/changelog.md`.

**Steps:**

- [ ] Update implemented-features / public API / config table / changelog 未发布.
- [ ] No architecture diagram (no new module box yet).

**Acceptance:**

- Pre-merge docs checklist for config + storage + discovery + changelog.

### Task 7: Live cost/agreement probe (measurement, not merge-blocking)

**Files:** `scripts/ml_tag_channel_probe.py`

**Steps:**

- [ ] Script selects teacher-allowlist rows, calls the channel with `--limit`, prints tokens/item and agreement.
- [ ] Run against `E:/otherproject/OpenBiliClaw/data/openbiliclaw.db` only when the user asks (live LLM spend).

**Acceptance:**

- Numeric gate for S1.8 rewrite later: record tokens/candidate vs `discovery.evaluate_batch`. No pass/fail on agreement for this PR.

## Verification after merge

Observe `openbiliclaw cost --by caller` on the live daemon: `discovery.tag_batch` must stay at 0 while mode is `off`. Rollback: set `tag_channel_mode = "off"` or revert the commit. Do not enable `enforce` on the live daemon until Task 7 numbers exist.

## Explicitly out of scope

- `src/openbiliclaw/ml/features.py` and training/inference
- `[discovery].relevance_scorer`
- Skipping `evaluate_batch` for predicted y=0
- Recalibrating C1–C7
- Ranking UI surfaces
- Changing the live daemon's current eval drain settings
