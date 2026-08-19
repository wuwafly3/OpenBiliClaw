# Wave 1 Tags Channel Spec — cheap topic/style/temporal without score or profile

**Created:** 2026-08-19
**Scope:** new tags-only LLM channel, `discovery_candidates` cheap-tag columns,
prompt-cache convention, cost caller, optional pipeline hook
**Out of scope:** `ml/features.py`, train/inference artifacts, `[discovery].relevance_scorer`,
skipping `evaluate_batch` for y=0, C1–C7 recalibration, `ranking_feature_log`,
UI / extension / mobile / CLI recommend surfaces (backend-only; exclusion is explicit)
**Parent:** [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md) S1.2a
**Evidence:** [`2026-08-16-ml-tags-ablation-probe.md`](./2026-08-16-ml-tags-ablation-probe.md)

## Goal

Wave 1 cannot run an ML admission model at inference time without tags:
the 2026-08-16 ablation showed teacher `topic_group` / `style_key` /
`temporal_class` are the main signal (Δρ +0.111, AUC 0.799 → 0.864). Those
fields today are a **side-product of the full scoring prompt**, so they do
not exist until `discovery.evaluate_batch` has already been paid.

This spec adds a cheaper channel that labels the same three fields **without**
a profile block, batch rubric, score, reason, franchise, or temporal v2
evidence group. Target outcomes:

- A distinct cost caller `discovery.tag_batch` visible in
  `openbiliclaw cost --by caller`.
- Cheap tags persist in **dedicated columns**; they never overwrite teacher
  `topic_group` / `style_key` / `temporal_*` / `score_source` / `llm_score_raw`.
- Default mode is `off`, so the live daemon does not double LLM spend.
- Unit-cost and teacher-agreement can be measured on the live allowlist with
  a read-mostly probe script.

Verification:

```bash
pytest tests/test_llm_prompts.py::test_prompt_builder_system_messages_are_call_invariant tests/test_tag_channel.py -q
openbiliclaw cost --by caller
python scripts/ml_tag_channel_probe.py E:/otherproject/OpenBiliClaw/data/openbiliclaw.db --limit 90
```

The probe is a measurement tool, not a merge blocker. Merge requires the
channel to exist, persist, stay default-off, and keep prompt-cache invariants.

## Design invariants (MUST hold in every phase)

1. **Prompt-cache convention:** `system_prompt` is a module-level byte-static
   constant. No f-string, no profile, no platform substitution in system.
   All per-call data lives in the user message, most-stable first.
   JSON uses `ensure_ascii=False, indent=2, sort_keys=True` unless the batch
   block deliberately uses canonical sparse JSON (same exception as
   `evaluate_batch`). Guarded by
   `tests/test_llm_prompts.py::test_prompt_builder_system_messages_are_call_invariant`.
2. **Column isolation:** cheap-channel writes only
   `tag_channel_*` columns (and their source/model/time stamps). Full
   `evaluate_batch` continues to own `topic_group` / `style_key` /
   `franchise_key` / `temporal_*` / `relevance_score` / `score_source` /
   `llm_score_raw` / `teacher_model`. A regression test must round-trip a
   row that has both teacher tags and cheap tags without either side
   clobbering the other.
3. **No score side effects:** the tags channel must not set
   `relevance_score`, `score_source`, `llm_score_raw`, admission status, or
   pool copy. Invalid / missing tags fail-soft to empty/`unknown` and log
   WARNING (hard-won rule 4).
4. **Default off:** `[discovery].tag_channel_mode` ∈ `{off, shadow, enforce}`,
   default `off`. `off` makes zero extra LLM calls. `shadow` tags a bounded
   sample of already-evaluated allowlist rows (or pending rows) without
   changing admission. `enforce` tags pending_eval before full eval still
   runs — it does **not** skip `evaluate_batch`. Skipping full eval is a
   later Wave 1 task gated on S1.5.
5. **Closed taxonomies:** `style_key` is 13-choose-1 via
   `normalize_style_key` (`discovery/style_keys.py`). `temporal_class` is
   six-choose-1 via `normalize_temporal_class` (`discovery/temporal.py`).
   Unknown style → `""`; unknown temporal → `"unknown"`. `topic_group` is
   free 2–4 character-word coarse label, trimmed, empty if missing.
6. **Four-surface contract:** this change is backend-only. Popup, desktop
   web, mobile web, and CLI recommendation UIs are unchanged and are
   declared excluded.

## Current diagnosis

### D1. Tags only exist after the expensive scoring call

`ContentDiscoveryEngine._evaluate_batch_once`
(`src/openbiliclaw/discovery/engine.py` ~3487–3637) calls
`build_batch_content_evaluation_prompt` with a full profile, then parses
`topic_group` / `style_key` / `franchise_key` / temporal v2 onto
`DiscoveredContent`. Cost caller is `discovery.evaluate_batch`
(`engine.py:3513,3529`). `max_tokens=4096`. There is no second builder
that emits tags without score.

Confirmed: `src/openbiliclaw/ml/` does not exist. Training artifacts are
scripts only (`scripts/ml_tags_ablation_probe.py`).

### D2. Teacher tags cannot be features at inference

S1.2 forbids using full-eval outputs as ML inputs at runtime because the
full eval has not happened yet. The ablation's v1 oracle is **what the
cheap channel is supposed to buy**. Meta-distillation (predicting tags
from other features) recovered only ~4% of the tags increment
(`docs/plans/2026-08-16-ml-multitask-probe.md`).

### D3. Writing cheap tags into teacher columns would poison Wave 1 labels

`Database.get_teacher_labeled_discovery_candidates`
(`storage/database.py:6776–6815`) exports `topic_group` / `style_key` /
`temporal_class` as aux labels. If the cheap channel overwrote those
columns, later S1.5 evaluation would mix two annotators. Dedicated
`tag_channel_*` columns are the isolation.

### D4. Live daemon must not silently double eval spend

The user daemon is on `E:\otherproject\OpenBiliClaw` with
`eval_min_batch_size=1` already draining pending_eval. A default-on tags
channel would add a second LLM call per candidate. Default `off` plus an
explicit probe script is the only safe rollout.

## Priority classification

| Phase | Content | Tier | Why |
| --- | --- | --- | --- |
| 0 | Static prompt builder + parse/validate + invariant test | **MUST** | Cache convention; closed taxonomies |
| 1 | `tag_channel_*` schema + persist API that cannot touch teacher columns | **MUST** | Label isolation (D3) |
| 2 | `ContentDiscoveryEngine.tag_content_batch` + caller `discovery.tag_batch` | **MUST** | Measurable cost channel |
| 3 | Config `tag_channel_mode` default `off`, round-trip, save-time reject of illegal values | **MUST** | D4; hard-won rule 7 |
| 4 | Probe script: unit cost + teacher agreement on allowlist sample | RECOMMENDED | Feeds S1.8 cost model rewrite |
| 5 | Pipeline hook for `enforce` (tag then still full-eval) | RECOMMENDED | Needed before ML inference; not needed to land 0–3 |

Phase 0–3 can ship without calling the live LLM. Phase 4 needs the live
DB + API key and is a measurement, not a code gate. Phase 5 must stay
off by default.

Later Wave 1 work (features.py, train, `relevance_scorer`, S1.5, skip
full eval for y=0) remains owned by the parent spec and is **out of
scope** here.

## Phase designs

### Phase 0 — Prompt and parser

- New builder `build_batch_tag_prompt` in `src/openbiliclaw/llm/prompts.py`.
- System constant `_BATCH_TAG_SYSTEM_PROMPT`: say only what to emit —
  JSON `results` with input `bvid`/`content_id`, `topic_group`,
  `style_key`, `temporal_class` and their taxonomies. Do not list
  score / reason / franchise / temporal-v2 as forbidden fields; the
  parser already drops extras.
- User message: candidate block only (title, description, optional
  truncated body, `source_platform`, `content_type`, duration,
  `published_at`). **No profile, no negatives, no source_context rubric.**
- Register in `_builder_test_inputs()`.
- Parser: reuse `extract_llm_json_list`; then `normalize_style_key` and
  `normalize_temporal_class`. Coerce invalid numbers/placeholders as
  missing. WARNING on coercions.

### Phase 1 — Schema

New columns on `discovery_candidates` (CREATE + `_ensure_discovery_candidate_columns`):

| Column | Type | Default |
| --- | --- | --- |
| `tag_channel_topic_group` | TEXT NOT NULL | `''` |
| `tag_channel_style_key` | TEXT NOT NULL | `''` |
| `tag_channel_temporal_class` | TEXT NOT NULL | `'unknown'` |
| `tag_channel_source` | TEXT NOT NULL | `''` (`llm` when written by this channel) |
| `tag_channel_model` | TEXT NOT NULL | `''` (provider/model, same shape as `teacher_model`) |
| `tag_channel_at` | TEXT NOT NULL | `''` (ISO timestamp) |

`update_discovery_candidate_tag_channel(rows)` updates **only** those
columns, keyed by `candidate_id`. Must not appear in
`update_discovery_candidate_evaluations`'s SET list.

### Phase 2 — Engine method

`ContentDiscoveryEngine.tag_content_batch(contents, *, batch_size=45) -> list[TagChannelResult]`

- Caller `discovery.tag_batch`
- `inject_core_memory=False`
- `json_mode=True`, `reasoning_effort=""`
- `max_tokens=1024` (tags-only; recalibrate if truncation appears — pitfall #3)
- Register in `llm/concurrency.py` `_EVALUATION_CALLERS` (same bucket as
  eval; it is maintenance scoring, not popup copy)
- Prefix `discovery.tag` already maps to evaluation bucket via
  `_ROUTE_BUCKET_PREFIXES` (`discovery.eval` is the existing prefix;
  add `discovery.tag` explicitly if longest-prefix would miss)
- Split-retry on missing members, same shape as `_evaluate_batch` but
  shallower `max_tokens`
- Cache: optional in-memory LRU keyed by content identity + prompt
  namespace; **never cache empty/failed results** (pitfall #2)

### Phase 3 — Config

`[discovery].tag_channel_mode = "off" | "shadow" | "enforce"`, default
`"off"`. Save-time reject of other strings (`PUT /api/config` 422).
GET/PUT, `config.example.toml`, `docs/modules/config.md`,
`docs/modules/discovery.md`, `docs/changelog.md` 未发布 bullet.

CLI / OpenClaw / RuntimeContext three assembly roots must load the field
(hard-won rule 5: shared backend config; UI surfaces excluded).

### Phase 4 — Probe

`scripts/ml_tag_channel_probe.py`: read-only select of teacher-allowlist
rows missing `tag_channel_source`, call the channel on a `--limit`
sample, persist cheap tags, print:

- tokens / candidate vs a recent `discovery.evaluate_batch` baseline from
  the usage ledger
- style exact-match rate vs teacher `style_key`
- temporal exact-match vs teacher `temporal_class`
- topic exact-match and a coarse note that topic is open vocabulary

Do not require agreement ≥ X to merge Phase 0–3. Record the numbers in
the PR when the probe is run against the live DB.

### Phase 5 — Enforce hook

When mode is `enforce`, `DiscoveryCandidatePipeline.evaluate_claim`
tags the claim items first (skip rows that already have
`tag_channel_source='llm'`), persists cheap tags, then runs the existing
full eval unchanged. Fail-open: tag-channel errors log WARNING and still
evaluate. `shadow` is a coordinator/backfill path, not in the claim
hot path.

## Expected impact

| Lever | Measured effect |
| --- | --- |
| Ablation (already measured) | Tags oracle Δρ +0.111, AUC +0.065 |
| This spec | Unlocks runtime features for later ML; unit cost TBD by probe |
| Live default | Zero extra spend (`off`) |

## Documentation obligations

- `docs/modules/discovery.md` — implemented-features table + public API
- `docs/modules/config.md` — `tag_channel_mode`
- `docs/modules/storage.md` — new columns
- `docs/changelog.md` — 未发布 bullet
- No architecture-diagram change (no new module box until `src/openbiliclaw/ml/` exists)
- No README highlights (not a user-facing release)
- CLI: only if a user-visible command is added; the probe script is
  `scripts/`, not a Typer command, unless we later add `openbiliclaw tag-probe`
