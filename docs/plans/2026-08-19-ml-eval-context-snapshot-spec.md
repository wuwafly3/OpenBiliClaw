# Eval Context Snapshot Spec — freeze the compact teacher prompt at labeling time

**Created:** 2026-08-19
**Scope:** `discovery_candidates` digest columns, `evaluation_context_snapshots` table,
`ContentDiscoveryEngine` bind/stamp, pipeline persist, export columns, teacher
self-consistency probe replay
**Out of scope:** relative cosine features / GroupKFold by digest (step 3),
stable/volatile teacher prompt split (step 5), skipping `evaluate_batch` for
predicted y=0, C1–C7 recalibration, enabling live `shadow`/`ml`, mixing
`openai_compatible` labels, popup / desktop web / mobile web / CLI recommend
UIs (backend-only; exclusion is explicit)
**Parent:** [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md),
[`2026-08-19-ml-teacher-self-consistency-probe.md`](./2026-08-19-ml-teacher-self-consistency-probe.md)

## Goal

The 2026-08-19 teacher self-consistency probe measured **agreement 0.711
(64/90)** against the **current** `soul.json` and current negative exemplars.
That number mixes sampling noise with profile drift, so it understates the
labeling-time ceiling.

This spec freezes the compact eval-visible context at teacher labeling time
so a later probe can replay the same prompt slice. Target outcomes:

- Every new teacher-allowlist row (`score_source ∈ {llm, cap_franchise, cap_style}`)
  stamps `profile_digest` / `negative_digest` using the same hashes as the
  in-memory eval cache.
- The compact payload (summary + recall pool + negative titles the teacher
  saw) is upserted into `evaluation_context_snapshots` keyed by that pair.
- Historical rows stay `''` (unknown). The probe never backfills today's
  digest onto old scores.
- Mixed digest pairs never share one `evaluate_content_batch`.
- Snapshot write is fail-open: evaluation still returns scores if upsert throws.

Verification:

```text
uv run pytest tests/test_eval_context.py tests/test_discovery_score_provenance.py tests/test_ml_teacher_self_consistency_probe.py -q
uv run --extra dev python scripts/ml_teacher_self_consistency_probe.py --dry-run --limit 90
```

Live historical rows will report snapshot misses until the daemon runs this
code and new labels accumulate.

## Design invariants (MUST hold in every phase)

1. **Same digest as the eval cache:** `profile_digest` is
   `stable_json_digest({"summary": compact_summary, "recall_pool": payload})`
   via `openbiliclaw.llm.prompt_cache.stable_json_digest` (24-char hex).
   `negative_digest` is `stable_json_digest(examples or [])`. Changing hash
   shape requires a cache-namespace bump. Verified by
   `tests/test_eval_context.py::test_profile_digest_matches_engine_payload_shape`
   and `tests/test_discovery_engine.py` digest coverage tests.
2. **Compact slice only, never the portrait:** snapshots store
   `compact_content_prompt_profile_summary` + recall-pool tuples + negative
   exemplar dicts. `personality_portrait` is not persisted. Negatives may
   repeat disliked titles already in the local event log.
3. **Teacher-only stamp:** only `score_source in LLM_JUDGMENT_SCORE_SOURCES`
   get digests. `prefilter` / `viewed` / `truncated` / `eval_error` /
   `response_missing` stay `''`.
4. **Legacy empty means unknown:** missing columns migrate to `DEFAULT ''`.
   Empty digest is not today's profile. Probe grouping treats `('', '')` as
   one live-profile fallback cohort unless `--require-snapshot`.
5. **Fail-open persist:** `_remember_evaluation_context` logs WARNING and
   continues if upsert raises. Eval must not fail because the snapshot table
   is locked.
6. **No silent backfill:** the self-consistency probe does not UPDATE
   `discovery_candidates.relevance_score` / `llm_score_raw` / digest columns
   on old rows.
7. **Four-surface contract:** backend labeling + local SQLite only. Recommend
   UIs unchanged.

## Current diagnosis

### D1. Teacher rows cannot reconstruct t0 prompt

`Database.get_teacher_labeled_discovery_candidates()` exported score provenance
and `teacher_model` but not the compact profile the judge saw
(`storage/database.py` teacher SELECT). `profile_digest` already existed on
the eval cache key (`discovery/engine.py` `_evaluation_profile_digest`) and
on prefilter audit rows, but was not stamped onto `discovery_candidates`.
The 2026-08-19 probe therefore replayed current `soul.json`.

### D2. Compact summary churns independently of the onion file

`compact_content_prompt_profile_summary` includes recent awareness / insights
/ interest ranks. Through 2026-08-20 that slice drove most of the 2h/12
`profile_digest` churn. From 2026-08-21 the **gate** writer uses
`compact_gate_evaluation_profile_summary` (same caps, recent keys popped), so
new snapshots and digests no longer move when only awareness/insights/speculations
change. Historical snapshot JSON may still contain recent keys; replay those
bytes as stored. Recommendation / ranker compact is unchanged and still includes
recent. The snapshot remains the labeling-time view, not a freeze of `soul.json`.

### D3. Mixed contexts cannot share a batch

Batch eval sends one profile block. Rows labeled under digest A and digest B
in the same `evaluate_content_batch` would poison both cache keys and the
replay prompt. Probe grouping is by `(profile_digest, negative_digest)`.

## Priority classification

| Phase | Content | Tier | Why |
| --- | --- | --- | --- |
| 0 | Helper + schema + engine bind/stamp + pipeline persist | **MUST** | Without this, new labels keep drifting |
| 1 | Probe grouping / snapshot override / `--require-snapshot` | **MUST** | Otherwise the next ceiling measurement is still mixed |
| 2 | Export JSONL carries digest columns | RECOMMENDED | Unlocks later GroupKFold |
| 3 | Relative features / GroupKFold by digest | later | Step 3, out of this spec |
| 4 | Stable vs volatile teacher prompt split | later | Step 5, out of this spec |

Wave A (this document) ships 0–2 independently. Work may stop after Wave A:
new labels become replayable even if step 3 never lands.

## Phase designs

### Phase 0 — stamp and persist

- `openbiliclaw.discovery.eval_context.EvaluationContextSnapshot` is frozen;
  `digests_match()` must pass before upsert.
- JSON storage uses compact `sort_keys=True` separators so list/tuple recall
  pools rehash identically.
- `ContentDiscoveryEngine._bind_evaluation_context` freezes live compact
  summary + recall pool + negatives for one `evaluate_content` /
  `evaluate_content_batch`. An existing override (probe) is not replaced
  (`owns=False`).
- Stamp after intra-batch caps so `cap_*` rows keep the digest.
- `DiscoveryCandidatePipeline._persist_evaluations` writes both digest
  columns. Omitted keys persist as `''`.

### Phase 1 — probe replay

- Load digest columns; `get_evaluation_context_snapshot`.
- Group sample by `(profile_digest, negative_digest)`.
- If snapshot exists: set `engine._evaluation_context_override` before the
  call.
- If missing: warn and use current profile; `--require-snapshot` skips.
- Dummy `OnionProfile()` is enough when a snapshot is bound; production probe
  still loads soul.json as the live fallback for legacy rows.

## Expected impact

| Lever | Measured effect |
| --- | --- |
| New teacher labels | Digest + snapshot present; round-trip tests pass |
| Historical 1435 exact-identity rows | snapshot misses until the daemon runs this code |
| Next self-consistency probe | Can isolate sampling noise from profile drift on new rows |

## Documentation obligations

- `docs/modules/discovery.md` — eval_context + stamp/persist
- `docs/modules/storage.md` — table + getters
- `docs/modules/ml.md` — digest grouping / replay
- `docs/changelog.md` unpublished bullet
- `docs/architecture.md` + `docs/spec.md` SQLite line
- Update the 2026-08-19 self-consistency probe writeup
- No `config.md` / CLI / README highlights (no user-facing switch)
