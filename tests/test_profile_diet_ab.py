"""Unit tests for the profile diet A/B replay helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_profile_diet_ab as replay_script
from scripts.run_profile_diet_ab import (
    _REPLAY_SOURCE_CONTEXT,
    ModelOverride,
    ReplayCandidate,
    ReplayEmbeddingAudit,
    ReplayEmbeddingValidationError,
    ReplayMetrics,
    ReplayPair,
    ReplayProfileSnapshot,
    ReplayRecallAudit,
    _build_engine,
    _DeterministicLLMService,
    _load_profile_snapshot,
    _print_report,
    _rows_to_contents,
    _score_contents,
    _write_artifact,
    admission_flip_summary,
    configured_topic_lifecycle_serialization,
    reason_off_prompts,
    relative_gate,
    replay_blocking_reasons,
    replay_call_attribution,
    run_scoped_embedding_audit,
    score_delta_summary,
    select_replay_rows,
    spearman_rank_correlation,
    validate_candidate_transport_experiment,
    validate_json_minify_transport,
    validate_reason_off_outputs,
    validate_replay_prefilter_compatibility,
    validate_replay_routes,
)

from openbiliclaw.discovery.engine import (
    ContentDiscoveryEngine,
    compact_evaluation_profile_summary,
    evaluation_profile_prompt_layers,
)
from openbiliclaw.discovery.eval_payload import (
    build_canonical_evaluation_batch,
    render_sparse_evaluation_json,
)
from openbiliclaw.discovery.strategies._utils import build_profile_summary
from openbiliclaw.llm.base import LLMRateLimitError, LLMResponse
from openbiliclaw.llm.evaluation_wire import encode_evaluation_row_wire
from openbiliclaw.llm.prompt_cache import PromptLayerRenderCache
from openbiliclaw.llm.prompts import build_batch_content_evaluation_prompt
from openbiliclaw.llm.service import LLMProviderExecutionError
from openbiliclaw.memory.manager import MemoryManager
from openbiliclaw.soul.overrides import ListEdit, ProfileOverrides
from openbiliclaw.soul.profile import InterestDomain, InterestTag, OnionProfile, SoulProfile
from openbiliclaw.soul.speculator import (
    SpeculativeInterest,
    SpeculativeState,
    save_speculative_state,
)


def test_score_delta_summary_reports_mean_and_nearest_rank_p95() -> None:
    summary = score_delta_summary([0.20, 0.60, 0.90, 0.40], [0.10, 0.65, 0.70, 0.40])

    assert summary.mean_abs_delta == pytest.approx(0.0875)
    assert summary.p95_abs_delta == pytest.approx(0.20)


def test_spearman_rank_correlation_handles_ordering_and_ties() -> None:
    assert spearman_rank_correlation([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(
        1.0
    )
    assert spearman_rank_correlation([0.1, 0.2, 0.3, 0.4], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(
        -1.0
    )
    assert spearman_rank_correlation([0.5, 0.5, 0.9], [0.4, 0.4, 0.8]) == pytest.approx(1.0)


def test_admission_flip_summary_uses_default_strategy_thresholds() -> None:
    candidates = [
        ReplayCandidate(candidate_id=1, title="search drops", source_strategy="search"),
        ReplayCandidate(candidate_id=2, title="explore rises", source_strategy="explore"),
        ReplayCandidate(candidate_id=3, title="unknown rises", source_strategy="custom"),
        ReplayCandidate(candidate_id=4, title="stable admitted", source_strategy="hot"),
    ]

    summary = admission_flip_summary(
        candidates,
        [0.61, 0.57, 0.59, 0.70],
        [0.59, 0.59, 0.61, 0.68],
    )

    assert summary.flip_count == 3
    assert summary.flip_rate == pytest.approx(0.75)
    assert summary.per_strategy == {"custom": 1, "explore": 1, "search": 1}


def test_select_replay_rows_filters_status_platform_and_orders_deterministically() -> None:
    rows = [
        {
            "id": 1,
            "status": "evaluated",
            "source_platform": "bilibili",
            "evaluated_at": "2026-07-04 10:00:00",
            "last_seen_at": "2026-07-04 10:00:00",
        },
        {
            "id": 2,
            "status": "cached",
            "source_platform": "xiaohongshu",
            "evaluated_at": "2026-07-04 11:00:00",
            "last_seen_at": "2026-07-04 11:00:00",
        },
        {
            "id": 3,
            "status": "evaluated",
            "source_platform": "bilibili",
            "evaluated_at": "2026-07-04 12:00:00",
            "last_seen_at": "2026-07-04 12:00:00",
        },
        {
            "id": 4,
            "status": "evaluated",
            "source_platform": "bilibili",
            "evaluated_at": "2026-07-04 12:00:00",
            "last_seen_at": "2026-07-04 12:00:00",
        },
        {
            "id": 5,
            "status": "pending_eval",
            "source_platform": "bilibili",
            "evaluated_at": "2026-07-05 12:00:00",
            "last_seen_at": "2026-07-05 12:00:00",
        },
        {
            "id": 6,
            "status": "rejected_low_score",
            "source_platform": "bilibili",
            "evaluated_at": "2026-07-04 13:00:00",
            "last_seen_at": "2026-07-04 13:00:00",
        },
    ]

    selected = select_replay_rows(rows, sample=4, platform="bilibili")

    assert [row["id"] for row in selected] == [6, 4, 3, 1]


def test_admission_flip_summary_uses_runtime_and_row_thresholds() -> None:
    candidates = [
        ReplayCandidate(
            candidate_id=1,
            title="custom floor",
            source_strategy="search",
            score_threshold=0.75,
        ),
        ReplayCandidate(candidate_id=2, title="global floor", source_strategy="search"),
    ]

    summary = admission_flip_summary(
        candidates,
        [0.74, 0.64],
        [0.76, 0.66],
        admission_min_score=0.65,
    )

    assert summary.flip_count == 2


def _pair(
    *,
    kind: str,
    repeat: int,
    flip_rate: float,
    spearman: float,
    admission_delta: float,
) -> ReplayPair:
    return ReplayPair(
        repeat=repeat,
        kind=kind,
        first_arm="A",
        scores_a=(0.6,),
        scores_b=(0.6,),
        metrics=ReplayMetrics(
            mean_abs_delta=0.0,
            p95_abs_delta=0.0,
            spearman=spearman,
            flip_rate=flip_rate,
            flip_count=round(flip_rate * 100),
            admitted_a=50,
            admitted_b=round(50 + admission_delta * 100),
            admission_rate_delta=admission_delta,
        ),
    )


def test_relative_gate_uses_repeated_control_envelope() -> None:
    controls = [
        _pair(kind="control", repeat=1, flip_rate=0.18, spearman=0.83, admission_delta=0.01),
        _pair(kind="control", repeat=2, flip_rate=0.21, spearman=0.80, admission_delta=-0.01),
        _pair(kind="control", repeat=3, flip_rate=0.19, spearman=0.82, admission_delta=0.00),
    ]
    treatments = [
        _pair(kind="treatment", repeat=1, flip_rate=0.17, spearman=0.84, admission_delta=0.01),
        _pair(kind="treatment", repeat=2, flip_rate=0.16, spearman=0.81, admission_delta=0.00),
        _pair(kind="treatment", repeat=3, flip_rate=0.18, spearman=0.82, admission_delta=0.02),
    ]

    passed, gate = relative_gate(controls, treatments)

    assert passed is True
    assert gate["control_flip_ceiling"] == pytest.approx(0.21)
    assert gate["control_spearman_floor"] == pytest.approx(0.80)


def test_relative_gate_rejects_admission_shrink() -> None:
    controls = [
        _pair(kind="control", repeat=index, flip_rate=0.02, spearman=0.98, admission_delta=0.0)
        for index in range(1, 4)
    ]
    treatments = [
        _pair(
            kind="treatment",
            repeat=index,
            flip_rate=0.02,
            spearman=0.98,
            admission_delta=-0.05,
        )
        for index in range(1, 4)
    ]

    passed, _gate = relative_gate(controls, treatments)

    assert passed is False


def _passing_replay_gate_inputs() -> dict[str, object]:
    return {
        "quality_passed": True,
        "route_audit": {"passed": True, "blocking_reasons": []},
        "embedding_audit": {"passed": True, "blocking_reasons": []},
        "recall_audit": {"passed": True, "blocking_reasons": []},
        "reason_output_audit": {"passed": True, "blocking_reasons": []},
        "prompt_transport_audit": {"passed": True, "blocking_reasons": []},
        "profile_snapshot_stable": True,
        "candidate_snapshot_stable": True,
    }


def test_replay_final_gate_accepts_only_complete_evidence() -> None:
    assert replay_blocking_reasons(**_passing_replay_gate_inputs()) == []  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("quality", "relative quality gate failed"),
        ("route", "route audit failed"),
        ("embedding", "embedding audit failed"),
        ("recall", "recall audit failed"),
        ("reason_output", "reason-output audit failed"),
        ("prompt_transport", "prompt-transport audit failed"),
        ("profile", "profile snapshot drifted"),
        ("candidate", "candidate snapshot drifted"),
    ],
)
def test_replay_final_gate_blocks_each_independent_failure(
    failure: str,
    expected_reason: str,
) -> None:
    inputs = _passing_replay_gate_inputs()
    if failure == "quality":
        inputs["quality_passed"] = False
    elif failure in {"route", "embedding", "recall", "reason_output", "prompt_transport"}:
        inputs[f"{failure}_audit"] = {"passed": False, "blocking_reasons": []}
    elif failure == "profile":
        inputs["profile_snapshot_stable"] = False
    elif failure == "candidate":
        inputs["candidate_snapshot_stable"] = False

    reasons = replay_blocking_reasons(**inputs)  # type: ignore[arg-type]

    assert expected_reason in " ".join(reasons)


def _json_minify_prompt_pair() -> tuple[dict[str, object], dict[str, object]]:
    profile = {
        "core_traits": ["PRIVATE TRAIT"],
        "interests": [{"name": "中文 兴趣", "weight": 0.9}],
        "current_phase": "PRIVATE PHASE",
    }
    content_items = [
        {
            "content_id": "PRIVATE-ID",
            "title": "PRIVATE TITLE",
            "body_text": "line one\nline two mentions <negative_examples> as data",
            "source_platform": "bilibili",
        }
    ]
    negative_examples = [{"title": "PRIVATE NEGATIVE", "reason": "quick_exit"}]

    def messages(*, compact: bool) -> list[dict[str, str]]:
        cache = PromptLayerRenderCache()
        return build_batch_content_evaluation_prompt(
            profile_summary=profile,
            profile_blocks=cache.render_json_layers(
                evaluation_profile_prompt_layers(profile),
                compact=compact,
            ),
            content_items=content_items,
            source_context="mixed",
            source_platform="bilibili",
            negative_examples=negative_examples,
            compact_json=compact,
        )

    pretty = messages(compact=False)
    compact = messages(compact=True)
    pretty_metadata = replay_script._prompt_transport_metadata(
        {
            "system_instruction": pretty[0]["content"],
            "user_input": pretty[1]["content"],
        },
        expected_compact_json=False,
    )
    compact_metadata = replay_script._prompt_transport_metadata(
        {
            "system_instruction": compact[0]["content"],
            "user_input": compact[1]["content"],
        },
        expected_compact_json=True,
    )
    return pretty_metadata, compact_metadata


def test_json_minify_prompt_metadata_proves_whitespace_only_and_is_private() -> None:
    pretty, compact = _json_minify_prompt_pair()

    assert pretty["system_digest"] == compact["system_digest"]
    assert pretty["prompt_semantic_digest"] == compact["prompt_semantic_digest"]
    assert pretty["prompt_digest"] != compact["prompt_digest"]
    assert int(compact["prompt_chars"]) < int(pretty["prompt_chars"])
    assert int(compact["prompt_bytes"]) < int(pretty["prompt_bytes"])
    assert pretty["all_target_json_pretty"] is True
    assert compact["all_target_json_compact"] is True
    assert int(compact["profile_json_block_count"]) >= 1
    assert compact["negative_examples_json_block_count"] == 1
    assert compact["content_batch_json_block_count"] == 1
    assert "PRIVATE" not in json.dumps([pretty, compact])


def test_json_minify_prompt_metadata_pairs_images_without_retaining_them() -> None:
    kwargs = {
        "system_instruction": "system",
        "user_input": '<content_batch>\n\n{"content_id": "safe"}\n\n</content_batch>',
        "image_inputs": [
            {
                "content_id": "PRIVATE IMAGE ID",
                "mime_type": "image/jpeg",
                "data_url": "data:image/jpeg;base64,PRIVATE IMAGE BYTES",
            }
        ],
    }

    first = replay_script._prompt_transport_metadata(
        kwargs,
        expected_compact_json=True,
    )
    same = replay_script._prompt_transport_metadata(
        kwargs,
        expected_compact_json=True,
    )
    changed = replay_script._prompt_transport_metadata(
        {
            **kwargs,
            "image_inputs": [
                {
                    "content_id": "PRIVATE IMAGE ID",
                    "mime_type": "image/jpeg",
                    "data_url": "data:image/jpeg;base64,DIFFERENT PRIVATE BYTES",
                }
            ],
        },
        expected_compact_json=True,
    )

    assert first["prompt_semantic_digest"] == same["prompt_semantic_digest"]
    assert first["prompt_semantic_digest"] != changed["prompt_semantic_digest"]
    assert first["image_input_count"] == 1
    assert "PRIVATE" not in json.dumps([first, same, changed])


def test_json_minify_prompt_metadata_rejects_malformed_tagged_json() -> None:
    with pytest.raises(RuntimeError, match="malformed replay JSON block"):
        replay_script._prompt_transport_metadata(
            {
                "system_instruction": "system",
                "user_input": "<content_batch>\n\n{not-json}\n\n</content_batch>",
            },
            expected_compact_json=True,
        )


def _candidate_transport_prompt_metadata() -> dict[str, dict[str, object]]:
    content_items = [
        {
            "content_id": "PRIVATE-GLOBAL-ID",
            "content_url": "https://private.invalid/item",
            "cover_url": "https://private.invalid/image",
            "title": "PRIVATE TITLE",
            "author_name": "PRIVATE AUTHOR",
            "body_text": "PRIVATE BODY\nwith tabs\tand slashes\\",
            "source_platform": "bilibili",
            "content_type": "video",
            "source_context": "mixed",
            "cover_image_ref": "cover:PRIVATE-GLOBAL-ID",
        }
    ]
    canonical = build_canonical_evaluation_batch(content_items)
    blocks = {
        "production-json": None,
        "sparse-json": render_sparse_evaluation_json(canonical),
        "row-wire-v1": encode_evaluation_row_wire(canonical.as_payload()),
    }
    metadata: dict[str, dict[str, object]] = {}
    for transport, candidate_block in blocks.items():
        local_ids = transport != "production-json"
        messages = build_batch_content_evaluation_prompt(
            profile_summary={"core_traits": ["PRIVATE PROFILE"]},
            content_items=content_items,
            source_context="mixed",
            source_platform="bilibili",
            candidate_block=candidate_block,
            local_result_ids=local_ids,
        )
        metadata[transport] = replay_script._prompt_transport_metadata(
            {
                "system_instruction": messages[0]["content"],
                "user_input": messages[1]["content"],
                "image_inputs": [
                    {
                        "content_id": "0" if local_ids else "PRIVATE-GLOBAL-ID",
                        "mime_type": "image/jpeg",
                        "data_url": "data:image/jpeg;base64,PRIVATE-IMAGE-BYTES",
                    }
                ],
            },
            expected_compact_json=False,
            expected_candidate_transport=transport,
            candidate_transport_audit_enabled=True,
        )
    return metadata


def test_candidate_transport_metadata_proves_canonical_and_image_equality_privately() -> None:
    metadata = _candidate_transport_prompt_metadata()
    production = metadata["production-json"]
    sparse = metadata["sparse-json"]
    row = metadata["row-wire-v1"]

    assert [metadata[name]["candidate_transport"] for name in metadata] == [
        "production-json",
        "sparse-json",
        "row-wire-v1",
    ]
    assert all(item["candidate_decode_valid"] is True for item in metadata.values())
    assert {item["candidate_canonical_digest"] for item in metadata.values()} == {
        production["candidate_canonical_digest"]
    }
    assert {item["user_context_digest"] for item in metadata.values()} == {
        production["user_context_digest"]
    }
    assert {item["image_payloads_digest"] for item in metadata.values()} == {
        production["image_payloads_digest"]
    }
    assert production["candidate_global_identity_field_count"] == 1
    assert production["candidate_url_field_count"] == 2
    for treatment in (sparse, row):
        assert treatment["candidate_local_id_coverage_complete"] is True
        assert treatment["candidate_global_identity_field_count"] == 0
        assert treatment["candidate_url_field_count"] == 0
        assert treatment["image_anchor_coverage_complete"] is True
    assert sparse["system_digest"] == row["system_digest"]
    assert "PRIVATE" not in json.dumps(metadata, ensure_ascii=False)


def test_candidate_transport_metadata_rejects_dangling_local_image_anchor() -> None:
    metadata = _candidate_transport_prompt_metadata()["sparse-json"]
    assert metadata["image_anchor_coverage_complete"] is True

    content_items = [
        {
            "content_id": "global-id",
            "title": "title",
            "author_name": "author",
            "source_platform": "bilibili",
            "content_type": "video",
            "cover_image_ref": "cover:global-id",
        }
    ]
    canonical = build_canonical_evaluation_batch(content_items)
    messages = build_batch_content_evaluation_prompt(
        profile_summary={},
        content_items=content_items,
        source_context="mixed",
        source_platform="bilibili",
        candidate_block=render_sparse_evaluation_json(canonical),
        local_result_ids=True,
    )
    mismatched = replay_script._prompt_transport_metadata(
        {
            "system_instruction": messages[0]["content"],
            "user_input": messages[1]["content"],
            "image_inputs": [
                {
                    "content_id": "1",
                    "mime_type": "image/jpeg",
                    "data_url": "data:image/jpeg;base64,safe",
                }
            ],
        },
        expected_compact_json=False,
        expected_candidate_transport="sparse-json",
        candidate_transport_audit_enabled=True,
    )

    assert mismatched["image_anchor_coverage_complete"] is False


def test_candidate_transport_metadata_accepts_reversed_prepared_image_order() -> None:
    content_items = [
        {
            "content_id": f"global-{index}",
            "title": f"title-{index}",
            "author_name": "author",
            "source_platform": "bilibili",
            "content_type": "video",
            "cover_image_ref": f"cover:global-{index}",
        }
        for index in range(2)
    ]
    canonical = build_canonical_evaluation_batch(content_items)
    messages = build_batch_content_evaluation_prompt(
        profile_summary={},
        content_items=content_items,
        source_context="mixed",
        source_platform="bilibili",
        candidate_block=render_sparse_evaluation_json(canonical),
        local_result_ids=True,
    )

    def metadata(order: tuple[int, int]) -> dict[str, object]:
        return replay_script._prompt_transport_metadata(
            {
                "system_instruction": messages[0]["content"],
                "user_input": messages[1]["content"],
                "image_inputs": [
                    {
                        "content_id": str(index),
                        "mime_type": "image/jpeg",
                        "data_url": f"data:image/jpeg;base64,image-{index}",
                    }
                    for index in order
                ],
            },
            expected_compact_json=False,
            expected_candidate_transport="sparse-json",
            candidate_transport_audit_enabled=True,
        )

    forward = metadata((0, 1))
    reversed_order = metadata((1, 0))

    assert forward["image_anchor_coverage_complete"] is True
    assert reversed_order["image_anchor_coverage_complete"] is True
    assert forward["image_payloads_digest"] != reversed_order["image_payloads_digest"]


def test_local_result_identity_metadata_never_position_binds_multi_member_errors() -> None:
    content_items = [
        {
            "content_id": f"global-{index}",
            "title": f"title-{index}",
            "author_name": "author",
            "source_platform": "bilibili",
            "content_type": "video",
        }
        for index in range(2)
    ]
    canonical = build_canonical_evaluation_batch(content_items)
    messages = build_batch_content_evaluation_prompt(
        profile_summary={},
        content_items=content_items,
        source_context="mixed",
        source_platform="bilibili",
        candidate_block=render_sparse_evaluation_json(canonical),
        local_result_ids=True,
    )
    context = replay_script._candidate_transport_context(
        messages[1]["content"],
        expected_transport="sparse-json",
    )
    response = SimpleNamespace(
        content=json.dumps(
            {
                "results": [
                    {"id": "0", "score": 0.8},
                    {"id": "0", "score": 0.7},
                    {"id": "unknown", "score": 0.6},
                    {"score": 0.5},
                ]
            }
        )
    )

    audit = replay_script._result_identity_metadata(
        response,
        context=context,
        expected_transport="sparse-json",
    )

    assert audit["result_identity_contract"] == "local-id"
    assert audit["result_local_id_binding_safe"] is True
    assert audit["result_duplicate_local_id_count"] == 1
    assert audit["result_unknown_local_id_count"] == 1
    assert audit["result_missing_local_id_count"] == 1

    mapped_global = replay_script._result_identity_metadata(
        SimpleNamespace(content=json.dumps({"PRIVATE-GLOBAL-ID": {"score": 0.9}})),
        context=context,
        expected_transport="sparse-json",
    )
    assert mapped_global["result_identity_contract"] == "global-id"
    assert mapped_global["result_global_identity_field_count"] == 1
    assert "PRIVATE" not in json.dumps(mapped_global)


def _json_minify_audit_calls(*, repeats: int = 3) -> list[dict[str, object]]:
    pretty, compact = _json_minify_prompt_pair()
    calls: list[dict[str, object]] = []

    def add_call(
        *,
        pair_kind: str,
        repeat: int,
        logical_run: str,
        arm: str,
    ) -> None:
        metadata = pretty if arm == "A" else compact
        if pair_kind == "treatment" and arm == "B":
            usage = {
                "input_tokens": 850,
                "output_tokens": 30,
                "total_tokens": 880,
                "cached_input_tokens": 85,
            }
        else:
            usage = {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "cached_input_tokens": 100,
            }
        classification_items = [
            {
                "candidate_key_digest": f"candidate-{index}",
                "position": index,
                "fields": {
                    field: {"digest": f"stable-{field}", "nonempty": True}
                    for field in replay_script._REPLAY_CLASSIFICATION_FIELDS
                },
            }
            for index in range(30)
        ]
        calls.append(
            {
                "pair_kind": pair_kind,
                "repeat": repeat,
                "logical_run": logical_run,
                "arm": arm,
                "method": "complete_structured_task",
                "caller": "discovery.evaluate_batch",
                "temperature": 0.0,
                "max_tokens": 4096,
                "request_kind": "root",
                "request_ordinal": 0,
                "request_candidate_count": 30,
                "cache_usage_semantics": "prompt_includes_cached",
                "cache_metric_supported": True,
                "status": "ok",
                "structured_item_count": 30,
                "classification_items": classification_items,
                "usage": usage,
                **metadata,
            }
        )

    for repeat in range(1, repeats + 1):
        add_call(pair_kind="control", repeat=repeat, logical_run="A1", arm="A")
        add_call(pair_kind="control", repeat=repeat, logical_run="A2", arm="A")
        add_call(pair_kind="treatment", repeat=repeat, logical_run="A", arm="A")
        add_call(pair_kind="treatment", repeat=repeat, logical_run="B", arm="B")
    return calls


def _passing_json_minify_audit(calls: list[dict[str, object]]) -> dict[str, object]:
    cache_stats = {
        "profile_core": {"digest": "digest-core", "hits": 11, "misses": 1},
        "profile_interests": {"digest": "digest-interests", "hits": 11, "misses": 1},
    }
    return validate_json_minify_transport(
        calls,
        enabled=True,
        repeats=3,
        arm_a_profile_cache_stats=cache_stats,
        arm_b_profile_cache_stats=cache_stats,
    )


def test_json_minify_transport_audit_passes_complete_evidence() -> None:
    audit = _passing_json_minify_audit(_json_minify_audit_calls())

    assert audit["passed"] is True
    token_gate = audit["token_gate"]
    assert token_gate["prompt_savings_median"] == pytest.approx(0.15)
    assert token_gate["total_savings_median"] == pytest.approx(0.20)
    treatment_a = audit["token_usage"]["treatment_comparison"]["A"]
    assert treatment_a["cached_input_tokens"] == 300
    assert treatment_a["uncached_input_tokens"] == 2700
    assert treatment_a["evaluated_item_count"] == 90
    assert treatment_a["evaluated_item_count_basis"] == "root_request_candidates"
    assert audit["repair"]["treatment_repair_call_delta_median"] == 0
    assert audit["classification"]["passed"] is True


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("compact", "non-compact target JSON"),
        ("pretty", "non-pretty target JSON"),
        ("compact_flag", "did not use the instance compact flag"),
        ("blocks", "invalid content-batch block count"),
        ("semantics", "root prompt semantics differ"),
        ("system", "system prompt digest changed"),
        ("raw", "raw prompt bytes violate arm contract"),
        ("size", "compact prompts are not smaller"),
        ("runtime", "changed provider call settings"),
        ("attribution", "missing root/repair attribution"),
        ("ordinal", "inconsistent repair ordinal"),
        ("candidate_count", "invalid request candidate count"),
        ("repair", "member-repair call amplification"),
        ("usage", "lacks complete token usage"),
        ("error_usage", "lacks complete token usage"),
        ("cache_metric", "supported cache usage omitted cached_input_tokens"),
        ("attempt_accounting", "lacks raw provider-attempt accounting"),
        ("classification", "topic_group agreement fell below the A/A noise floor"),
        ("savings", "prompt-token savings missed the 10% gate"),
        ("total_savings", "total-token savings missed the 8% gate"),
        ("cache", "provider cache ratio regressed beyond A/A noise"),
        ("cached_bounds", "cached input tokens exceed prompt tokens"),
    ],
)
def test_json_minify_transport_audit_fails_closed(
    failure: str,
    expected_reason: str,
) -> None:
    calls = _json_minify_audit_calls()
    if failure == "compact":
        calls[-1]["all_target_json_compact"] = False
    elif failure == "pretty":
        calls[0]["all_target_json_pretty"] = False
    elif failure == "compact_flag":
        calls[-1]["expected_compact_json"] = False
    elif failure == "blocks":
        calls[-1]["content_batch_json_block_count"] = 0
    elif failure == "semantics":
        calls[-1]["prompt_semantic_digest"] = "semantic-drift"
    elif failure == "system":
        calls[-1]["system_digest"] = "system-drift"
    elif failure == "raw":
        treatment_a = next(
            call
            for call in calls
            if call["pair_kind"] == "treatment" and call["repeat"] == 3 and call["arm"] == "A"
        )
        calls[-1]["prompt_digest"] = treatment_a["prompt_digest"]
    elif failure == "size":
        treatment_a = next(
            call
            for call in calls
            if call["pair_kind"] == "treatment" and call["repeat"] == 3 and call["arm"] == "A"
        )
        calls[-1]["prompt_chars"] = treatment_a["prompt_chars"]
        calls[-1]["prompt_bytes"] = treatment_a["prompt_bytes"]
    elif failure == "runtime":
        calls[-1]["max_tokens"] = 8192
    elif failure == "attribution":
        calls[-1].pop("request_kind")
    elif failure == "ordinal":
        calls[-1]["request_ordinal"] = 1
    elif failure == "candidate_count":
        calls[-1]["request_candidate_count"] = 0
    elif failure == "repair":
        for repeat in range(1, 4):
            root = next(
                call
                for call in calls
                if call["pair_kind"] == "treatment"
                and call["repeat"] == repeat
                and call["arm"] == "B"
            )
            repair = dict(root)
            repair.update(
                {
                    "request_kind": "repair",
                    "request_ordinal": 1,
                    "request_candidate_count": 1,
                    "structured_item_count": 1,
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "total_tokens": 55,
                        "cached_input_tokens": 5,
                    },
                }
            )
            calls.append(repair)
    elif failure == "usage":
        calls[-1]["usage"] = None
    elif failure == "error_usage":
        calls[-1]["status"] = "error"
        calls[-1]["usage"] = None
    elif failure == "cache_metric":
        for call in calls:
            if call["pair_kind"] == "treatment" and call["arm"] == "B":
                assert isinstance(call["usage"], dict)
                call["usage"].pop("cached_input_tokens")
    elif failure == "attempt_accounting":
        calls[-1]["provider"] = "openai_compatible"
    elif failure == "classification":
        for call in calls:
            if call["pair_kind"] == "treatment" and call["arm"] == "B":
                items = call["classification_items"]
                assert isinstance(items, list)
                fields = items[0]["fields"]
                assert isinstance(fields, dict)
                topic = fields["topic_group"]
                assert isinstance(topic, dict)
                topic["digest"] = "classification-drift"
    elif failure == "savings":
        for call in calls:
            if call["pair_kind"] == "treatment" and call["arm"] == "B":
                call["usage"] = {
                    "prompt_tokens": 950,
                    "completion_tokens": 120,
                    "total_tokens": 1070,
                    "cached_input_tokens": 95,
                }
    elif failure == "cache":
        for call in calls:
            if call["pair_kind"] == "treatment" and call["arm"] == "B":
                call["usage"] = {
                    "prompt_tokens": 850,
                    "completion_tokens": 30,
                    "total_tokens": 880,
                    "cached_input_tokens": 0,
                }
    elif failure == "total_savings":
        for call in calls:
            if call["pair_kind"] == "treatment" and call["arm"] == "B":
                call["usage"] = {
                    "prompt_tokens": 850,
                    "completion_tokens": 230,
                    "total_tokens": 1080,
                    "cached_input_tokens": 85,
                }
    elif failure == "cached_bounds":
        calls[-1]["usage"] = {
            "prompt_tokens": 850,
            "completion_tokens": 30,
            "total_tokens": 880,
            "cached_input_tokens": 851,
        }

    audit = _passing_json_minify_audit(calls)

    assert audit["passed"] is False
    assert expected_reason in " ".join(audit["blocking_reasons"])


def test_json_minify_transport_requires_profile_layer_cache_evidence() -> None:
    audit = validate_json_minify_transport(
        _json_minify_audit_calls(),
        enabled=True,
        repeats=3,
        arm_a_profile_cache_stats=None,
        arm_b_profile_cache_stats=None,
    )

    assert audit["passed"] is False
    assert "profile-layer cache evidence is missing" in " ".join(audit["blocking_reasons"])


def test_json_minify_transport_requires_profile_layer_cache_reuse() -> None:
    cold_stats = {
        "profile_core": {"digest": "digest-core", "hits": 0, "misses": 1},
    }
    audit = validate_json_minify_transport(
        _json_minify_audit_calls(),
        enabled=True,
        repeats=3,
        arm_a_profile_cache_stats=cold_stats,
        arm_b_profile_cache_stats=cold_stats,
    )

    assert audit["passed"] is False
    assert "recorded no reuse" in " ".join(audit["blocking_reasons"])


def test_json_minify_transport_empty_evidence_fails_with_strict_json_artifact() -> None:
    audit = validate_json_minify_transport([], enabled=True, repeats=3)

    assert audit["passed"] is False
    json.dumps(audit, allow_nan=False)


def _candidate_transport_audit_calls(experiment: str) -> list[dict[str, object]]:
    config = replay_script._CANDIDATE_TRANSPORT_EXPERIMENTS[experiment]
    arm_transports = {
        "A": str(config["arm_a_transport"]),
        "B": str(config["arm_b_transport"]),
    }
    calls: list[dict[str, object]] = []
    local_transports = {"sparse-json", "row-wire-v1"}

    def add_call(*, pair_kind: str, repeat: int, logical_run: str, arm: str) -> None:
        transport = arm_transports[arm]
        if experiment == "sparse-json":
            prompt_tokens, completion_tokens = (1000, 100) if arm == "A" else (750, 150)
            prompt_bytes = 10000 if arm == "A" else 6000
            candidate_bytes = 8000 if arm == "A" else 4000
            system_digest = f"stable-{arm}-identity-system"
        else:
            prompt_tokens, completion_tokens = (1000, 100) if arm == "A" else (900, 150)
            prompt_bytes = 6000 if arm == "A" else 5000
            candidate_bytes = 4000 if arm == "A" else 3200
            system_digest = "stable-local-id-system"
        classification_items = [
            {
                "candidate_key_digest": f"candidate-{index}",
                "position": index,
                "fields": {
                    field: {"digest": f"stable-{field}", "nonempty": True}
                    for field in replay_script._REPLAY_CLASSIFICATION_FIELDS
                },
            }
            for index in range(30)
        ]
        raw_prompt_digest = f"{transport}-prompt"
        calls.append(
            {
                "pair_kind": pair_kind,
                "repeat": repeat,
                "logical_run": logical_run,
                "arm": arm,
                "method": "complete_structured_task",
                "caller": "discovery.evaluate_batch",
                "temperature": 0.0,
                "max_tokens": 4096,
                "request_kind": "root",
                "request_ordinal": 0,
                "request_candidate_count": 30,
                "expected_candidate_transport": transport,
                "candidate_transport": transport,
                "candidate_decode_valid": True,
                "candidate_item_count": 30,
                "candidate_canonical_digest": "run-salted-canonical",
                "candidate_payload_bytes": candidate_bytes,
                "candidate_local_id_coverage_complete": transport in local_transports,
                "candidate_global_identity_field_count": 0,
                "candidate_url_field_count": 0,
                "user_context_digest": "run-salted-context",
                "image_input_count": 0,
                "image_payloads_digest": "run-salted-images",
                "image_anchor_coverage_complete": True,
                "result_identity_contract": (
                    "local-id" if transport in local_transports else "global-id"
                ),
                "result_local_id_binding_safe": transport in local_transports,
                "system_digest": system_digest,
                "prompt_digest": raw_prompt_digest,
                "prompt_chars": prompt_bytes,
                "prompt_bytes": prompt_bytes,
                "cache_metric_supported": False,
                "status": "ok",
                "structured_item_count": 30,
                "reason_field_count": 30,
                "classification_items": classification_items,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    for repeat in range(1, 4):
        add_call(pair_kind="control", repeat=repeat, logical_run="A1", arm="A")
        add_call(pair_kind="control", repeat=repeat, logical_run="A2", arm="A")
        add_call(pair_kind="treatment", repeat=repeat, logical_run="A", arm="A")
        add_call(pair_kind="treatment", repeat=repeat, logical_run="B", arm="B")
    return calls


@pytest.mark.parametrize(
    ("experiment", "expected_prompt_savings", "expected_total_savings"),
    [
        ("sparse-json", 0.25, pytest.approx(1 - 900 / 1100)),
        ("row-wire-v1", 0.10, pytest.approx(1 - 1050 / 1100)),
    ],
)
def test_candidate_transport_audit_passes_independent_locked_gates(
    experiment: str,
    expected_prompt_savings: float,
    expected_total_savings: object,
) -> None:
    audit = validate_candidate_transport_experiment(
        _candidate_transport_audit_calls(experiment),
        experiment=experiment,
        repeats=3,
    )

    assert audit["passed"] is True
    assert audit["arm_transports"] == (
        {"A": "production-json", "B": "sparse-json"}
        if experiment == "sparse-json"
        else {"A": "sparse-json", "B": "row-wire-v1"}
    )
    assert audit["token_gate"]["prompt_savings_median"] == pytest.approx(expected_prompt_savings)
    assert audit["token_gate"]["total_savings_median"] == expected_total_savings
    assert audit["cache_diagnostics_only"] is True
    assert audit["repair"]["passed"] is True
    assert audit["classification"]["passed"] is True


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("transport", "rendered production-json instead of sparse-json"),
        ("decode", "did not decode canonically"),
        ("count", "member count drifted"),
        ("canonical", "canonical digest is missing"),
        ("context", "non-candidate prompt digest is missing"),
        ("image_digest", "ordered image payload digest is missing"),
        ("local_ids", "local-ID request coverage is incomplete"),
        ("global_id", "leaked a global identity field"),
        ("url", "leaked a URL field"),
        ("image", "image/local-ID anchors do not match"),
        ("result_contract", "response did not use local IDs"),
        ("binding", "response local-ID binding was unsafe"),
        ("reason_contract", "response changed the reason output contract"),
        ("production_contract", "production response identity contract was not verified"),
        ("semantics", "canonical candidate semantics differ"),
        ("raw", "raw prompt contract failed"),
        ("size", "candidate transport is not smaller"),
        ("usage", "lacks complete billable usage"),
        ("zero_usage", "reported zero or incomplete billable usage"),
        ("savings", "prompt-token savings missed the 20% gate"),
        ("total", "total-token savings missed the >= 15% gate"),
        ("classification", "topic_group agreement fell below the A/A noise floor"),
    ],
)
def test_sparse_json_transport_audit_fails_closed(
    failure: str,
    expected_reason: str,
) -> None:
    calls = _candidate_transport_audit_calls("sparse-json")
    treatment_b = [
        call for call in calls if call["pair_kind"] == "treatment" and call["arm"] == "B"
    ]
    if failure == "transport":
        treatment_b[-1]["candidate_transport"] = "production-json"
    elif failure == "decode":
        treatment_b[-1]["candidate_decode_valid"] = False
    elif failure == "count":
        treatment_b[-1]["candidate_item_count"] = 29
    elif failure == "canonical":
        treatment_b[-1]["candidate_canonical_digest"] = ""
    elif failure == "context":
        treatment_b[-1]["user_context_digest"] = ""
    elif failure == "image_digest":
        treatment_b[-1]["image_payloads_digest"] = ""
    elif failure == "local_ids":
        treatment_b[-1]["candidate_local_id_coverage_complete"] = False
    elif failure == "global_id":
        treatment_b[-1]["candidate_global_identity_field_count"] = 1
    elif failure == "url":
        treatment_b[-1]["candidate_url_field_count"] = 1
    elif failure == "image":
        treatment_b[-1]["image_anchor_coverage_complete"] = False
    elif failure == "result_contract":
        treatment_b[-1]["result_identity_contract"] = "global-id"
    elif failure == "binding":
        treatment_b[-1]["result_local_id_binding_safe"] = False
    elif failure == "reason_contract":
        treatment_b[-1]["reason_field_count"] = 29
    elif failure == "production_contract":
        calls[0]["result_identity_contract"] = "unverified-global-id"
    elif failure == "semantics":
        treatment_b[-1]["candidate_canonical_digest"] = "semantic-drift"
    elif failure == "raw":
        treatment_b[-1]["prompt_digest"] = "production-json-prompt"
    elif failure == "size":
        treatment_b[-1]["candidate_payload_bytes"] = 9000
    elif failure == "usage":
        treatment_b[-1]["usage"] = None
    elif failure == "zero_usage":
        treatment_b[-1]["usage"] = {}
    elif failure == "savings":
        for call in treatment_b:
            call["usage"] = {
                "prompt_tokens": 850,
                "completion_tokens": 150,
                "total_tokens": 1000,
            }
    elif failure == "total":
        for call in treatment_b:
            call["usage"] = {
                "prompt_tokens": 750,
                "completion_tokens": 200,
                "total_tokens": 950,
            }
    elif failure == "classification":
        for call in treatment_b:
            items = call["classification_items"]
            assert isinstance(items, list)
            items[0]["fields"]["topic_group"]["digest"] = "classification-drift"

    audit = validate_candidate_transport_experiment(
        calls,
        experiment="sparse-json",
        repeats=3,
    )

    assert audit["passed"] is False
    assert expected_reason in " ".join(audit["blocking_reasons"])


def test_candidate_transport_reason_contract_ignores_control_arm_provider_noise() -> None:
    calls = _candidate_transport_audit_calls("sparse-json")
    control_a = next(call for call in calls if call["pair_kind"] == "control")
    control_a["reason_field_count"] = int(control_a["structured_item_count"]) - 1

    audit = validate_candidate_transport_experiment(
        calls,
        experiment="sparse-json",
        repeats=3,
    )

    assert audit["passed"] is True, audit["blocking_reasons"]


def test_row_wire_transport_requires_strictly_positive_total_savings() -> None:
    calls = _candidate_transport_audit_calls("row-wire-v1")
    for call in calls:
        if call["pair_kind"] == "treatment" and call["arm"] == "B":
            call["usage"] = {
                "prompt_tokens": 900,
                "completion_tokens": 200,
                "total_tokens": 1100,
            }

    audit = validate_candidate_transport_experiment(
        calls,
        experiment="row-wire-v1",
        repeats=3,
    )

    assert audit["passed"] is False
    assert "total-token savings missed the > 0% gate" in " ".join(audit["blocking_reasons"])


def test_artifact_keeps_raw_scores_digests_usage_routes_without_private_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.run_profile_diet_ab._git_metadata",
        lambda: {"commit": "abc123", "dirty": False},
    )
    candidate = ReplayCandidate(
        candidate_id=1,
        title="PRIVATE TITLE",
        source_strategy="feed",
        source_platform="twitter",
        content_id="item-1",
    )
    pair = _pair(
        kind="control",
        repeat=1,
        flip_rate=0.0,
        spearman=1.0,
        admission_delta=0.0,
    )
    output = tmp_path / "artifact.json"
    calls = [
        {
            "pair_kind": "treatment",
            "repeat": 1,
            "logical_run": "A",
            "arm": "A",
            "provider": "openai",
            "status": "ok",
            "structured_output_parseable": True,
            "structured_item_count": 1,
            "reason_field_count": 1,
            "classification_items": [],
            "usage": {"output_tokens": 7},
        }
    ]
    reason_output_audit = validate_reason_off_outputs(calls, enabled=False)

    _write_artifact(
        output,
        args=SimpleNamespace(arm_b="json-minify", repeats=3, platform=None),
        db_path=tmp_path / "production.db",
        config_path=tmp_path / "config.toml",
        rows=[
            {
                "id": 1,
                "status": "evaluated",
                "body_text": "PRIVATE BODY",
                "title": "PRIVATE TITLE",
            }
        ],
        profile_snapshot=ReplayProfileSnapshot(
            raw_profile=object(),
            effective_profile=object(),
            raw_digest="raw-digest",
            effective_digest="effective-digest",
            overrides_present=True,
            active_speculation_count=2,
        ),
        negative_examples=None,
        candidates=[candidate],
        control_pairs=[pair],
        treatment_pairs=[pair],
        gate_passed=True,
        gate={"blocking_reasons": []},
        admission_min_score=0.6,
        calls=calls,
        route_audit={"passed": True, "logical_runs": []},
        embedding_audit={"passed": True, "namespace": "embed-v1"},
        recall_audit={"passed": True, "injected_label_count": 0},
        reason_output_audit=reason_output_audit,
        prompt_transport_audit={
            "enabled": True,
            "passed": True,
            "blocking_reasons": [],
            "logical_runs": [{"prompt_chars": 123, "prompt_bytes": 145}],
        },
        production_prefilter_mode="shadow",
        topic_lifecycle_serialization=True,
    )

    raw_artifact = output.read_text(encoding="utf-8")
    artifact = json.loads(raw_artifact)
    assert artifact["schema_version"] == 4
    assert artifact["snapshot"]["raw_profile_digest"] != "raw-digest"
    assert len(artifact["snapshot"]["raw_profile_digest"]) == 64
    assert artifact["candidates"][0]["candidate_ordinal"] == 0
    assert "candidate_id" not in artifact["candidates"][0]
    assert artifact["control_pairs"][0]["scores_a"] == [0.6]
    assert artifact["control_pairs"][0]["scores_a_digest"]
    assert artifact["llm_calls"][0]["usage"] == {"output_tokens": 7}
    assert artifact["routes"]["passed"] is True
    assert artifact["prompt_transport"]["passed"] is True
    assert artifact["prompt_transport"]["logical_runs"][0]["prompt_bytes"] == 145
    assert (
        artifact["reason_output"]["token_usage"]["treatment_comparison"]["A"]["completion_tokens"]
        == 7
    )
    assert (
        artifact["reason_output"]["token_usage"]["treatment_comparison"]["A"]["prompt_tokens"] == 0
    )
    assert artifact["gate_constants"]["llm_max_tokens"] == 4096
    assert artifact["gate_constants"]["rate_limit_retry_delays_seconds"] == [
        65.0,
        130.0,
        260.0,
        520.0,
    ]
    assert artifact["production_context"] == {
        "eval_prefilter_mode": "shadow",
        "topic_lifecycle_serialization": "on",
    }
    assert artifact["replay_context"] == {"eval_prefilter_mode": "off"}
    assert "PRIVATE TITLE" not in raw_artifact
    assert "PRIVATE BODY" not in raw_artifact
    assert "item-1" not in raw_artifact
    assert "production.db" not in raw_artifact
    assert "config.toml" not in raw_artifact
    assert "raw-digest" not in raw_artifact


def test_select_replay_rows_preserves_recent_production_mix() -> None:
    """The gate must not reweight platform/strategy groups."""
    rows = []
    for index in range(6):
        rows.append(
            {
                "id": 100 + index,
                "status": "cached",
                "source_platform": "reddit",
                "source_strategy": "subreddit",
                "evaluated_at": f"2026-07-05 12:0{index}:00",
            }
        )
    rows.append(
        {
            "id": 200,
            "status": "cached",
            "source_platform": "bilibili",
            "source_strategy": "search",
            "evaluated_at": "2026-07-01 08:00:00",
        }
    )

    selected = select_replay_rows(rows, sample=4)

    assert [row["id"] for row in selected] == [105, 104, 103, 102]
    # Deterministic: same input -> same output.
    assert [row["id"] for row in select_replay_rows(rows, sample=4)] == [
        row["id"] for row in selected
    ]


def test_profile_snapshot_matches_effective_soul_profile_contract(tmp_path: Path) -> None:
    memory = MemoryManager(tmp_path)
    raw_profile = OnionProfile()
    raw_profile.core.core_traits = ["raw trait"]
    soul_layer = memory.get_layer("soul")
    soul_layer.data.update(raw_profile.to_dict())
    soul_layer.save()
    memory.save_profile_overrides(
        ProfileOverrides(
            list_edits={"core.core_traits": ListEdit(add=["user pinned"], remove=["raw trait"])}
        )
    )
    save_speculative_state(
        tmp_path,
        SpeculativeState(
            active=[
                SpeculativeInterest(domain="active guess", reason="evidence", status="active"),
                SpeculativeInterest(domain="confirmed", reason="done", status="confirmed"),
            ]
        ),
    )

    snapshot = _load_profile_snapshot(tmp_path)

    assert snapshot.raw_profile.core.core_traits == ["raw trait"]
    assert snapshot.effective_profile.core.core_traits == ["user pinned"]
    assert snapshot.overrides_present is True
    assert snapshot.active_speculation_count == 1
    assert [
        item.domain
        for item in snapshot.effective_profile._active_speculations  # type: ignore[attr-defined]
    ] == ["active guess"]
    assert snapshot.raw_digest != snapshot.effective_digest


def test_replay_mirrors_and_restores_topic_lifecycle_serialization_config() -> None:
    from openbiliclaw.soul.profile_views import (
        set_topic_lifecycle_serialization,
        topic_lifecycle_serialization_enabled,
    )

    profile = OnionProfile()
    profile.interest.likes = [
        InterestDomain(domain="active topic", weight=0.8, state="active"),
        InterestDomain(domain="archived topic", weight=0.9, state="archived"),
    ]
    enabled_config = SimpleNamespace(soul=SimpleNamespace(topic_lifecycle_serialization="on"))
    disabled_config = SimpleNamespace(soul=SimpleNamespace(topic_lifecycle_serialization="off"))

    set_topic_lifecycle_serialization(False)
    try:
        with configured_topic_lifecycle_serialization(enabled_config) as enabled:
            assert enabled is True
            assert topic_lifecycle_serialization_enabled() is True
            assert [
                item["domain"] for item in build_profile_summary(profile)["interest_domains"]
            ] == ["active topic"]
        assert topic_lifecycle_serialization_enabled() is False

        set_topic_lifecycle_serialization(True)
        with configured_topic_lifecycle_serialization(disabled_config) as enabled:
            assert enabled is False
            assert topic_lifecycle_serialization_enabled() is False
        assert topic_lifecycle_serialization_enabled() is True
    finally:
        set_topic_lifecycle_serialization(False)


@pytest.mark.parametrize("mode", ["off", "shadow", "invalid"])
def test_replay_accepts_non_enforcing_production_prefilter(mode: str) -> None:
    config = SimpleNamespace(discovery=SimpleNamespace(eval_prefilter_mode=mode))

    assert validate_replay_prefilter_compatibility(config) == (
        mode if mode in {"off", "shadow"} else "shadow"
    )


def test_replay_rejects_enforcing_production_prefilter() -> None:
    config = SimpleNamespace(discovery=SimpleNamespace(eval_prefilter_mode="enforce"))

    with pytest.raises(RuntimeError, match="production config is enforce"):
        validate_replay_prefilter_compatibility(config)


class _ReplayDiscoveryConfig:
    multimodal_evaluation_enabled = False
    multimodal_batch_size = 8
    multimodal_image_max_px = 384
    multimodal_image_quality = 72
    multimodal_image_timeout_seconds = 6


class _ReplayConfig:
    discovery = _ReplayDiscoveryConfig()


class _ReplayEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if text else []


class _SequenceEmbedding:
    cache_model_namespace = "provider:model#namespace=test"
    similarity_threshold = 0.82

    def __init__(self, results: list[object]) -> None:
        self.results = list(results)

    async def embed(self, text: str) -> object:
        del text
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([], "empty or non-list"),
        ([1.0, float("nan")], "NaN or infinity"),
        ([1.0, "bad"], "non-numeric"),
        (RuntimeError("provider down"), "raised RuntimeError"),
    ],
)
async def test_embedding_audit_fails_closed_on_invalid_results(
    result: object,
    message: str,
) -> None:
    audit = ReplayEmbeddingAudit(_SequenceEmbedding([result]))

    with pytest.raises(ReplayEmbeddingValidationError, match=message):
        await audit.embed("interest")

    assert audit.calls[0]["status"] == "error"
    assert audit.errors


@pytest.mark.asyncio
async def test_embedding_audit_rejects_dimension_drift() -> None:
    audit = ReplayEmbeddingAudit(_SequenceEmbedding([[1.0, 0.0], [1.0, 0.0, 0.0]]))

    await audit.embed("interest")
    with pytest.raises(ReplayEmbeddingValidationError, match="dimension drift"):
        await audit.embed("content")


@pytest.mark.asyncio
async def test_embedding_audit_accepts_complete_vectors_with_zero_injection() -> None:
    audit = ReplayEmbeddingAudit(_SequenceEmbedding([[1.0, 0.0], [0.0, 1.0]]))
    recall = ReplayRecallAudit()
    with replay_call_attribution(
        pair_kind="treatment",
        repeat=1,
        logical_run="B",
        arm="B",
    ):
        await audit.embed("tail interest")
        await audit.embed("unrelated content")
        recall.record_batch({}, candidate_count=1)

    summary = audit.summary(eligible_tail_count=1, recall_audit=recall)

    assert summary["passed"] is True
    assert summary["call_count"] == 2
    assert recall.payload()["injected_label_count"] == 0
    assert "tail interest" not in json.dumps(summary)
    assert "unrelated content" not in json.dumps(summary)
    assert summary["calls"][0]["request_digest"] != replay_script._digest("tail interest")


def test_embedding_audit_accepts_zero_tail_without_requests() -> None:
    audit = ReplayEmbeddingAudit(_SequenceEmbedding([]))

    summary = audit.summary(eligible_tail_count=0, recall_audit=ReplayRecallAudit())

    assert summary["passed"] is True
    assert summary["call_count"] == 0


def test_run_scoped_embedding_cache_lives_through_context_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cache:
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Service:
        cache_model_namespace = "test:model"

        def __init__(self) -> None:
            self._l2_cache = _Cache()

    config = SimpleNamespace(
        data_dir=str(tmp_path / "production"),
        llm=SimpleNamespace(embedding=SimpleNamespace(provider="test")),
    )
    service = _Service()
    monkeypatch.setattr(
        "scripts.run_profile_diet_ab._build_embedding_service",
        lambda _config: service,
    )

    with run_scoped_embedding_audit(config, allow_no_embedding=False) as audit:
        cache_dir = Path(config.data_dir)
        assert cache_dir.exists()
        assert audit is not None
        assert service._l2_cache.closed is False

    assert not cache_dir.exists()
    assert service._l2_cache.closed is True
    assert config.data_dir == str(tmp_path / "production")


def test_no_embedding_requires_explicit_degraded_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        data_dir=str(tmp_path / "production"),
        llm=SimpleNamespace(embedding=SimpleNamespace(provider="", fallback_provider="")),
    )
    monkeypatch.setattr(
        "scripts.run_profile_diet_ab._build_embedding_service",
        lambda _config: None,
    )

    with (
        pytest.raises(RuntimeError, match="--allow-no-embedding"),
        run_scoped_embedding_audit(config, allow_no_embedding=False),
    ):
        pass
    with run_scoped_embedding_audit(config, allow_no_embedding=True) as audit:
        assert audit is None


def test_degraded_flag_cannot_mask_configured_embedding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        data_dir=str(tmp_path / "production"),
        llm=SimpleNamespace(embedding=SimpleNamespace(provider="ollama", fallback_provider="")),
    )
    monkeypatch.setattr(
        "scripts.run_profile_diet_ab._build_embedding_service",
        lambda _config: None,
    )

    with (
        pytest.raises(RuntimeError, match="could not be constructed"),
        run_scoped_embedding_audit(config, allow_no_embedding=True),
    ):
        pass


class _RecordingMultimodalService:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.response_content = json.dumps(
            {
                "content_id": "item-1",
                "score": 0.7,
                "topic_group": "系统",
                "style_key": "deep_focus",
                "franchise_key": "",
            },
            ensure_ascii=False,
        )

    async def complete_multimodal_structured_task(self, **kwargs: object) -> LLMResponse:
        self.kwargs = dict(kwargs)
        return LLMResponse(
            content=self.response_content,
            provider="sensenova",
            instance_id="gateway",
            model="reasoning-model",
            usage={"output_tokens": 12},
        )


def test_replay_usage_normalizes_supported_cold_cache_to_explicit_zero() -> None:
    usage, semantics, supported = replay_script._normalized_replay_usage(
        SimpleNamespace(
            provider="openai_compatible",
            usage={"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
        )
    )

    assert supported is True
    assert semantics == "prompt_includes_cached"
    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "total_tokens": 128,
        "cached_input_tokens": 0,
    }


def test_replay_usage_normalizes_claude_cache_accounting_to_total_prompt() -> None:
    usage, semantics, supported = replay_script._normalized_replay_usage(
        SimpleNamespace(
            provider="claude",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_input_tokens": 100,
                "cache_creation_input_tokens": 20,
            },
        )
    )

    assert supported is True
    assert semantics == "prompt_excludes_cached"
    assert usage == {
        "prompt_tokens": 130,
        "completion_tokens": 5,
        "cached_input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "provider_uncached_input_tokens": 10,
        "total_tokens": 135,
    }


@pytest.mark.asyncio
async def test_replay_accounts_hidden_openai_adapter_attempt_usage() -> None:
    class Provider:
        name = "openai_compatible"

        def __init__(self) -> None:
            self.responses = [
                SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=100,
                        completion_tokens=10,
                        total_tokens=110,
                    )
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=90,
                        completion_tokens=5,
                        total_tokens=95,
                    )
                ),
            ]

        async def _request_with_retry(self, **kwargs: object) -> object:
            del kwargs
            return self.responses.pop(0)

    class Registry:
        available_providers = ["gateway"]

        def __init__(self, provider: Provider) -> None:
            self.provider = provider

        def provider_type(self, name: str) -> str:
            assert name == "gateway"
            return "openai_compatible"

        def get(self, name: str) -> Provider:
            assert name == "gateway"
            return self.provider

    class Inner:
        def __init__(self, provider: Provider) -> None:
            self.provider = provider

        async def complete_structured_task(self, **kwargs: object) -> LLMResponse:
            del kwargs
            await self.provider._request_with_retry()
            await self.provider._request_with_retry()
            return LLMResponse(
                content='{"results": [{"content_id": "safe", "score": 0.7}]}',
                provider="openai_compatible",
                model="test-model",
                usage={"prompt_tokens": 90, "completion_tokens": 5, "total_tokens": 95},
            )

    provider = Provider()
    recorder = replay_script._ProviderAttemptUsageRecorder()
    recorder.instrument_registry(Registry(provider))
    service = _DeterministicLLMService(
        Inner(provider),
        service="arm_a",
        attempt_usage_recorder=recorder,
    )

    await service.complete_structured_task(system_instruction="system", user_input="user")

    call = service.calls[0]
    assert call["provider_attempt_count"] == 2
    assert call["provider_hidden_retry_count"] == 1
    assert call["provider_attempt_usage_complete"] is True
    assert call["provider_attempt_accounting"] == "raw_adapter_attempts"
    assert call["usage"] == {
        "prompt_tokens": 190,
        "completion_tokens": 15,
        "total_tokens": 205,
        "cached_input_tokens": 0,
    }


@pytest.mark.asyncio
async def test_deterministic_wrapper_keeps_production_budget_for_multimodal() -> None:
    inner = _RecordingMultimodalService()
    service = _DeterministicLLMService(inner, service="arm_a")

    with replay_call_attribution(
        pair_kind="control",
        repeat=2,
        logical_run="A1",
        arm="A",
    ):
        await service.complete_multimodal_structured_task(
            system_instruction="system",
            user_input="user",
            image_inputs=[],
            max_tokens=4096,
        )

    assert inner.kwargs["temperature"] == 0.0
    assert inner.kwargs["max_tokens"] == 4096
    output_metadata = replay_script._structured_output_metadata(
        SimpleNamespace(content=inner.response_content),
    )
    prompt_metadata = replay_script._prompt_transport_metadata(
        {"system_instruction": "system", "user_input": "user"},
        expected_compact_json=False,
    )
    assert service.calls == [
        {
            "service": "arm_a",
            "pair_kind": "control",
            "repeat": 2,
            "logical_run": "A1",
            "arm": "A",
            "method": "complete_multimodal_structured_task",
            "caller": "",
            "provider": "sensenova",
            "instance_id": "gateway",
            "model": "reasoning-model",
            "temperature": 0.0,
            "max_tokens": 4096,
            **prompt_metadata,
            "cache_usage_semantics": "unsupported",
            "cache_metric_supported": False,
            "usage": {"output_tokens": 12},
            "provider_attempt_count": 1,
            "provider_hidden_retry_count": 0,
            "provider_attempt_usage_complete": True,
            "provider_attempt_accounting": "logical_response_only",
            "provider_attempts": [],
            "status": "ok",
            **output_metadata,
        }
    ]


def test_reason_output_metadata_counts_fields_without_retaining_payload() -> None:
    response = SimpleNamespace(
        content=(
            '{"results": ['
            '{"content_id": "PRIVATE-ID-1", "score": 0.8, '
            '"reason": "PRIVATE REASON", "topic_group": "PRIVATE TOPIC", '
            '"style_key": "deep_dive", "franchise_key": "PRIVATE IP"}, '
            '{"content_id": "PRIVATE-ID-2", "score": 0.4, '
            '"diagnostic": {"reason": "PRIVATE NESTED REASON"}, '
            '"topic_group": "PRIVATE TOPIC 2", "style_key": "social_chat", '
            '"franchise_key": ""}]}'
        )
    )

    metadata = replay_script._structured_output_metadata(response)

    assert metadata["structured_output_parseable"] is True
    assert metadata["structured_item_count"] == 2
    assert metadata["reason_field_count"] == 1
    classification_items = metadata["classification_items"]
    assert isinstance(classification_items, list)
    assert len(classification_items) == 2
    assert classification_items[0]["fields"]["topic_group"]["nonempty"] is True
    assert classification_items[0]["fields"]["style_key"]["nonempty"] is True
    assert classification_items[1]["fields"]["franchise_key"]["nonempty"] is False
    assert "PRIVATE" not in json.dumps(metadata)


def _reason_audit_call(
    *,
    pair_kind: str,
    repeat: int,
    logical_run: str,
    arm: str,
    topic_group: str = "系统",
    style_key: str = "deep_focus",
    franchise_key: str = "",
    missing_field: str = "",
) -> dict[str, object]:
    result: dict[str, object] = {
        "content_id": "item-1",
        "score": 0.7,
        "topic_group": topic_group,
        "style_key": style_key,
        "franchise_key": franchise_key,
    }
    if arm == "A":
        result["reason"] = "production diagnostic"
    if missing_field:
        result.pop(missing_field)
    metadata = replay_script._structured_output_metadata(
        SimpleNamespace(content=json.dumps({"results": [result]})),
    )
    if pair_kind == "control":
        usage = {"prompt_tokens": 1000, "completion_tokens": 999, "total_tokens": 1999}
    elif arm == "A":
        usage = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
    else:
        usage = {"input_tokens": 90, "output_tokens": 20, "total_tokens": 110}
    return {
        "pair_kind": pair_kind,
        "repeat": repeat,
        "logical_run": logical_run,
        "arm": arm,
        "status": "ok",
        "usage": usage,
        **metadata,
    }


def _complete_reason_off_audit_calls(
    *,
    b_topic_group: str = "系统",
    b_style_key: str = "deep_focus",
    b_missing_field: str = "",
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for repeat in range(1, 4):
        calls.extend(
            [
                _reason_audit_call(
                    pair_kind="control",
                    repeat=repeat,
                    logical_run="A1",
                    arm="A",
                ),
                _reason_audit_call(
                    pair_kind="control",
                    repeat=repeat,
                    logical_run="A2",
                    arm="A",
                ),
                _reason_audit_call(
                    pair_kind="treatment",
                    repeat=repeat,
                    logical_run="A",
                    arm="A",
                ),
                _reason_audit_call(
                    pair_kind="treatment",
                    repeat=repeat,
                    logical_run="B",
                    arm="B",
                    topic_group=b_topic_group,
                    style_key=b_style_key,
                    missing_field=b_missing_field,
                ),
            ]
        )
    return calls


def test_reason_off_output_audit_attributes_usage_and_contract_to_each_arm() -> None:
    calls = _complete_reason_off_audit_calls()
    calls.append(
        {
            "pair_kind": "treatment",
            "repeat": 1,
            "logical_run": "B",
            "arm": "B",
            "status": "error",
            "usage": None,
            "structured_output_parseable": False,
            "structured_item_count": 0,
            "reason_field_count": 0,
            "classification_items": [],
        }
    )
    audit = validate_reason_off_outputs(calls, enabled=True)

    assert audit["passed"] is True
    assert audit["reason_field_count"] == {"A": 9, "B": 0}
    token_usage = audit["token_usage"]
    comparison = token_usage["treatment_comparison"]
    assert comparison["A"]["prompt_tokens_per_evaluated_item"] == 100
    assert comparison["B"]["prompt_tokens_per_evaluated_item"] == 90
    assert comparison["A"]["completion_tokens_per_evaluated_item"] == 40
    assert comparison["B"]["completion_tokens_per_evaluated_item"] == 20
    assert comparison["A"]["total_tokens_per_evaluated_item"] == 140
    assert comparison["B"]["total_tokens_per_evaluated_item"] == 110
    assert comparison["B"]["error_call_count"] == 1
    assert comparison["B"]["usage_missing_call_count"] == 1
    assert len(token_usage["logical_runs"]) == 12
    classification = audit["classification"]
    assert classification["passed"] is True
    assert classification["gate"]["topic_group"]["treatment_agreement_median"] == 1.0
    assert classification["cap_drop_audit"].startswith("not measured")


@pytest.mark.parametrize(
    ("calls", "expected_reason"),
    [
        (
            _complete_reason_off_audit_calls(b_topic_group="不同主题"),
            "topic_group agreement fell below",
        ),
        (
            _complete_reason_off_audit_calls(b_style_key=""),
            "style_key fill rate regressed",
        ),
        (
            _complete_reason_off_audit_calls(b_missing_field="franchise_key"),
            "franchise_key presence rate regressed",
        ),
    ],
)
def test_reason_off_classification_audit_blocks_non_score_regressions(
    calls: list[dict[str, object]],
    expected_reason: str,
) -> None:
    audit = validate_reason_off_outputs(calls, enabled=True)

    assert audit["passed"] is False
    assert expected_reason in " ".join(audit["blocking_reasons"])


@pytest.mark.parametrize(
    ("call", "expected_reason"),
    [
        (
            {
                "arm": "B",
                "status": "ok",
                "structured_output_parseable": False,
                "structured_item_count": 0,
                "reason_field_count": 0,
            },
            "without verifiable scored JSON",
        ),
        (
            {
                "arm": "B",
                "status": "ok",
                "structured_output_parseable": True,
                "structured_item_count": 1,
                "reason_field_count": 1,
            },
            "one or more reason fields",
        ),
    ],
)
def test_reason_off_output_audit_fails_closed(
    call: dict[str, object],
    expected_reason: str,
) -> None:
    audit = validate_reason_off_outputs([call], enabled=True)

    assert audit["passed"] is False
    assert expected_reason in " ".join(audit["blocking_reasons"])


def test_reason_output_audit_is_non_blocking_for_other_replay_arms() -> None:
    audit = validate_reason_off_outputs(
        [
            {
                "arm": "B",
                "status": "ok",
                "structured_output_parseable": True,
                "structured_item_count": 1,
                "reason_field_count": 1,
            }
        ],
        enabled=False,
    )

    assert audit["passed"] is True
    assert audit["blocking_reasons"] == []


@pytest.mark.asyncio
async def test_deterministic_wrapper_labels_transient_rate_limit_for_route_audit() -> None:
    class _RateLimitedService:
        async def complete_structured_task(self, **kwargs: object) -> LLMResponse:
            del kwargs
            try:
                raise LLMRateLimitError("openai_compatible rate limit exceeded")
            except LLMRateLimitError as exc:
                raise LLMProviderExecutionError("All providers failed") from exc

    service = _DeterministicLLMService(_RateLimitedService(), service="arm_a")

    with (
        replay_call_attribution(
            pair_kind="control",
            repeat=1,
            logical_run="A1",
            arm="A",
        ),
        pytest.raises(LLMProviderExecutionError),
    ):
        await service.complete_structured_task(system_instruction="system", user_input="user")

    assert service.calls[0]["status"] == "error"
    assert service.calls[0]["error_kind"] == "transient_rate_limit"


def _attributed_route_calls(
    *,
    treatment_b_route: tuple[str, str, str] = ("openai", "primary", "model-a"),
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    baseline_route = ("openai", "primary", "model-a")
    for repeat in range(1, 4):
        for pair_kind, logical_run, arm in (
            ("control", "A1", "A"),
            ("control", "A2", "A"),
            ("treatment", "A", "A"),
            ("treatment", "B", "B"),
        ):
            route = treatment_b_route if pair_kind == "treatment" and arm == "B" else baseline_route
            calls.append(
                {
                    "pair_kind": pair_kind,
                    "repeat": repeat,
                    "logical_run": logical_run,
                    "arm": arm,
                    "provider": route[0],
                    "instance_id": route[1],
                    "model": route[2],
                    "status": "ok",
                }
            )
    return calls


def test_route_audit_enforces_non_model_arm_equivalence() -> None:
    passed = validate_replay_routes(
        _attributed_route_calls(),
        repeats=3,
        model_override=None,
    )
    drifted = validate_replay_routes(
        _attributed_route_calls(treatment_b_route=("openai_compatible", "fallback", "model-b")),
        repeats=3,
        model_override=None,
    )

    assert passed["passed"] is True
    assert drifted["passed"] is False
    assert "drifted route" in " ".join(drifted["blocking_reasons"])


def test_route_audit_allows_only_requested_model_treatment_route() -> None:
    audit = validate_replay_routes(
        _attributed_route_calls(
            treatment_b_route=("openai_compatible", "diet-instance", "diet-model")
        ),
        repeats=3,
        model_override=ModelOverride(provider="diet-instance", model=""),
    )

    assert audit["passed"] is True
    assert len(audit["logical_runs"]) == 12


def test_route_audit_rejects_consistent_unexpected_failover() -> None:
    audit = validate_replay_routes(
        _attributed_route_calls(),
        repeats=3,
        model_override=None,
        expected_control_instance="configured-primary",
        expected_treatment_instance="configured-primary",
    )

    assert audit["passed"] is False
    assert "unexpectedly failed over" in " ".join(audit["blocking_reasons"])


def test_route_audit_rejects_empty_and_mixed_routes_within_logical_run() -> None:
    calls = _attributed_route_calls()
    calls[0]["model"] = ""
    calls.append(
        {
            **calls[1],
            "pair_kind": "control",
            "repeat": 1,
            "logical_run": "A2",
            "instance_id": "unexpected",
        }
    )

    audit = validate_replay_routes(calls, repeats=3, model_override=None)
    reasons = " ".join(audit["blocking_reasons"])

    assert audit["passed"] is False
    assert "empty actual route" in reasons
    assert "mixed 2 actual routes" in reasons


def test_route_audit_allows_a_recovered_transient_rate_limit() -> None:
    calls = _attributed_route_calls()
    calls.append(
        {
            **calls[0],
            "provider": "",
            "instance_id": "",
            "model": "",
            "status": "error",
            "error_kind": "transient_rate_limit",
        }
    )

    audit = validate_replay_routes(calls, repeats=3, model_override=None)

    assert audit["passed"] is True
    assert audit["recovered_rate_limit_call_count"] == 1
    recovered_run = next(
        run
        for run in audit["logical_runs"]
        if run["pair_kind"] == "control" and run["repeat"] == 1 and run["logical_run"] == "A1"
    )
    assert recovered_run["call_count"] == 2
    assert recovered_run["successful_call_count"] == 1
    assert recovered_run["recovered_rate_limit_call_count"] == 1


class _MissingResponseEngine:
    _EVALUATE_BATCH_HARD_CAP = 90

    async def evaluate_content_batch(
        self,
        contents: list[object],
        profile: object,
        *,
        source_context: str,
        batch_size: int,
    ) -> list[float]:
        del profile, source_context, batch_size
        contents[0].relevance_reason = "evaluation_response_missing"
        return [0.0 for _content in contents]


@pytest.mark.asyncio
async def test_score_contents_rejects_missing_evaluation_responses() -> None:
    class _Content:
        content_id = "failed-item"
        title = "failed"
        relevance_reason = ""

    with pytest.raises(RuntimeError, match="cannot be counted as zero-score"):
        await _score_contents(
            _MissingResponseEngine(),
            [_Content()],
            object(),
            source_context="replay",
        )


@pytest.mark.asyncio
async def test_score_contents_matches_production_claim_grouping_and_context() -> None:
    class _RecordingEngine:
        _EVALUATE_BATCH_HARD_CAP = 90

        def __init__(self) -> None:
            self.calls: list[tuple[int, str, int]] = []

        async def evaluate_content_batch(
            self,
            contents: list[object],
            profile: object,
            *,
            source_context: str,
            batch_size: int,
        ) -> list[float]:
            del profile
            self.calls.append((len(contents), source_context, batch_size))
            return [0.7] * len(contents)

    engine = _RecordingEngine()
    contents = [
        SimpleNamespace(content_id=f"candidate-{index}", relevance_reason="")
        for index in range(100)
    ]

    scores = await _score_contents(
        engine,
        contents,
        object(),
        source_context=_REPLAY_SOURCE_CONTEXT,
    )

    assert _REPLAY_SOURCE_CONTEXT == "mixed"
    assert engine.calls == [
        (30, "mixed", 30),
        (30, "mixed", 30),
        (30, "mixed", 30),
        (10, "mixed", 30),
    ]
    assert scores == [0.7] * 100


@pytest.mark.asyncio
async def test_score_contents_retries_transient_rate_limit_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class _RateLimitedOnceEngine:
        _EVALUATE_BATCH_HARD_CAP = 90

        def __init__(self) -> None:
            self.states: list[tuple[float, str]] = []

        async def evaluate_content_batch(
            self,
            contents: list[object],
            profile: object,
            *,
            source_context: str,
            batch_size: int,
        ) -> list[float]:
            del profile, source_context, batch_size
            content = contents[0]
            self.states.append((content.relevance_score, content.relevance_reason))
            if len(self.states) == 1:
                content.relevance_score = 0.99
                content.relevance_reason = "partial failed attempt"
                try:
                    raise LLMRateLimitError("openai_compatible rate limit exceeded")
                except LLMRateLimitError as exc:
                    raise LLMProviderExecutionError("All providers failed") from exc
            return [0.7]

    monkeypatch.setattr(replay_script.asyncio, "sleep", fake_sleep)
    engine = _RateLimitedOnceEngine()
    content = SimpleNamespace(
        content_id="candidate-1",
        title="candidate",
        relevance_score=0.1,
        relevance_reason="original",
    )

    scores = await _score_contents(engine, [content], object(), source_context="mixed")

    assert scores == [0.7]
    assert sleeps == [65.0]
    assert engine.states == [(0.1, "original"), (0.1, "original")]


def test_replay_rate_limit_classification_stops_at_normalized_provider_error() -> None:
    raw_sdk_error = RuntimeError("429 response metadata included a billing field")
    try:
        raise LLMRateLimitError("openai_compatible rate limit exceeded") from raw_sdk_error
    except LLMRateLimitError as normalized:
        try:
            raise LLMProviderExecutionError("All providers failed") from normalized
        except LLMProviderExecutionError as wrapped:
            assert replay_script._is_retryable_replay_rate_limit(wrapped)


@pytest.mark.asyncio
async def test_score_contents_resets_rate_limit_budget_for_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class _PerChunkRateLimitedEngine:
        _EVALUATE_BATCH_HARD_CAP = 90

        def __init__(self) -> None:
            self.calls_by_start: dict[int, int] = {}

        async def evaluate_content_batch(
            self,
            contents: list[object],
            profile: object,
            *,
            source_context: str,
            batch_size: int,
        ) -> list[float]:
            del profile, source_context, batch_size
            start = int(str(contents[0].content_id))
            self.calls_by_start[start] = self.calls_by_start.get(start, 0) + 1
            failures_before_success = 2 if start == 0 else 1
            if self.calls_by_start[start] <= failures_before_success:
                try:
                    raise LLMRateLimitError("openai_compatible rate limit exceeded")
                except LLMRateLimitError as exc:
                    raise LLMProviderExecutionError("All providers failed") from exc
            return [0.7] * len(contents)

    monkeypatch.setattr(replay_script.asyncio, "sleep", fake_sleep)
    engine = _PerChunkRateLimitedEngine()
    contents = [
        SimpleNamespace(
            content_id=str(index),
            relevance_score=0.0,
            relevance_reason="",
        )
        for index in range(40)
    ]

    scores = await _score_contents(engine, contents, object(), source_context="mixed")

    assert scores == [0.7] * 40
    assert engine.calls_by_start == {0: 3, 30: 2}
    assert sleeps == [65.0, 130.0, 65.0]


@pytest.mark.asyncio
async def test_score_contents_uses_bounded_extended_budget_for_sustained_throttling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class _SustainedRateLimitedEngine:
        _EVALUATE_BATCH_HARD_CAP = 90

        def __init__(self) -> None:
            self.calls = 0

        async def evaluate_content_batch(
            self,
            contents: list[object],
            profile: object,
            *,
            source_context: str,
            batch_size: int,
        ) -> list[float]:
            del profile, source_context, batch_size
            self.calls += 1
            if self.calls <= len(replay_script.RATE_LIMIT_RETRY_DELAYS_SECONDS):
                try:
                    raise LLMRateLimitError("openai_compatible rate limit exceeded")
                except LLMRateLimitError as exc:
                    raise LLMProviderExecutionError("All providers failed") from exc
            return [0.7] * len(contents)

    monkeypatch.setattr(replay_script.asyncio, "sleep", fake_sleep)
    engine = _SustainedRateLimitedEngine()
    content = SimpleNamespace(
        content_id="candidate-1",
        relevance_score=0.0,
        relevance_reason="",
    )

    scores = await _score_contents(engine, [content], object(), source_context="mixed")

    assert scores == [0.7]
    assert engine.calls == 5
    assert sleeps == [65.0, 130.0, 260.0, 520.0]


@pytest.mark.asyncio
async def test_score_contents_does_not_retry_non_transient_quota_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class _BillingLimitedEngine:
        _EVALUATE_BATCH_HARD_CAP = 90

        async def evaluate_content_batch(
            self,
            contents: list[object],
            profile: object,
            *,
            source_context: str,
            batch_size: int,
        ) -> list[float]:
            del contents, profile, source_context, batch_size
            try:
                raise LLMRateLimitError("provider backoff: HTTP 402 insufficient balance")
            except LLMRateLimitError as exc:
                raise LLMProviderExecutionError("All providers failed") from exc

    monkeypatch.setattr(replay_script.asyncio, "sleep", fake_sleep)
    content = SimpleNamespace(
        content_id="candidate-1",
        title="candidate",
        relevance_score=0.1,
        relevance_reason="original",
    )

    with pytest.raises(LLMProviderExecutionError, match="All providers failed"):
        await _score_contents(_BillingLimitedEngine(), [content], object(), source_context="mixed")

    assert sleeps == []


def _many_interest_profile() -> SoulProfile:
    profile = SoulProfile()
    profile.preferences.interests = [
        InterestTag(name=f"兴趣{index}", category="测试", weight=1.0 - index / 1000)
        for index in range(100)
    ]
    return profile


def test_compact_replay_arm_a_forces_legacy_full_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After production becomes compact, compact-arm A must still be full-profile legacy."""

    def production_summary(profile: SoulProfile) -> dict[str, object]:
        return compact_evaluation_profile_summary(build_profile_summary(profile))

    monkeypatch.setattr(
        ContentDiscoveryEngine,
        "_evaluation_profile_summary",
        staticmethod(production_summary),
    )

    engine = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=False,
        negative_examples=None,
        legacy_profile=True,
        embedding_service=None,
    )

    summary = engine._evaluation_profile_summary(_many_interest_profile())
    interests = summary["interests"]
    assert isinstance(interests, list)
    assert len(interests) == 100


def test_replay_engine_receives_embedding_service_for_production_recall() -> None:
    embedding = _ReplayEmbedding()

    engine = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=embedding,
    )

    assert engine._embedding_service is embedding  # noqa: SLF001


def test_json_minify_replay_flag_is_instance_scoped_and_default_off() -> None:
    arm_a = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
    )
    arm_b = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        compact_evaluation_json=True,
    )

    assert arm_a.compact_evaluation_json is False
    assert arm_b.compact_evaluation_json is True


def test_candidate_transport_replay_flag_is_instance_scoped_and_default_off() -> None:
    production = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
    )
    sparse = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        evaluation_candidate_transport="sparse-json",
    )
    row = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        evaluation_candidate_transport="row-wire-v1",
    )

    assert production.evaluation_candidate_transport == "production"
    assert sparse.evaluation_candidate_transport == "sparse-json"
    assert row.evaluation_candidate_transport == "row-wire-v1"


class _MemberRepairService:
    def __init__(self) -> None:
        self.call_count = 0

    async def complete_structured_task(self, **kwargs: object) -> LLMResponse:
        del kwargs
        content_id = "item-1" if self.call_count == 0 else "item-2"
        self.call_count += 1
        return LLMResponse(
            content=json.dumps(
                {
                    "results": [
                        {
                            "content_id": content_id,
                            "score": 0.7,
                            "reason": "match",
                            "topic_group": "topic",
                            "style_key": "deep_focus",
                            "franchise_key": "",
                        }
                    ]
                }
            ),
            provider="openai",
            instance_id="gateway",
            model="model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )


@pytest.mark.asyncio
async def test_replay_engine_attributes_root_and_member_repair_calls() -> None:
    service = _DeterministicLLMService(
        _MemberRepairService(),
        service="arm_b",
        expected_compact_json=True,
    )
    engine = _build_engine(
        service,
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        compact_evaluation_json=True,
    )
    contents = _rows_to_contents(
        [
            {
                "content_id": content_id,
                "source_platform": "bilibili",
                "source_strategy": "feed",
                "title": content_id,
                "body_text": "body",
            }
            for content_id in ("item-1", "item-2")
        ]
    )

    with replay_call_attribution(
        pair_kind="treatment",
        repeat=1,
        logical_run="B",
        arm="B",
    ):
        scores = await engine.evaluate_content_batch(
            contents,
            SoulProfile(),
            source_context="mixed",
            batch_size=30,
        )

    assert scores == [0.7, 0.7]
    assert [call["request_kind"] for call in service.calls] == ["root", "repair"]
    assert [call["request_ordinal"] for call in service.calls] == [0, 1]
    assert [call["request_candidate_count"] for call in service.calls] == [2, 1]
    assert all(call["all_target_json_compact"] is True for call in service.calls)


class _LocalMemberRepairService:
    def __init__(self) -> None:
        self.call_count = 0

    async def complete_structured_task(self, **kwargs: object) -> LLMResponse:
        del kwargs
        score = 0.7 + self.call_count / 10
        self.call_count += 1
        return LLMResponse(
            content=json.dumps(
                {
                    "results": [
                        {
                            "id": "0",
                            "score": score,
                            "reason": "match",
                            "topic_group": "topic",
                            "style_key": "deep_focus",
                            "franchise_key": "",
                        }
                    ]
                }
            ),
            provider="test-provider",
            instance_id="gateway",
            model="model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )


@pytest.mark.parametrize("transport", ["sparse-json", "row-wire-v1"])
@pytest.mark.asyncio
async def test_replay_engine_audits_local_candidate_transport_and_member_repair(
    transport: str,
) -> None:
    service = _DeterministicLLMService(
        _LocalMemberRepairService(),
        service="arm_b",
        expected_candidate_transport=transport,
        candidate_transport_audit_enabled=True,
    )
    engine = _build_engine(
        service,
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        evaluation_candidate_transport=transport,
    )
    contents = _rows_to_contents(
        [
            {
                "content_id": content_id,
                "content_url": f"https://private.invalid/{content_id}",
                "source_platform": "bilibili",
                "source_strategy": "feed",
                "title": content_id,
                "body_text": "body",
            }
            for content_id in ("PRIVATE-GLOBAL-1", "PRIVATE-GLOBAL-2")
        ]
    )

    with replay_call_attribution(
        pair_kind="treatment",
        repeat=1,
        logical_run="B",
        arm="B",
    ):
        scores = await engine.evaluate_content_batch(
            contents,
            SoulProfile(),
            source_context="mixed",
            batch_size=30,
        )

    assert scores == pytest.approx([0.7, 0.8])
    assert [call["request_kind"] for call in service.calls] == ["root", "repair"]
    assert [call["request_candidate_count"] for call in service.calls] == [2, 1]
    assert all(call["candidate_transport"] == transport for call in service.calls)
    assert all(call["candidate_decode_valid"] is True for call in service.calls)
    assert all(call["candidate_local_id_coverage_complete"] is True for call in service.calls)
    assert all(call["candidate_global_identity_field_count"] == 0 for call in service.calls)
    assert all(call["candidate_url_field_count"] == 0 for call in service.calls)
    assert all(call["result_identity_contract"] == "local-id" for call in service.calls)
    assert all(call["result_local_id_binding_safe"] is True for call in service.calls)
    assert "PRIVATE" not in json.dumps(service.calls, ensure_ascii=False)


class _CandidateGateService:
    def __init__(
        self,
        *,
        transport: str,
        content_ids: list[str],
        usage: dict[str, int],
    ) -> None:
        self.transport = transport
        self.content_ids = content_ids
        self.usage = usage

    async def complete_structured_task(self, **kwargs: object) -> LLMResponse:
        del kwargs
        identifiers = (
            self.content_ids
            if self.transport == "production-json"
            else [str(index) for index in range(len(self.content_ids))]
        )
        identity_field = "content_id" if self.transport == "production-json" else "id"
        return LLMResponse(
            content=json.dumps(
                {
                    "results": [
                        {
                            identity_field: identifier,
                            "score": 0.7,
                            "reason": "stable",
                            "topic_group": "topic",
                            "style_key": "deep_focus",
                            "franchise_key": "",
                        }
                        for identifier in identifiers
                    ]
                }
            ),
            provider="test-provider",
            instance_id="test-instance",
            model="test-model",
            usage=self.usage,
        )


@pytest.mark.parametrize("experiment", ["sparse-json", "row-wire-v1"])
@pytest.mark.asyncio
async def test_candidate_transport_real_prompt_matrix_passes_independent_audit(
    experiment: str,
) -> None:
    raw_config = replay_script._CANDIDATE_TRANSPORT_EXPERIMENTS[experiment]
    arm_a_transport = str(raw_config["arm_a_transport"])
    arm_b_transport = str(raw_config["arm_b_transport"])
    content_ids = [f"PRIVATE-GLOBAL-{index}" for index in range(30)]
    rows = [
        {
            "content_id": content_id,
            "content_url": f"https://private.invalid/{content_id}",
            "source_platform": "bilibili",
            "content_type": "video",
            "source_strategy": "feed",
            "title": f"Candidate {index} with a moderately long private title",
            "author_name": "private author",
            "body_text": "private body text " * 8,
            "view_count": 1000 + index,
            "like_count": 100 + index,
        }
        for index, content_id in enumerate(content_ids)
    ]
    if experiment == "sparse-json":
        usage_a = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
        usage_b = {"prompt_tokens": 750, "completion_tokens": 150, "total_tokens": 900}
    else:
        usage_a = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
        usage_b = {"prompt_tokens": 900, "completion_tokens": 150, "total_tokens": 1050}

    service_a = _DeterministicLLMService(
        _CandidateGateService(
            transport=arm_a_transport,
            content_ids=content_ids,
            usage=usage_a,
        ),
        service="arm_a",
        expected_candidate_transport=arm_a_transport,
        candidate_transport_audit_enabled=True,
    )
    service_b = _DeterministicLLMService(
        _CandidateGateService(
            transport=arm_b_transport,
            content_ids=content_ids,
            usage=usage_b,
        ),
        service="arm_b",
        expected_candidate_transport=arm_b_transport,
        candidate_transport_audit_enabled=True,
    )
    engine_a = _build_engine(
        service_a,
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        evaluation_candidate_transport=replay_script._ENGINE_CANDIDATE_TRANSPORTS[arm_a_transport],
    )
    engine_b = _build_engine(
        service_b,
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=None,
        evaluation_candidate_transport=replay_script._ENGINE_CANDIDATE_TRANSPORTS[arm_b_transport],
    )

    async def score(
        engine: object,
        *,
        pair_kind: str,
        repeat: int,
        logical_run: str,
        arm: str,
    ) -> None:
        with replay_call_attribution(
            pair_kind=pair_kind,
            repeat=repeat,
            logical_run=logical_run,
            arm=arm,
        ):
            scores = await engine.evaluate_content_batch(
                _rows_to_contents(rows),
                SoulProfile(),
                source_context="mixed",
                batch_size=30,
            )
        assert len(scores) == len(rows)
        assert scores[:8] == pytest.approx([0.7] * 8)

    for repeat in range(1, 4):
        await score(
            engine_a,
            pair_kind="control",
            repeat=repeat,
            logical_run="A1",
            arm="A",
        )
        await score(
            engine_a,
            pair_kind="control",
            repeat=repeat,
            logical_run="A2",
            arm="A",
        )
        await score(
            engine_a,
            pair_kind="treatment",
            repeat=repeat,
            logical_run="A",
            arm="A",
        )
        await score(
            engine_b,
            pair_kind="treatment",
            repeat=repeat,
            logical_run="B",
            arm="B",
        )

    audit = validate_candidate_transport_experiment(
        [*service_a.calls, *service_b.calls],
        experiment=experiment,
        repeats=3,
    )

    assert audit["passed"] is True, audit["blocking_reasons"]
    assert audit["repair"]["passed"] is True
    assert audit["classification"]["passed"] is True
    assert "PRIVATE" not in json.dumps(audit, ensure_ascii=False)


def test_compact_arm_b_uses_exact_production_profile_view() -> None:
    profile = _many_interest_profile()
    engine = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=_ReplayEmbedding(),
    )

    assert engine._evaluation_profile_summary(  # noqa: SLF001
        profile
    ) == ContentDiscoveryEngine(llm_service=None)._evaluation_profile_summary(profile)  # noqa: SLF001


@pytest.mark.asyncio
async def test_replay_engine_audits_current_batch_recall_result_path() -> None:
    embedding = _ReplayEmbedding()
    recall = ReplayRecallAudit()
    engine = _build_engine(
        object(),
        _ReplayConfig(),
        compact_profile=True,
        negative_examples=None,
        legacy_profile=False,
        embedding_service=embedding,
        recall_audit=recall,
    )
    content = _rows_to_contents(
        [
            {
                "content_id": "item-1",
                "source_platform": "twitter",
                "source_strategy": "feed",
                "title": "matching content",
                "body_text": "matching body",
            }
        ]
    )[0]

    with replay_call_attribution(
        pair_kind="treatment",
        repeat=1,
        logical_run="B",
        arm="B",
    ):
        result = await engine._related_interests_for_batch_result(  # noqa: SLF001
            [content],
            _many_interest_profile(),
        )

    assert result.complete_indices == frozenset({0})
    assert recall.events[0]["complete_candidate_count"] == 1
    assert recall.events[0]["logical_run"] == "B"


def test_replay_report_mentions_when_compact_recall_is_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_report(
        arm_b="compact",
        candidates=[ReplayCandidate(candidate_id=1, title="item", source_strategy="search")],
        scores_a=[0.7],
        scores_b=[0.7],
        platform=None,
        recall_note="related_interests recall disabled: embedding service unavailable",
    )

    output = capsys.readouterr().out
    assert "related_interests recall disabled: embedding service unavailable" in output


def test_legacy_reason_prompts_swaps_and_restores() -> None:
    """reason-diet arm A must really restore the legacy prompts, then undo."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_profile_diet_ab as script

    from openbiliclaw.llm import prompts as prompts_module

    before_single = prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
    before_batch = prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
    with script.legacy_reason_prompts():
        assert "只写一句中文" in prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
        assert "3a. reason" not in prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
        assert "reason(一句中文)" in prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
    assert before_single == prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
    assert before_batch == prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT


def test_reason_off_prompts_remove_output_fields_and_restore_production() -> None:
    from openbiliclaw.llm import prompts as prompts_module

    before_single = prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
    before_batch = prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
    assert '"reason":' in before_single
    assert '"reason":' in before_batch

    with reason_off_prompts():
        single = prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
        batch = prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
        assert "输出中禁止包含 reason 字段" in single
        assert "每个条目都禁止包含 reason 字段" in batch
        assert '"reason":' not in single
        assert '"reason":' not in batch
        assert "、reason、topic_group" not in batch

    assert before_single == prompts_module._SINGLE_CONTENT_EVALUATION_SYSTEM_PROMPT
    assert before_batch == prompts_module._BATCH_CONTENT_EVALUATION_SYSTEM_PROMPT
