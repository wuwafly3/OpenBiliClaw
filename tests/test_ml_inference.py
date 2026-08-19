"""Cheap-tag feature isolation and numpy admission inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbiliclaw.discovery.engine import ContentDiscoveryEngine, DiscoveredContent
from openbiliclaw.discovery.tag_channel import TAG_CHANNEL_SOURCE_LLM
from openbiliclaw.ml.features import FEATURE_VERSION, encode_features, feature_record_from_content
from openbiliclaw.ml.inference import (
    AdmissionModel,
    AdmissionModelError,
    default_model_path,
    resolve_model_path,
)

pytest.importorskip("numpy")


def _content(**overrides: object) -> DiscoveredContent:
    payload: dict[str, object] = {
        "bvid": "BV1TEST",
        "title": "hello",
        "source_platform": "bilibili",
        "source_strategy": "search",
        "content_type": "video",
        "style_key": "deep_focus",
        "temporal_class": "evergreen",
        "topic_group": "教师主题",
        "tag_channel_style_key": "quick_scan",
        "tag_channel_temporal_class": "current",
        "tag_channel_topic_group": "廉价主题",
        "tag_channel_source": TAG_CHANNEL_SOURCE_LLM,
    }
    payload.update(overrides)
    return DiscoveredContent(**payload)  # type: ignore[arg-type]


def test_cheap_records_ignore_teacher_tags() -> None:
    content = _content()
    cheap = feature_record_from_content(content, tag_source="cheap")
    teacher = feature_record_from_content(content, tag_source="teacher_oracle")
    assert cheap["style"] == "quick_scan"
    assert cheap["topic"] == "廉价主题"
    assert teacher["style"] == "deep_focus"
    assert teacher["topic"] == "教师主题"
    names = encode_features([cheap])[1]
    joined = " ".join(names)
    assert "franchise" not in joined
    assert "relevance_score" not in joined
    assert "teacher_score" not in joined


def _toy_payload(*, intercept: float = 0.0, threshold: float = 0.9) -> dict[str, object]:
    names = ["sim", "sim_available"]
    return {
        "feature_version": FEATURE_VERSION,
        "tags_source": "teacher_oracle",
        "feature_names": names,
        "vocab": {
            "platforms": ["bilibili"],
            "content_types": ["video"],
            "strategies": ["search"],
            "contexts": ["no_audit"],
            "topics": ["廉价主题"],
        },
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "coef": [0.0, 0.0],
        "intercept": intercept,
        "isotonic_x": [0.0, 1.0],
        "isotonic_y": [0.0, 1.0],
        "decision_threshold": threshold,
    }


def test_admission_model_scores_and_rejects_version_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")
    model = AdmissionModel.load(path)
    result = model.predict(
        [
            {
                "sim": 0.1,
                "platform": "bilibili",
                "strategy": "search",
                "content_type": "video",
                "style": "quick_scan",
                "temporal": "current",
                "topic": "廉价主题",
                "title_len": 1,
                "desc_len": 0,
                "body_len": 0,
                "duration_s": 0,
                "rating_score": 0,
                "source_rank": 0,
            }
        ]
    )
    assert result.ok
    assert result.predictions[0] is not None
    assert result.predictions[0].admitted is False
    assert 0.4 < result.predictions[0].probability < 0.6

    bad = _toy_payload()
    bad["feature_version"] = "nope"
    with pytest.raises(AdmissionModelError, match="feature_version"):
        AdmissionModel.from_payload(bad)


def test_score_admission_batch_fail_open_without_cheap_tags(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")
    engine = ContentDiscoveryEngine(
        relevance_scorer="shadow",
        relevance_model_path=str(path),
    )
    missing = _content(tag_channel_source="")
    tagged = _content()
    engine.score_admission_batch([missing, tagged])
    assert missing.ml_admission_status == "fail_open"
    assert missing.ml_admission_reason == "missing_cheap_tags"
    assert tagged.ml_admission_status == "scored"
    assert tagged.ml_admission_admitted is False
    assert tagged.relevance_score == 0.0


def test_score_admission_batch_llm_mode_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")
    engine = ContentDiscoveryEngine(
        relevance_scorer="llm",
        relevance_model_path=str(path),
    )
    content = _content()
    engine.score_admission_batch([content])
    assert content.ml_admission_status == ""
    assert content.ml_admission_proba is None


def test_score_admission_batch_missing_artifact_fail_open(tmp_path: Path) -> None:
    engine = ContentDiscoveryEngine(
        relevance_scorer="shadow",
        relevance_model_path=str(tmp_path / "missing.json"),
    )
    content = _content()
    engine.score_admission_batch([content])
    assert content.ml_admission_status == "fail_open"
    assert "not found" in content.ml_admission_reason
    assert content.relevance_score == 0.0


def test_resolve_model_path_defaults_under_data_dir(tmp_path: Path) -> None:
    expected = Path(tmp_path) / "ml_artifacts" / "admission_teacher_v1.json"
    assert default_model_path(tmp_path) == expected
    assert resolve_model_path("", tmp_path) == str(expected)
    assert resolve_model_path("C:/custom.json", tmp_path) == "C:/custom.json"


def test_admission_model_high_logit_admits(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload(intercept=10.0, threshold=0.5)), encoding="utf-8")
    model = AdmissionModel.load(path)
    result = model.predict(
        [
            {
                "sim": 0.1,
                "platform": "bilibili",
                "strategy": "search",
                "content_type": "video",
                "style": "quick_scan",
                "temporal": "current",
                "topic": "廉价主题",
                "title_len": 1,
                "desc_len": 0,
                "body_len": 0,
                "duration_s": 0,
                "rating_score": 0,
                "source_rank": 0,
            }
        ]
    )
    assert result.ok
    assert result.predictions[0] is not None
    assert result.predictions[0].admitted is True


def test_score_admission_batch_version_mismatch_fail_open(tmp_path: Path) -> None:
    payload = _toy_payload()
    payload["feature_version"] = "nope"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    engine = ContentDiscoveryEngine(
        relevance_scorer="shadow",
        relevance_model_path=str(path),
    )
    content = _content()
    engine.score_admission_batch([content])
    assert content.ml_admission_status == "fail_open"
    assert "feature_version" in content.ml_admission_reason
    assert content.relevance_score == 0.0


def test_ml_mode_warns_observational_and_does_not_write_relevance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")
    engine = ContentDiscoveryEngine(
        relevance_scorer="ml",
        relevance_model_path=str(path),
    )
    content = _content()
    with caplog.at_level("WARNING"):
        engine.score_admission_batch([content])
        engine.score_admission_batch([content])
    warnings = [record.message for record in caplog.records if "observational" in record.message]
    assert len(warnings) == 1
    assert content.ml_admission_status == "scored"
    assert content.relevance_score == 0.0


@pytest.mark.asyncio
async def test_shadow_scoring_does_not_skip_llm(tmp_path: Path) -> None:
    from .test_search_strategy import FakeLLMService, _build_profile

    path = tmp_path / "model.json"
    path.write_text(json.dumps(_toy_payload()), encoding="utf-8")
    llm_service = FakeLLMService(
        '{"score": 0.82, "reason": "匹配", "topic_group": "摄影", "style_key": "quick_scan"}'
    )
    engine = ContentDiscoveryEngine(
        llm_service=llm_service,
        relevance_scorer="shadow",
        relevance_model_path=str(path),
        eval_prefilter_mode="off",
    )
    content = _content()
    score = await engine.evaluate_content(content, _build_profile())
    assert llm_service.calls
    assert score == pytest.approx(0.82)
    assert content.relevance_score == pytest.approx(0.82)
    assert content.ml_admission_status == "scored"
    assert content.ml_admission_admitted is False
