"""Pure numpy admission forward pass (spec S1.3 / S1.7).

sklearn is training-only. A missing artifact, version mismatch, or encode
error fails open — the caller must keep the LLM path and never invent a
default admit/reject.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from openbiliclaw.ml.features import FEATURE_VERSION, encode_features

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_ARTIFACT_NAME = "admission_teacher_v1.json"


class AdmissionModelError(RuntimeError):
    """Raised when an artifact cannot be used. Callers fail open."""


@dataclass(frozen=True)
class AdmissionPrediction:
    probability: float
    admitted: bool


@dataclass(frozen=True)
class ScoreBatchResult:
    ok: bool
    reason: str
    predictions: tuple[AdmissionPrediction | None, ...]


def default_model_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "ml_artifacts" / DEFAULT_ARTIFACT_NAME


def resolve_model_path(configured: object, data_dir: str | Path) -> str:
    raw = str(configured or "").strip()
    if raw:
        return raw
    return str(default_model_path(data_dir))


def _as_float_array(values: object, *, name: str, size: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) != size:
        raise AdmissionModelError(f"artifact {name} must be a list of length {size}")
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise AdmissionModelError(f"artifact {name} contains non-finite values")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class AdmissionModel:
    """Versioned logistic + isotonic admission scorer."""

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        vocab: dict[str, list[str]],
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        iso_x: np.ndarray,
        iso_y: np.ndarray,
        threshold: float,
        feature_version: str,
        tags_source: str,
    ) -> None:
        self.feature_names = feature_names
        self.vocab = vocab
        self.mean = mean
        self.scale = scale
        self.coef = coef
        self.intercept = intercept
        self.iso_x = iso_x
        self.iso_y = iso_y
        self.threshold = threshold
        self.feature_version = feature_version
        self.tags_source = tags_source

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdmissionModel:
        version = str(payload.get("feature_version") or "")
        if version != FEATURE_VERSION:
            raise AdmissionModelError(
                f"artifact feature_version {version!r} != {FEATURE_VERSION!r}"
            )
        names_raw = payload.get("feature_names")
        if not isinstance(names_raw, list) or not names_raw:
            raise AdmissionModelError("artifact feature_names missing")
        names = tuple(str(item) for item in names_raw)
        vocab_raw = payload.get("vocab")
        if not isinstance(vocab_raw, dict):
            raise AdmissionModelError("artifact vocab missing")
        vocab = {
            str(key): [str(item) for item in value]
            for key, value in vocab_raw.items()
            if isinstance(value, list)
        }
        for required in ("platforms", "content_types", "strategies", "contexts", "topics"):
            if required not in vocab:
                raise AdmissionModelError(f"artifact vocab.{required} missing")
        size = len(names)
        mean = _as_float_array(payload.get("scaler_mean"), name="scaler_mean", size=size)
        scale = _as_float_array(payload.get("scaler_scale"), name="scaler_scale", size=size)
        coef = _as_float_array(payload.get("coef"), name="coef", size=size)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        intercept = float(payload.get("intercept", 0.0))
        if not math.isfinite(intercept):
            raise AdmissionModelError("artifact intercept is not finite")
        iso_x = np.asarray(payload.get("isotonic_x") or [], dtype=float)
        iso_y = np.asarray(payload.get("isotonic_y") or [], dtype=float)
        if iso_x.size == 0 or iso_x.shape != iso_y.shape or not np.isfinite(iso_x).all():
            raise AdmissionModelError("artifact isotonic map is invalid")
        if not np.isfinite(iso_y).all():
            raise AdmissionModelError("artifact isotonic map is invalid")
        threshold = float(payload.get("decision_threshold", 0.5))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise AdmissionModelError("artifact decision_threshold out of range")
        return cls(
            feature_names=names,
            vocab=vocab,
            mean=mean,
            scale=scale,
            coef=coef,
            intercept=intercept,
            iso_x=iso_x,
            iso_y=iso_y,
            threshold=threshold,
            feature_version=version,
            tags_source=str(payload.get("tags_source") or ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> AdmissionModel:
        artifact = Path(path)
        if not artifact.is_file():
            raise AdmissionModelError(f"admission artifact not found: {artifact}")
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdmissionModelError(f"admission artifact unreadable: {exc}") from exc
        if not isinstance(payload, dict):
            raise AdmissionModelError("admission artifact is not an object")
        return cls.from_payload(payload)

    def predict(self, records: Sequence[Mapping[str, Any]]) -> ScoreBatchResult:
        if not records:
            return ScoreBatchResult(ok=True, reason="", predictions=())
        try:
            matrix, names, _vocab = encode_features(
                records,
                vocab=self.vocab,
                feature_names=self.feature_names,
            )
        except Exception as exc:
            return ScoreBatchResult(
                ok=False,
                reason=f"encode_failed:{exc}",
                predictions=tuple(None for _ in records),
            )
        if names != list(self.feature_names) or matrix.shape != (
            len(records),
            len(self.feature_names),
        ):
            return ScoreBatchResult(
                ok=False,
                reason="feature_shape_mismatch",
                predictions=tuple(None for _ in records),
            )
        try:
            scaled = (matrix - self.mean) / self.scale
            logits = scaled @ self.coef + self.intercept
            raw = _sigmoid(logits)
            calibrated = np.interp(raw, self.iso_x, self.iso_y)
            calibrated = np.clip(calibrated, 0.0, 1.0)
        except Exception as exc:
            return ScoreBatchResult(
                ok=False,
                reason=f"forward_failed:{exc}",
                predictions=tuple(None for _ in records),
            )
        predictions: list[AdmissionPrediction] = []
        for probability in calibrated.tolist():
            value = float(probability)
            if not math.isfinite(value):
                return ScoreBatchResult(
                    ok=False,
                    reason="non_finite_probability",
                    predictions=tuple(None for _ in records),
                )
            predictions.append(
                AdmissionPrediction(probability=value, admitted=value >= self.threshold)
            )
        return ScoreBatchResult(ok=True, reason="", predictions=tuple(predictions))
