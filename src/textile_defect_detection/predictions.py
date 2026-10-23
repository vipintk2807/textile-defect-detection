"""Prediction contract: structured results independent of the user interface."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import InferenceError
from .provenance import utc_now_iso

STATUS_COMPLETED = "completed"
STATUS_LOW_CONFIDENCE = "completed_low_confidence"
STATUS_FAILED = "failed"
ALL_STATUSES = (STATUS_COMPLETED, STATUS_LOW_CONFIDENCE, STATUS_FAILED)

STATUS_LABELS = {
    STATUS_COMPLETED: "Prediction generated",
    STATUS_LOW_CONFIDENCE: "Prediction generated - below confidence threshold, review required",
    STATUS_FAILED: "Inference failed",
}


@dataclass(frozen=True)
class ClassScore:
    class_index: int
    class_name: str
    score: float


@dataclass(frozen=True)
class Prediction:
    """Single-label classification output. Scores are softmax values, not calibrated
    probabilities."""

    predicted_index: int
    predicted_class: str
    score: float
    ranked: tuple[ClassScore, ...]
    scoring: str = "softmax"

    def to_dict(self) -> dict:
        return {
            "scoring": self.scoring,
            "predicted_index": self.predicted_index,
            "predicted_class": self.predicted_class,
            "score": self.score,
            "ranked": [
                {"class_index": c.class_index, "class_name": c.class_name, "score": c.score}
                for c in self.ranked
            ],
        }


@dataclass(frozen=True)
class InspectionResult:
    """One completed inspection (prediction contract, technical specification section 7)."""

    inspection_id: str
    created_at: str
    image_name: str
    model_id: str
    task_type: str
    status: str
    prediction: Prediction
    threshold: float
    processing_time_ms: float
    image_width: int
    image_height: int

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_LOW_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "inspection_id": self.inspection_id,
            "created_at": self.created_at,
            "image_name": self.image_name,
            "model_id": self.model_id,
            "task_type": self.task_type,
            "status": self.status,
            "predictions": self.prediction.to_dict(),
            "threshold": self.threshold,
            "processing_time_ms": self.processing_time_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


def new_inspection_id() -> str:
    return uuid.uuid4().hex


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector of finite logits."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise InferenceError("The model produced output with an unexpected shape.")
    if not np.all(np.isfinite(x)):
        raise InferenceError("The model produced invalid (non-finite) output values.")
    shifted = x - x.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def decide_status(top_score: float, threshold: float) -> str:
    return STATUS_COMPLETED if top_score >= threshold else STATUS_LOW_CONFIDENCE


def logits_to_prediction(logits: np.ndarray, class_names: Sequence[str],
                         top_k: int | None = None) -> Prediction:
    """Convert raw model logits to a :class:`Prediction` using the verified class mapping."""
    arr = np.asarray(logits)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != len(class_names):
        raise InferenceError(
            "The model output does not match the configured class mapping "
            f"(expected {len(class_names)} scores, got shape {tuple(arr.shape)}).")
    scores = softmax(arr)
    order = np.argsort(-scores, kind="stable")
    ranked = tuple(ClassScore(int(i), str(class_names[int(i)]), float(scores[int(i)]))
                   for i in order)
    top = ranked[0]
    if top_k is not None:
        ranked = ranked[:top_k]
    return Prediction(top.class_index, top.class_name, top.score, ranked)


def build_result(prediction: Prediction, *, image_name: str, model_id: str, threshold: float,
                 processing_time_ms: float, image_width: int,
                 image_height: int) -> InspectionResult:
    return InspectionResult(
        inspection_id=new_inspection_id(),
        created_at=utc_now_iso(),
        image_name=image_name,
        model_id=model_id,
        task_type="classification",
        status=decide_status(prediction.score, threshold),
        prediction=prediction,
        threshold=threshold,
        processing_time_ms=processing_time_ms,
        image_width=image_width,
        image_height=image_height,
    )
