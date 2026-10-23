"""Classification metrics implemented from first principles (NumPy only).

Conventions (documented because they matter for honest reporting):

* Rows of the confusion matrix are TRUE classes, columns are PREDICTED classes, both in the
  fixed class-mapping order.
* ``support`` = number of true samples of the class.
* ``recall`` is undefined (None) when support == 0; ``precision`` is undefined (None) when the
  class was never predicted.
* ``f1 = 2TP / (2TP + FP + FN)``. It is reported as undefined (None) for a class with zero
  support, because a class absent from the evaluated samples has no meaningful estimate.
  A class with support > 0 that is never predicted has F1 = 0.0 (all of it was missed).
* ``macro_f1`` is the unweighted mean of F1 over *evaluated* classes (support > 0). Classes
  without support are listed in ``classes_without_support`` and excluded from the mean.
* ``accuracy`` = correct / total samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import EvaluationError


@dataclass
class ClassMetrics:
    class_name: str
    support: int
    predicted_count: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "support": self.support,
                "predicted_count": self.predicted_count, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


@dataclass
class ClassificationMetrics:
    sample_count: int
    accuracy: float
    macro_f1: float | None
    balanced_accuracy: float | None
    per_class: list[ClassMetrics]
    confusion_matrix: list[list[int]]
    class_names: list[str]
    evaluated_class_count: int
    classes_without_support: list[str]
    classes_never_predicted: list[str]

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "evaluated_class_count": self.evaluated_class_count,
            "classes_without_support": self.classes_without_support,
            "classes_never_predicted": self.classes_never_predicted,
            "per_class": [c.to_dict() for c in self.per_class],
            "confusion_matrix": {"orientation": "rows=true classes, columns=predicted classes",
                                 "class_order": self.class_names,
                                 "matrix": self.confusion_matrix},
        }


def _validate(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int):
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    if t.ndim != 1 or p.ndim != 1:
        raise EvaluationError("Labels and predictions must be 1-D sequences.")
    if t.size == 0:
        raise EvaluationError("Cannot compute metrics on an empty set of samples.")
    if t.shape != p.shape:
        raise EvaluationError(
            f"Labels and predictions differ in length ({t.size} vs {p.size}).")
    if not (np.issubdtype(t.dtype, np.integer) and np.issubdtype(p.dtype, np.integer)):
        raise EvaluationError("Labels and predictions must be integer class indices.")
    if t.min() < 0 or p.min() < 0 or t.max() >= num_classes or p.max() >= num_classes:
        raise EvaluationError(
            f"Class indices must be in 0..{num_classes - 1}.")
    return t, p


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    t, p = _validate(y_true, y_pred, num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (t, p), 1)
    return matrix


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                                   class_names: Sequence[str]) -> ClassificationMetrics:
    num_classes = len(class_names)
    if num_classes < 2:
        raise EvaluationError("At least two classes are required.")
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    tp = np.diag(matrix).astype(np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    per_class: list[ClassMetrics] = []
    f1_values: list[float] = []
    recalls: list[float] = []
    for c, name in enumerate(class_names):
        precision = float(tp[c] / predicted[c]) if predicted[c] > 0 else None
        recall = float(tp[c] / support[c]) if support[c] > 0 else None
        denominator = 2 * tp[c] + (predicted[c] - tp[c]) + (support[c] - tp[c])
        f1 = float(2 * tp[c] / denominator) if (support[c] > 0 and denominator > 0) else None
        if f1 is not None:
            f1_values.append(f1)
        if recall is not None:
            recalls.append(recall)
        per_class.append(ClassMetrics(str(name), int(support[c]), int(predicted[c]),
                                      precision, recall, f1))

    return ClassificationMetrics(
        sample_count=total,
        accuracy=float(tp.sum() / total),
        macro_f1=float(np.mean(f1_values)) if f1_values else None,
        balanced_accuracy=float(np.mean(recalls)) if recalls else None,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        class_names=[str(n) for n in class_names],
        evaluated_class_count=len(f1_values),
        classes_without_support=[str(n) for c, n in enumerate(class_names) if support[c] == 0],
        classes_never_predicted=[str(n) for c, n in enumerate(class_names) if predicted[c] == 0],
    )
