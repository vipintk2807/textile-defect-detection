"""Held-out evaluation and traceable report generation.

``run_evaluation`` takes a *predictor* callable (see ``inference.Predictor``) so the whole
evaluation logic is independent of PyTorch. The report is generated only from the actual
labels and predictions of the evaluated manifest rows; nothing is filled in by hand.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import Sample, resolve_sample_path
from .errors import EvaluationError, InferenceError
from .image_processing import load_image_file, preprocess_image
from .metrics import ClassificationMetrics, compute_classification_metrics
from .model_metadata import ModelMetadata
from .predictions import softmax
from .provenance import environment_info, git_revision, sha256_file, utc_now_iso

logger = logging.getLogger(__name__)
REPORT_SCHEMA_VERSION = 1
MAX_LISTED_ERRORS = 50


def _fmt(value: float | None, digits: int = 4) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    m = report["metrics"]
    lines = [f"# Evaluation report - {report['evaluation_id']}", ""]
    if report["dataset"].get("synthetic"):
        lines += ["> **SYNTHETIC DATA.** This evaluation used generated test images. It verifies "
                  "software behaviour only and says nothing about real textile defect "
                  "recognition.", ""]
    for warning in report["warnings"]:
        lines.append(f"> **Warning:** {warning}")
    if report["warnings"]:
        lines.append("")
    lines += [
        "## Identity", "",
        f"- Generated at: {report['generated_at']}",
        f"- Model id: `{report['model']['model_id']}` (weights sha256 `{report['model']['weights_sha256'][:16]}...`)",
        f"- Split evaluated: `{report['split']}` ({report['sample_count']} samples)",
        f"- Dataset: {report['dataset'].get('dataset_name', 'unknown')} "
        f"(version {report['dataset'].get('version', 'unknown')}, "
        f"manifest sha256 `{str(report['dataset'].get('manifest_sha256', ''))[:16]}...`)",
        f"- Evaluations of this split recorded before this one: "
        f"{report['previous_evaluations_of_split']}",
        f"- Source revision: {report['git_revision'] or 'unavailable'}",
        "", "## Overall", "",
        f"- Accuracy: {_fmt(m['accuracy'])}",
        f"- Macro F1 (over {m['evaluated_class_count']} evaluated classes): {_fmt(m['macro_f1'])}",
        f"- Balanced accuracy: {_fmt(m['balanced_accuracy'])}",
        "", "## Per-class results", "",
        "| class | support | precision | recall | F1 |", "|---|---|---|---|---|",
    ]
    for c in m["per_class"]:
        lines.append(f"| {c['class_name']} | {c['support']} | {_fmt(c['precision'])} | "
                     f"{_fmt(c['recall'])} | {_fmt(c['f1'])} |")
    if m["classes_without_support"]:
        lines += ["", f"Classes with no true samples in this split (metrics unavailable): "
                  f"{', '.join(m['classes_without_support'])}."]
    if m["classes_never_predicted"]:
        lines += ["", f"Classes the model never predicted (precision undefined): "
                  f"{', '.join(m['classes_never_predicted'])}."]
    names = m["confusion_matrix"]["class_order"]
    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "",
              "| true \\ predicted | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for name, row in zip(names, m["confusion_matrix"]["matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    ea = report["error_analysis"]
    lines += ["", "## Error analysis (from actual predictions)", "",
              f"- Misclassified samples: {ea['misclassified_count']} of {report['sample_count']}"]
    if ea["top_confusions"]:
        lines += ["- Most frequent confusions (true -> predicted): " + "; ".join(
            f"{c['true']} -> {c['predicted']} ({c['count']})" for c in ea["top_confusions"])]
    if ea["misclassified"]:
        lines += ["", "| sample | true | predicted | top score |", "|---|---|---|---|"]
        for e in ea["misclassified"][:20]:
            lines.append(f"| {e['sample_id']} | {e['true']} | {e['predicted']} | "
                         f"{e['top_score']:.3f} |")
    lines += ["", "Causes of errors are hypotheses until the images are inspected.", "",
              "## What this does and does not establish", "",
              "- It measures this checkpoint on this split of this dataset only.",
              "- It does not establish performance on other fabrics, cameras, lighting or on a "
              "production line, and softmax scores are not calibrated probabilities.", ""]
    return "\n".join(lines)


def _count_previous(parent: Path, split: str) -> int:
    return len(list(parent.glob(f"{split}-*"))) if parent.is_dir() else 0


def run_evaluation(predictor: Callable[[np.ndarray], np.ndarray], metadata: ModelMetadata,
                   samples: list[Sample], images_root: str | Path, *, split: str,
                   evaluations_dir: str | Path, manifest_path: str | Path | None = None,
                   batch_size: int = 16, dataset_info: dict | None = None,
                   weights_path: str | Path | None = None) -> dict:
    """Evaluate ``predictor`` on the manifest rows of ``split`` and write the report files.

    Output folder: ``<evaluations_dir>/<split>-<timestamp>/`` (never overwritten). Returns the
    report dictionary. Missing/corrupt images abort the run instead of being skipped silently.
    """
    chosen = [s for s in samples if s.split == split]
    if not chosen:
        raise EvaluationError(f"The manifest has no samples in split '{split}'.")
    class_names = list(metadata.class_names)
    index_of = {n: i for i, n in enumerate(class_names)}
    unknown = sorted({s.class_label for s in chosen} - set(index_of))
    if unknown:
        raise EvaluationError(
            "The manifest contains labels that the model's class mapping does not know: "
            f"{', '.join(unknown)}.")

    evaluations_dir = Path(evaluations_dir)
    previous = _count_previous(evaluations_dir, split)
    warnings: list[str] = []
    if split != "test":
        warnings.append(f"Split '{split}' is used for training or model selection; these numbers "
                        "are NOT an independent held-out estimate.")
    elif previous > 0:
        warnings.append(f"The test split has already been evaluated {previous} time(s) for this "
                        "experiment. Repeated inspection weakens its independence; do not tune "
                        "the model against it.")

    y_true: list[int] = []
    y_pred: list[int] = []
    top_scores: list[float] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(chosen), batch_size):
        batch_samples = chosen[start:start + batch_size]
        arrays = []
        for s in batch_samples:
            try:
                path = resolve_sample_path(images_root, s.relative_image_path)
                loaded = load_image_file(path)
            except Exception as exc:  # noqa: BLE001
                msg = getattr(exc, "user_message", "image could not be loaded")
                raise EvaluationError(
                    f"Sample '{s.sample_id}' could not be loaded for evaluation: {msg}") from exc
            arrays.append(preprocess_image(loaded.image, metadata.preprocessing))
        try:
            logits = np.asarray(predictor(np.stack(arrays, axis=0)))
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError("The model failed while evaluating a batch.",
                                  detail=repr(exc)) from exc
        if logits.shape != (len(batch_samples), len(class_names)):
            raise EvaluationError(
                f"Unexpected model output shape {logits.shape}; expected "
                f"({len(batch_samples)}, {len(class_names)}).")
        for s, row in zip(batch_samples, logits):
            try:
                probs = softmax(row)
            except InferenceError as exc:
                raise EvaluationError(
                    f"Invalid model output for sample '{s.sample_id}'.") from exc
            y_true.append(index_of[s.class_label])
            y_pred.append(int(np.argmax(probs)))
            top_scores.append(float(probs.max()))
            all_scores.append(probs)

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    report = _build_report(metrics, metadata, chosen, y_true, y_pred, top_scores, split,
                           previous, warnings, manifest_path, dataset_info or {}, weights_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = evaluations_dir / f"{split}-{stamp}"
    suffix = 1
    while out_dir.exists():
        suffix += 1
        out_dir = evaluations_dir / f"{split}-{stamp}-{suffix}"
    out_dir.mkdir(parents=True)
    (out_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "evaluation_report.md").write_text(render_markdown(report), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "true_label", "predicted_label", "top_score"]
                        + [f"score_{n}" for n in class_names])
        for s, t, p, top, probs in zip(chosen, y_true, y_pred, top_scores, all_scores):
            writer.writerow([s.sample_id, class_names[t], class_names[p], f"{top:.6f}"]
                            + [f"{v:.6f}" for v in probs])
    report["output_directory"] = str(out_dir)
    return report


def _build_report(metrics: ClassificationMetrics, metadata: ModelMetadata, chosen: list[Sample],
                  y_true: list[int], y_pred: list[int], top_scores: list[float], split: str,
                  previous: int, warnings: list[str], manifest_path, dataset_info: dict,
                  weights_path) -> dict:
    names = list(metadata.class_names)
    wrong = [(s, t, p, sc) for s, t, p, sc in zip(chosen, y_true, y_pred, top_scores) if t != p]
    wrong.sort(key=lambda item: -item[3])  # high-confidence mistakes first
    confusions = Counter((names[t], names[p]) for _, t, p, _ in wrong)
    if metrics.classes_without_support:
        warnings.append("Some classes have no samples in this split; their metrics are "
                        "unavailable: " + ", ".join(metrics.classes_without_support) + ".")
    dataset = dict(metadata.dataset)
    dataset.update({k: v for k, v in dataset_info.items() if k not in dataset or not dataset[k]})
    if manifest_path and Path(manifest_path).is_file():
        dataset["manifest_sha256"] = sha256_file(manifest_path)
    model = {"model_id": metadata.model_id, "architecture": metadata.architecture,
             "weights_sha256": metadata.weights_sha256,
             "selected_epoch": metadata.training.get("best_epoch"),
             "training_experiment_id": metadata.training.get("experiment_id")}
    if weights_path and Path(weights_path).is_file():
        actual = sha256_file(weights_path)
        model["evaluated_weights_sha256"] = actual
        if actual != metadata.weights_sha256:
            raise EvaluationError("The weights file does not match the metadata checksum.")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_id": f"{metadata.model_id}-{split}",
        "generated_at": utc_now_iso(),
        "split": split,
        "sample_count": len(chosen),
        "previous_evaluations_of_split": previous,
        "warnings": warnings,
        "model": model,
        "dataset": dataset,
        "class_names": names,
        "preprocessing": metadata.preprocessing.to_dict(),
        "metrics": metrics.to_dict(),
        "error_analysis": {
            "misclassified_count": len(wrong),
            "top_confusions": [{"true": a, "predicted": b, "count": n}
                               for (a, b), n in confusions.most_common(5)],
            "misclassified": [{"sample_id": s.sample_id, "true": names[t],
                               "predicted": names[p], "top_score": sc}
                              for s, t, p, sc in wrong[:MAX_LISTED_ERRORS]],
        },
        "environment": environment_info(),
        "git_revision": git_revision(),
    }
