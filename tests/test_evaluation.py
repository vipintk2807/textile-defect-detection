import json

import numpy as np
import pytest

from helpers import SMALL_PREPROCESSING, prepared_synthetic_dataset
from textile_defect_detection.errors import EvaluationError
from textile_defect_detection.evaluation import run_evaluation
from textile_defect_detection.model_metadata import ModelMetadata
from textile_defect_detection.synthetic import SYNTHETIC_CLASSES


def metadata(synthetic=True, classes=SYNTHETIC_CLASSES):
    return ModelMetadata(model_id="eval-test", class_names=tuple(sorted(classes)),
                         preprocessing=SMALL_PREPROCESSING, weights_sha256="1" * 64,
                         dataset={"dataset_name": "SYNTHETIC", "synthetic": synthetic})


def oracle(samples, meta, root, flip=False):
    """Stand-in predictor that 'knows' labels (by looking at file order) - verifies the
    evaluation arithmetic only. Implemented via a lookup queue of expected labels."""
    queue = [meta.class_names.index(s.class_label) for s in samples]
    pos = {"i": 0}

    def predict(batch):
        n = batch.shape[0]
        out = np.zeros((n, len(meta.class_names)), dtype=np.float32)
        for k in range(n):
            truth = queue[pos["i"]]
            pos["i"] += 1
            guess = (truth + 1) % len(meta.class_names) if flip else truth
            out[k, guess] = 6.0
        return out

    return predict


def test_perfect_predictor_gives_perfect_metrics_and_traceable_report(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    test_rows = [s for s in samples if s.split == "test"]
    report = run_evaluation(oracle(test_rows, meta, root), meta, samples, root, split="test",
                            evaluations_dir=tmp_path / "evals",
                            manifest_path=tmp_path / "proc" / "samples.csv", batch_size=5)
    assert report["sample_count"] == len(test_rows) == 12
    assert report["metrics"]["accuracy"] == 1.0 and report["metrics"]["macro_f1"] == 1.0
    assert report["dataset"]["synthetic"] is True and len(report["dataset"]["manifest_sha256"]) == 64
    assert report["model"]["model_id"] == "eval-test" and report["split"] == "test"
    out = tmp_path / "evals"
    (folder,) = list(out.iterdir())
    assert {p.name for p in folder.iterdir()} == {"evaluation_report.json", "evaluation_report.md",
                                                  "predictions.csv"}
    md = (folder / "evaluation_report.md").read_text()
    assert "SYNTHETIC DATA" in md and "does not establish" in md
    assert json.loads((folder / "evaluation_report.json").read_text())["metrics"]["accuracy"] == 1.0


def test_wrong_predictor_gives_zero_accuracy_from_actual_predictions(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    rows = [s for s in samples if s.split == "test"]
    report = run_evaluation(oracle(rows, meta, root, flip=True), meta, samples, root,
                            split="test", evaluations_dir=tmp_path / "evals")
    assert report["metrics"]["accuracy"] == 0.0
    assert report["error_analysis"]["misclassified_count"] == len(rows)
    assert report["error_analysis"]["top_confusions"]


def test_confusion_matrix_totals_match_sample_count(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    rows = [s for s in samples if s.split == "val"]
    report = run_evaluation(oracle(rows, meta, root), meta, samples, root, split="val",
                            evaluations_dir=tmp_path / "evals")
    matrix = np.array(report["metrics"]["confusion_matrix"]["matrix"])
    assert matrix.sum() == report["sample_count"]
    assert any("NOT an independent" in w for w in report["warnings"])  # val is not a test estimate


def test_repeated_test_evaluation_is_flagged(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    rows = [s for s in samples if s.split == "test"]
    first = run_evaluation(oracle(rows, meta, root), meta, samples, root, split="test",
                           evaluations_dir=tmp_path / "evals")
    second = run_evaluation(oracle(rows, meta, root), meta, samples, root, split="test",
                            evaluations_dir=tmp_path / "evals")
    assert first["previous_evaluations_of_split"] == 0 and not first["warnings"]
    assert second["previous_evaluations_of_split"] == 1
    assert any("already been evaluated" in w for w in second["warnings"])


def test_unknown_labels_missing_images_and_empty_split_fail_loudly(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    with pytest.raises(EvaluationError, match="no samples"):
        run_evaluation(lambda b: b, meta, [s for s in samples if s.split != "test"], root,
                       split="test", evaluations_dir=tmp_path / "e1")
    other = metadata(classes=("x", "y"))
    with pytest.raises(EvaluationError, match="does not know"):
        run_evaluation(lambda b: b, other, samples, root, split="test", evaluations_dir=tmp_path / "e2")
    victim = next(s for s in samples if s.split == "test")
    (root / victim.relative_image_path).unlink()
    with pytest.raises(EvaluationError, match="could not be loaded"):
        run_evaluation(lambda b: np.zeros((b.shape[0], 4)), meta, samples, root, split="test",
                       evaluations_dir=tmp_path / "e3")


def test_bad_model_output_aborts_evaluation(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    with pytest.raises(EvaluationError, match="shape"):
        run_evaluation(lambda b: np.zeros((b.shape[0], 2)), meta, samples, root, split="test",
                       evaluations_dir=tmp_path / "e")
    with pytest.raises(EvaluationError, match="Invalid model output"):
        run_evaluation(lambda b: np.full((b.shape[0], 4), np.nan), meta, samples, root,
                       split="test", evaluations_dir=tmp_path / "e2")


def test_absent_class_in_split_is_reported_as_unavailable(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    meta = metadata()
    rows = [s for s in samples if s.split == "test" and s.class_label != "synthetic_line"]
    subset = [s for s in samples if s.split != "test"] + rows
    report = run_evaluation(oracle(rows, meta, root), meta, subset, root, split="test",
                            evaluations_dir=tmp_path / "evals")
    assert report["metrics"]["classes_without_support"] == ["synthetic_line"]
    assert any("no samples" in w for w in report["warnings"])
    assert "undefined" in (next((tmp_path / "evals").iterdir()) / "evaluation_report.md").read_text()
