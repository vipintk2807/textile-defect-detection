import numpy as np
import pytest

from textile_defect_detection.errors import InferenceError
from textile_defect_detection.predictions import (STATUS_COMPLETED, STATUS_LOW_CONFIDENCE,
                                                  build_result, decide_status,
                                                  logits_to_prediction, softmax)

NAMES = ["normal", "hole", "stain"]


def test_softmax_sums_to_one_and_is_stable_for_large_logits():
    probs = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert abs(probs.sum() - 1.0) < 1e-12 and np.all(np.isfinite(probs))
    assert int(np.argmax(probs)) == 1


def test_softmax_known_value():
    probs = softmax(np.array([0.0, 0.0]))
    assert abs(probs[0] - 0.5) < 1e-12


@pytest.mark.parametrize("bad", [np.array([np.nan, 1.0]), np.array([np.inf, 0.0]),
                                 np.array([]), np.zeros((2, 2))])
def test_invalid_logits_are_rejected(bad):
    with pytest.raises(InferenceError):
        softmax(bad)


def test_index_maps_to_correct_class_and_ranking_is_consistent():
    pred = logits_to_prediction(np.array([[0.1, 3.0, 1.0]]), NAMES)
    assert (pred.predicted_index, pred.predicted_class) == (1, "hole")
    assert [c.class_name for c in pred.ranked] == ["hole", "stain", "normal"]
    assert abs(sum(c.score for c in pred.ranked) - 1.0) < 1e-9
    assert pred.score == pred.ranked[0].score


def test_ties_resolve_to_lowest_index():
    pred = logits_to_prediction(np.zeros(3), NAMES)
    assert pred.predicted_index == 0


def test_top_k_limits_ranked_list_but_not_the_prediction():
    pred = logits_to_prediction(np.array([0.0, 5.0, 1.0]), NAMES, top_k=2)
    assert len(pred.ranked) == 2 and pred.predicted_class == "hole"


def test_class_count_mismatch_is_an_error_not_a_guess():
    with pytest.raises(InferenceError, match="class mapping"):
        logits_to_prediction(np.zeros(4), NAMES)
    with pytest.raises(InferenceError):
        logits_to_prediction(np.zeros((2, 3)), NAMES)


def test_threshold_decides_status_boundary_inclusive():
    assert decide_status(0.5, 0.5) == STATUS_COMPLETED
    assert decide_status(0.4999, 0.5) == STATUS_LOW_CONFIDENCE


def test_result_contract_fields():
    pred = logits_to_prediction(np.array([0.0, 4.0, 0.0]), NAMES)
    result = build_result(pred, image_name="a.png", model_id="m", threshold=0.9,
                          processing_time_ms=1.5, image_width=10, image_height=20)
    d = result.to_dict()
    for key in ("inspection_id", "model_id", "task_type", "status", "predictions", "threshold",
                "processing_time_ms", "image_width", "image_height", "created_at"):
        assert key in d
    assert d["task_type"] == "classification" and d["predictions"]["scoring"] == "softmax"
    assert result.status == STATUS_COMPLETED and not result.needs_review
