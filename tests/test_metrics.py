import numpy as np
import pytest

from textile_defect_detection.errors import EvaluationError
from textile_defect_detection.metrics import compute_classification_metrics, confusion_matrix

NAMES = ["a", "b", "c"]


def test_hand_computed_example():
    # true = a a b b c ; pred = a b b b a   (METRIC FIXTURE, not textile results)
    m = compute_classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], NAMES)
    assert m.sample_count == 5 and abs(m.accuracy - 3 / 5) < 1e-12
    a, b, c = m.per_class
    assert (a.precision, a.recall, a.f1) == (0.5, 0.5, 0.5)
    assert abs(b.precision - 2 / 3) < 1e-12 and b.recall == 1.0 and abs(b.f1 - 0.8) < 1e-12
    assert c.precision is None and c.recall == 0.0 and c.f1 == 0.0
    assert abs(m.macro_f1 - (0.5 + 0.8 + 0.0) / 3) < 1e-12
    assert [x.support for x in m.per_class] == [2, 2, 1]
    assert m.classes_never_predicted == ["c"] and m.classes_without_support == []


def test_confusion_matrix_orientation_and_counts():
    cm = confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 0, 2], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]  # rows true, cols predicted
    assert cm.sum() == 6


def test_perfect_prediction():
    m = compute_classification_metrics([0, 1, 2, 2], [0, 1, 2, 2], NAMES)
    assert m.accuracy == 1.0 and m.macro_f1 == 1.0 and m.balanced_accuracy == 1.0


def test_absent_class_is_unavailable_not_zero():
    m = compute_classification_metrics([0, 0, 1], [0, 1, 1], NAMES)  # class c has no true samples
    c = m.per_class[2]
    assert c.support == 0 and c.recall is None and c.f1 is None
    assert m.classes_without_support == ["c"] and m.evaluated_class_count == 2
    # a: P=1 R=.5 F1=2/3 ; b: P=.5 R=1 F1=2/3 ; c has no support and is excluded from the mean
    assert abs(m.macro_f1 - 2 / 3) < 1e-12


def test_absent_class_that_is_predicted_still_has_no_f1():
    m = compute_classification_metrics([0, 0, 1], [0, 2, 1], NAMES)
    c = m.per_class[2]
    assert c.support == 0 and c.precision == 0.0 and c.f1 is None


def test_single_class_subset():
    m = compute_classification_metrics([1, 1, 1], [1, 1, 0], NAMES)
    assert m.evaluated_class_count == 1 and abs(m.accuracy - 2 / 3) < 1e-12
    assert m.per_class[1].recall == 2 / 3


@pytest.mark.parametrize("t,p", [([], []), ([0, 1], [0]), ([0, 3], [0, 1]), ([0, -1], [0, 1]),
                                 ([0.5, 1], [0, 1])])
def test_invalid_inputs_raise(t, p):
    with pytest.raises(EvaluationError):
        compute_classification_metrics(t, p, NAMES)


def test_needs_two_classes():
    with pytest.raises(EvaluationError):
        compute_classification_metrics([0], [0], ["only"])


def test_matches_scikit_learn_where_conventions_agree():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    y_pred = np.where(rng.random(200) < 0.6, y_true, rng.integers(0, 4, 200))
    names = list("abcd")
    m = compute_classification_metrics(y_true, y_pred, names)
    assert abs(m.accuracy - sk.accuracy_score(y_true, y_pred)) < 1e-12
    assert abs(m.macro_f1 - sk.f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3])) < 1e-12
    prec, rec, f1, sup = sk.precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3],
                                                            zero_division=0)
    for i, c in enumerate(m.per_class):
        assert abs(c.precision - prec[i]) < 1e-12 and abs(c.recall - rec[i]) < 1e-12
        assert abs(c.f1 - f1[i]) < 1e-12 and c.support == sup[i]
    assert np.array_equal(np.array(m.confusion_matrix),
                          sk.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))


def test_to_dict_keeps_undefined_as_null():
    d = compute_classification_metrics([0, 0, 1], [0, 1, 1], NAMES).to_dict()
    assert d["per_class"][2]["f1"] is None
    assert d["confusion_matrix"]["orientation"].startswith("rows=true")
