import numpy as np

from helpers import image_bytes, make_app_config, make_engine, one_hot_predictor
from textile_defect_detection.errors import StorageError
from textile_defect_detection.inspection_store import InspectionStore
from textile_defect_detection.workflow import (load_model_state, open_store, run_analysis,
                                               validate_upload)


def test_successful_analysis_without_history():
    outcome = run_analysis(make_engine(), image_bytes(), "a.png")
    assert outcome.ok and outcome.saved is None and outcome.error_message is None


def test_no_engine_means_no_prediction():
    outcome = run_analysis(None, image_bytes(), "a.png")
    assert not outcome.ok and outcome.result is None and "unavailable" in outcome.error_message


def test_invalid_image_gives_error_outcome_never_a_prediction():
    outcome = run_analysis(make_engine(), b"junk", "a.png")
    assert not outcome.ok and outcome.result is None and "decoded" in outcome.error_message


def test_inference_failure_does_not_leave_a_result():
    def boom(batch):
        raise RuntimeError("x")

    outcome = run_analysis(make_engine(boom), image_bytes(), "a.png")
    assert outcome.result is None and outcome.error_message


def test_history_saving_matches_the_displayed_result(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    engine = make_engine(one_hot_predictor(2))
    outcome = run_analysis(engine, image_bytes(), "a.png", store=store, save_history=True)
    assert outcome.saved is True
    (record,) = store.list()
    assert record["inspection_id"] == outcome.result.inspection_id
    assert record["predicted_class"] == outcome.result.prediction.predicted_class == "gamma"
    assert abs(record["top_score"] - outcome.result.prediction.score) < 1e-12


def test_history_not_written_unless_requested(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    run_analysis(make_engine(), image_bytes(), "a.png", store=store, save_history=False)
    assert store.count() == 0


def test_history_failure_is_reported_honestly_but_result_survives(tmp_path):
    class BrokenStore:
        def add(self, result):
            raise StorageError("The inspection could not be saved to the history database.")

    outcome = run_analysis(make_engine(), image_bytes(), "a.png", store=BrokenStore(),
                           save_history=True)
    assert outcome.ok and outcome.saved is False and "could not be saved" in outcome.save_error


def test_failed_attempts_are_stored_as_failures_not_successes(tmp_path):
    store = InspectionStore(tmp_path / "h.db")
    run_analysis(make_engine(), b"junk", "bad.png", store=store, save_history=True)
    (record,) = store.list()
    assert record["status"] == "failed" and record["predicted_class"] is None


def test_validate_upload_uses_configured_limits(tmp_path):
    cfg = make_app_config(tmp_path)
    assert validate_upload(image_bytes(), "a.png", cfg.input).width == 64


def test_model_state_when_artifacts_missing_is_not_ready_and_explains(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "inference.yaml").write_text("model: {}\n")
    state = load_model_state(tmp_path / "configs" / "inference.yaml")
    assert state.ready is False and state.engine is None
    assert "models" in state.message and "Train" in state.message


def test_model_state_with_bad_config_is_not_ready(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "inference.yaml").write_text("inference:\n  confidence_threshold: 9\n")
    state = load_model_state(tmp_path / "configs" / "inference.yaml")
    assert not state.ready and "Configuration problem" in state.message


def test_open_store_reports_unusable_location(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = make_app_config(tmp_path)
    from dataclasses import replace
    bad = replace(cfg, storage=replace(cfg.storage, database_path=blocker / "sub" / "h.db"))
    store, problem = open_store(bad)
    assert store is None and "unavailable" in problem


def test_multiple_images_each_get_their_own_result():
    engine = make_engine(lambda b: np.array([[0.0, 5.0, 0.0]] * b.shape[0]))
    a = run_analysis(engine, image_bytes(colour=(1, 2, 3)), "a.png")
    b = run_analysis(engine, image_bytes(colour=(9, 9, 9), size=(30, 30)), "b.png")
    assert a.result.inspection_id != b.result.inspection_id
    assert (a.result.image_width, b.result.image_width) == (64, 30)
