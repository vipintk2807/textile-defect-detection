import numpy as np
import pytest

from helpers import CLASS_NAMES, image_bytes, make_app_config, make_engine, one_hot_predictor
from textile_defect_detection.errors import (ImageValidationError, InferenceError,
                                             ModelArtifactError)
from textile_defect_detection.inference import InferenceEngine
from textile_defect_detection.predictions import STATUS_COMPLETED, STATUS_LOW_CONFIDENCE


def test_end_to_end_with_stand_in_predictor_maps_index_to_class():
    # NOTE: the predictor is a test double; this verifies software wiring, not model accuracy.
    for index, name in enumerate(CLASS_NAMES):
        result = make_engine(one_hot_predictor(index)).inspect_bytes(image_bytes(), "s.png")
        assert result.prediction.predicted_class == name
        assert result.status == STATUS_COMPLETED
        assert result.image_width == 64 and result.image_height == 48
        assert result.processing_time_ms >= 0 and result.model_id == "test-model"


def test_predictor_receives_correctly_shaped_float32_batch():
    seen = {}

    def spy(batch):
        seen["shape"], seen["dtype"] = batch.shape, batch.dtype
        return np.zeros((batch.shape[0], 3), dtype=np.float32)

    make_engine(spy).inspect_bytes(image_bytes(), "s.png")
    assert seen["shape"] == (1, 3, 32, 32) and seen["dtype"] == np.float32


def test_low_confidence_is_reported_as_uncertain_not_as_defect_free():
    flat = lambda batch: np.zeros((batch.shape[0], 3), dtype=np.float32)  # noqa: E731
    result = make_engine(flat, threshold=0.9).inspect_bytes(image_bytes(), "s.png")
    assert result.status == STATUS_LOW_CONFIDENCE and result.needs_review


@pytest.mark.parametrize("data,name", [(b"", "a.png"), (b"junk", "a.png"),
                                       (image_bytes(), "a.gif")])
def test_invalid_input_never_reaches_the_model(data, name):
    calls = []

    def predictor(batch):
        calls.append(1)
        return np.zeros((1, 3))

    with pytest.raises(ImageValidationError):
        make_engine(predictor).inspect_bytes(data, name)
    assert calls == []


def test_oversize_upload_rejected_before_processing():
    from helpers import noisy_image_bytes  # engine limit is 1 MB in make_engine
    with pytest.raises(ImageValidationError, match="larger than"):
        make_engine().inspect_bytes(noisy_image_bytes(0, (700, 700)), "big.png")


def test_predictor_failures_become_safe_inference_errors():
    def boom(batch):
        raise RuntimeError("secret internal path C:\\Users\\me\\x")

    with pytest.raises(InferenceError) as info:
        make_engine(boom).inspect_bytes(image_bytes(), "s.png")
    assert "secret" not in info.value.user_message and "Users" not in info.value.user_message

    def oom(batch):
        raise MemoryError()

    with pytest.raises(InferenceError, match="memory"):
        make_engine(oom).inspect_bytes(image_bytes(), "s.png")


@pytest.mark.parametrize("output", [np.zeros((1, 4)), np.full((1, 3), np.nan),
                                    np.full((1, 3), np.inf), np.zeros((2, 3))])
def test_invalid_model_output_is_rejected(output):
    with pytest.raises(InferenceError):
        make_engine(lambda batch: output).inspect_bytes(image_bytes(), "s.png")


def test_filename_is_sanitised_in_the_result():
    result = make_engine().inspect_bytes(image_bytes(), "x/../evil\n.png")
    assert result.image_name == "evil.png"


def test_inspect_path_and_missing_file(tmp_path):
    path = tmp_path / "ok.png"
    path.write_bytes(image_bytes())
    assert make_engine().inspect_path(path).image_name == "ok.png"
    with pytest.raises(ImageValidationError):
        make_engine().inspect_path(tmp_path / "missing.png")


def test_from_config_with_missing_artifacts_fails_clearly_without_fallback(tmp_path):
    with pytest.raises(ModelArtifactError, match="metadata file"):
        InferenceEngine.from_config(make_app_config(tmp_path))


def test_from_config_with_metadata_but_no_weights(tmp_path):
    from helpers import make_metadata
    from textile_defect_detection.model_metadata import save_metadata
    cfg = make_app_config(tmp_path)
    save_metadata(make_metadata(), cfg.model.metadata_path)
    with pytest.raises(ModelArtifactError, match="weights file"):
        InferenceEngine.from_config(cfg)


def test_model_info_flags_synthetic_training_data():
    assert make_engine().info.trained_on_synthetic_data is True
