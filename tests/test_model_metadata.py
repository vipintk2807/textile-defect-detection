import pytest

from helpers import make_metadata
from textile_defect_detection.errors import ModelArtifactError
from textile_defect_detection.model_metadata import (ModelMetadata, load_metadata, save_metadata,
                                                     verify_weights_file)
from textile_defect_detection.provenance import sha256_file


def test_round_trip_preserves_class_order_and_preprocessing(tmp_path):
    meta = make_metadata(("zeta", "alpha", "mid"))
    path = save_metadata(meta, tmp_path / "m.json")
    loaded = load_metadata(path)
    assert loaded.class_names == ("zeta", "alpha", "mid")  # order is the output-index mapping
    assert loaded.preprocessing == meta.preprocessing and loaded.trained_on_synthetic_data


def test_save_refuses_to_overwrite(tmp_path):
    save_metadata(make_metadata(), tmp_path / "m.json")
    with pytest.raises(ModelArtifactError, match="overwrite"):
        save_metadata(make_metadata(), tmp_path / "m.json")


@pytest.mark.parametrize("change", [
    {"class_names": ["only"]},
    {"class_names": ["a", "a"]},
    {"class_names": ["a", " "]},
    {"weights_sha256": "xyz"},
    {"num_classes": 5},
    {"architecture": "vgg16"},
    {"task": "detection"},
    {"schema_version": 99},
])
def test_invalid_metadata_is_rejected(change):
    data = make_metadata().to_dict()
    data.update(change)
    with pytest.raises(ModelArtifactError):
        ModelMetadata.from_dict(data)


def test_missing_required_fields_and_bad_json(tmp_path):
    data = make_metadata().to_dict()
    del data["class_names"]
    with pytest.raises(ModelArtifactError, match="incomplete"):
        ModelMetadata.from_dict(data)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ModelArtifactError, match="not valid JSON"):
        load_metadata(bad)
    with pytest.raises(ModelArtifactError, match="not found"):
        load_metadata(tmp_path / "missing.json")


def test_weights_checksum_verification(tmp_path):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"pretend weights")
    meta = ModelMetadata.from_dict({**make_metadata().to_dict(),
                                    "weights_sha256": sha256_file(weights)})
    verify_weights_file(weights, meta)  # matches -> no error
    weights.write_bytes(b"tampered")
    with pytest.raises(ModelArtifactError, match="checksum"):
        verify_weights_file(weights, meta)
    with pytest.raises(ModelArtifactError, match="not found"):
        verify_weights_file(tmp_path / "gone.pth", meta)
