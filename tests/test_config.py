import pytest

from textile_defect_detection.config import (PreprocessingConfig, load_app_config,
                                             load_training_config)
from textile_defect_detection.errors import ConfigError


def write(tmp_path, text, name="inference.yaml"):
    folder = tmp_path / "configs"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_default_inference_config_loads_and_resolves_paths(tmp_path):
    cfg = load_app_config(write(tmp_path, "model: {}\n"))
    assert cfg.model.weights_path == tmp_path.resolve() / "models" / "model_weights.pth"
    assert cfg.inference.confidence_threshold == 0.5
    assert cfg.input.max_bytes == 10 * 1024 * 1024
    assert cfg.model.device == "cpu"


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown setting"):
        load_app_config(write(tmp_path, "inference:\n  confidence_treshold: 0.5\n"))


@pytest.mark.parametrize("body", [
    "inference:\n  confidence_threshold: 1.5\n",
    "inference:\n  confidence_threshold: high\n",
    "input:\n  max_file_size_mb: 0\n",
    "input:\n  supported_formats: [svg]\n",
    "model:\n  device: tpu\n",
    "model:\n  architecture: vgg\n",
])
def test_invalid_values_are_rejected(tmp_path, body):
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, body))


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_app_config(tmp_path / "configs" / "nope.yaml")
    with pytest.raises(ConfigError):
        load_app_config(write(tmp_path, "model: [unclosed\n"))
    with pytest.raises(ConfigError, match="mapping"):
        load_app_config(write(tmp_path, "- a\n- b\n"))


def test_preprocessing_validation():
    with pytest.raises(ConfigError):
        PreprocessingConfig(resize_size=100, crop_size=224)
    with pytest.raises(ConfigError):
        PreprocessingConfig(std=(0.0, 1.0, 1.0))
    assert PreprocessingConfig(resize_size=64, crop_size=64, resize_mode="squash").crop_size == 64
    round_trip = PreprocessingConfig.from_dict(PreprocessingConfig().to_dict())
    assert round_trip == PreprocessingConfig()


def test_training_config_defaults_and_snapshot_round_trip(tmp_path):
    text = ("dataset:\n  manifest_path: data/processed/x/samples.csv\n  images_root: data/raw/x\n"
            "training:\n  batch_size: 4\n")
    path = write(tmp_path, text, "training.yaml")
    cfg = load_training_config(path)
    assert cfg.training.batch_size == 4 and cfg.training.max_epochs == 20
    assert cfg.dataset.class_mapping_path.name == "class_mapping.json"
    assert cfg.preprocessing.crop_size == 224
    # snapshot -> reload gives identical settings (resume relies on this)
    import yaml
    snap = write(tmp_path, yaml.safe_dump(cfg.to_dict()), "snapshot.yaml")
    assert load_training_config(snap).to_dict() == cfg.to_dict()


def test_training_config_requires_dataset_and_valid_names(tmp_path):
    with pytest.raises(ConfigError, match="required"):
        load_training_config(write(tmp_path, "model: {}\n", "training.yaml"))
    bad = ("dataset:\n  manifest_path: a.csv\n  images_root: b\n"
           "output:\n  experiment_name: '../evil'\n")
    with pytest.raises(ConfigError, match="experiment_name"):
        load_training_config(write(tmp_path, bad, "training.yaml"))


def test_shipped_configs_are_valid():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    load_app_config(root / "configs" / "inference.yaml")
    load_training_config(root / "configs" / "training.yaml")
    load_training_config(root / "configs" / "training_synthetic_smoke.yaml")
