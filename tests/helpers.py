"""Shared test helpers. Everything generated here is SYNTHETIC test data, never real textile data."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from textile_defect_detection.config import (AppConfig, InferenceSettings, InputConfig,
                                             ModelConfig, PreprocessingConfig, StorageConfig)
from textile_defect_detection.dataset import (build_class_mapping, read_samples_csv,
                                              scan_class_folders, write_samples_csv,
                                              save_class_mapping)
from textile_defect_detection.inference import InferenceEngine
from textile_defect_detection.model_metadata import ModelMetadata
from textile_defect_detection.splits import assign_splits
from textile_defect_detection.synthetic import generate_synthetic_dataset

CLASS_NAMES = ("alpha", "beta", "gamma")
SMALL_PREPROCESSING = PreprocessingConfig(resize_size=40, crop_size=32)


def image_bytes(fmt: str = "PNG", size=(64, 48), colour=(120, 80, 40), mode: str = "RGB") -> bytes:
    img = Image.new("RGB", size, colour)
    if mode != "RGB":
        img = img.convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, fmt)
    return buffer.getvalue()


def noisy_image_bytes(seed: int = 0, size=(64, 64)) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, "PNG")
    return buffer.getvalue()


def make_metadata(class_names=CLASS_NAMES, synthetic: bool = True) -> ModelMetadata:
    return ModelMetadata(
        model_id="test-model", class_names=tuple(class_names), preprocessing=SMALL_PREPROCESSING,
        weights_sha256="0" * 64, created_at="2026-01-01T00:00:00Z",
        dataset={"dataset_name": "SYNTHETIC test fixture", "synthetic": synthetic})


def one_hot_predictor(index: int, num_classes: int = len(CLASS_NAMES), strength: float = 8.0):
    """Stand-in for a model that always favours ``index`` (tests software, not accuracy)."""
    def predict(batch: np.ndarray) -> np.ndarray:
        logits = np.zeros((batch.shape[0], num_classes), dtype=np.float32)
        logits[:, index] = strength
        return logits
    return predict


def make_engine(predictor=None, *, threshold: float = 0.5, class_names=CLASS_NAMES) -> InferenceEngine:
    return InferenceEngine(make_metadata(class_names), predictor or one_hot_predictor(1),
                           input_config=InputConfig(max_file_size_mb=1), threshold=threshold)


def make_app_config(tmp: Path) -> AppConfig:
    return AppConfig(
        model=ModelConfig("classification", "resnet18", tmp / "w.pth", tmp / "m.json", "cpu"),
        input=InputConfig(), inference=InferenceSettings(),
        storage=StorageConfig(tmp / "h.db", tmp / "exports"), project_root=tmp)


def prepared_synthetic_dataset(tmp: Path, images_per_class: int = 12, seed: int = 0):
    """Generate a synthetic dataset and a split manifest. Returns (images_root, samples, mapping)."""
    root = generate_synthetic_dataset(tmp / "syn", images_per_class=images_per_class, seed=seed)
    samples = scan_class_folders(root, source_dataset="synthetic-demo", group_regex=r"_(g\d+)_")
    mapping = build_class_mapping([s.class_label for s in samples])
    samples = assign_splits(samples, seed=1).samples
    manifest_dir = tmp / "proc"
    write_samples_csv(samples, manifest_dir / "samples.csv")
    save_class_mapping(mapping, manifest_dir / "class_mapping.json")
    shutil.copyfile(root / "dataset_info.json", manifest_dir / "dataset_manifest.json")
    return root, read_samples_csv(manifest_dir / "samples.csv"), mapping
