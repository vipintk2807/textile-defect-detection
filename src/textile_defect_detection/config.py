"""Typed, validated configuration for inference and training.

Configuration is loaded from YAML with ``yaml.safe_load`` (never evaluated as code). Unknown
keys are rejected so that typos are caught instead of silently ignored. Relative paths are
resolved against the project root, i.e. the parent of the directory holding the config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
APP_IMAGE_FORMATS = ("jpg", "jpeg", "png", "webp")
SUPPORTED_ARCHITECTURES = ("resnet18",)
RESIZE_MODES = ("center_crop", "squash")


# --------------------------------------------------------------------------- helpers
def _check_keys(data: dict, allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown setting(s) in '{where}': {', '.join(sorted(unknown))}.")


def _as_dict(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{where}' must be a mapping of settings.")
    return value


def _number(value: Any, name: str, *, low: float | None = None, high: float | None = None,
            integer: bool = False, low_inclusive: bool = True) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number.")
    if integer and (isinstance(value, float) and not value.is_integer()):
        raise ConfigError(f"'{name}' must be a whole number.")
    value = int(value) if integer else float(value)
    if low is not None and (value < low or (not low_inclusive and value == low)):
        raise ConfigError(f"'{name}' must be {'>=' if low_inclusive else '>'} {low}.")
    if high is not None and value > high:
        raise ConfigError(f"'{name}' must be <= {high}.")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false.")
    return value


def _choice(value: Any, name: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ConfigError(f"'{name}' must be one of: {', '.join(options)}.")
    return value


def _resolve(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def load_yaml_mapping(path: str | Path) -> dict:
    """Read a YAML file that must contain a mapping."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file '{path.name}' was not found.", detail=str(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Configuration file '{path.name}' could not be read as YAML.",
                          detail=repr(exc)) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file '{path.name}' must contain a mapping.")
    return data


def project_root_for(config_path: str | Path) -> Path:
    """Project root = parent of the folder containing the config file."""
    return Path(config_path).resolve().parent.parent


# --------------------------------------------------------------------------- preprocessing
@dataclass(frozen=True)
class PreprocessingConfig:
    """Image preprocessing shared by training, evaluation and inference.

    Defaults follow the documented torchvision ResNet-18 ImageNet recipe: resize shorter side
    to 256 (bilinear), centre-crop 224, scale to [0, 1], normalise with ImageNet mean/std.
    ``resize_mode="squash"`` instead resizes the whole image to ``crop_size`` x ``crop_size``
    (keeps all content, distorts aspect ratio); it must be evaluated against the baseline.
    """

    resize_size: int = 256
    crop_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    resize_mode: str = "center_crop"

    def __post_init__(self) -> None:
        if self.crop_size < 8 or self.resize_size < 8:
            raise ConfigError("Preprocessing sizes must be at least 8 pixels.")
        if self.resize_mode not in RESIZE_MODES:
            raise ConfigError(f"'resize_mode' must be one of: {', '.join(RESIZE_MODES)}.")
        if self.resize_mode == "center_crop" and self.resize_size < self.crop_size:
            raise ConfigError("'resize_size' must be >= 'crop_size' for center_crop mode.")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigError("'mean' and 'std' must each have 3 values (RGB).")
        if any(s <= 0 for s in self.std):
            raise ConfigError("'std' values must be positive.")

    def to_dict(self) -> dict:
        return {
            "resize_size": self.resize_size,
            "crop_size": self.crop_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "resize_mode": self.resize_mode,
        }

    @classmethod
    def from_dict(cls, data: dict | None, where: str = "preprocessing") -> "PreprocessingConfig":
        data = _as_dict(data, where)
        _check_keys(data, {"resize_size", "crop_size", "mean", "std", "resize_mode"}, where)
        defaults = cls()
        mean = data.get("mean", defaults.mean)
        std = data.get("std", defaults.std)
        for label, seq in (("mean", mean), ("std", std)):
            if not isinstance(seq, (list, tuple)) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq
            ):
                raise ConfigError(f"'{where}.{label}' must be a list of numbers.")
        return cls(
            resize_size=int(_number(data.get("resize_size", defaults.resize_size),
                                    f"{where}.resize_size", low=8, integer=True)),
            crop_size=int(_number(data.get("crop_size", defaults.crop_size),
                                  f"{where}.crop_size", low=8, integer=True)),
            mean=tuple(float(v) for v in mean),  # type: ignore[arg-type]
            std=tuple(float(v) for v in std),  # type: ignore[arg-type]
            resize_mode=_choice(data.get("resize_mode", defaults.resize_mode),
                                f"{where}.resize_mode", RESIZE_MODES),
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-only augmentation. Never applied to validation, test or inference."""

    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: float = 0.0
    brightness: float = 0.1
    contrast: float = 0.1

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "rotation_degrees": self.rotation_degrees,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }

    @classmethod
    def from_dict(cls, data: dict | None, where: str = "augmentation") -> "AugmentationConfig":
        data = _as_dict(data, where)
        allowed = {"enabled", "horizontal_flip", "vertical_flip", "rotation_degrees",
                   "brightness", "contrast"}
        _check_keys(data, allowed, where)
        d = cls()
        return cls(
            enabled=_boolean(data.get("enabled", d.enabled), f"{where}.enabled"),
            horizontal_flip=_boolean(data.get("horizontal_flip", d.horizontal_flip),
                                     f"{where}.horizontal_flip"),
            vertical_flip=_boolean(data.get("vertical_flip", d.vertical_flip),
                                   f"{where}.vertical_flip"),
            rotation_degrees=float(_number(data.get("rotation_degrees", d.rotation_degrees),
                                           f"{where}.rotation_degrees", low=0, high=45)),
            brightness=float(_number(data.get("brightness", d.brightness),
                                     f"{where}.brightness", low=0, high=0.9)),
            contrast=float(_number(data.get("contrast", d.contrast),
                                   f"{where}.contrast", low=0, high=0.9)),
        )


# --------------------------------------------------------------------------- app / inference
@dataclass(frozen=True)
class ModelConfig:
    task: str
    architecture: str
    weights_path: Path
    metadata_path: Path
    device: str


@dataclass(frozen=True)
class InputConfig:
    max_file_size_mb: float = 10.0
    supported_formats: tuple[str, ...] = APP_IMAGE_FORMATS
    max_image_pixels: int = 40_000_000
    min_dimension: int = 16

    @property
    def max_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)


@dataclass(frozen=True)
class InferenceSettings:
    confidence_threshold: float = 0.5
    top_k: int | None = None


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path
    export_directory: Path


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    input: InputConfig
    inference: InferenceSettings
    storage: StorageConfig
    project_root: Path = field(default_factory=Path.cwd)


def load_app_config(path: str | Path) -> AppConfig:
    """Load and validate ``configs/inference.yaml``."""
    path = Path(path)
    data = load_yaml_mapping(path)
    base = project_root_for(path)
    _check_keys(data, {"model", "input", "inference", "storage"}, "config")

    m = _as_dict(data.get("model"), "model")
    _check_keys(m, {"task", "architecture", "weights_path", "metadata_path", "device"}, "model")
    model = ModelConfig(
        task=_choice(m.get("task", "classification"), "model.task", ("classification",)),
        architecture=_choice(m.get("architecture", "resnet18"), "model.architecture",
                             SUPPORTED_ARCHITECTURES),
        weights_path=_resolve(m.get("weights_path", "models/model_weights.pth"), base,
                              "model.weights_path"),
        metadata_path=_resolve(m.get("metadata_path", "models/model_metadata.json"), base,
                               "model.metadata_path"),
        device=_choice(m.get("device", "cpu"), "model.device", ("cpu", "cuda")),
    )

    i = _as_dict(data.get("input"), "input")
    _check_keys(i, {"max_file_size_mb", "supported_formats", "max_image_pixels",
                    "min_dimension"}, "input")
    formats = i.get("supported_formats", list(APP_IMAGE_FORMATS))
    if not isinstance(formats, list) or not formats:
        raise ConfigError("'input.supported_formats' must be a non-empty list.")
    normalised = tuple(str(f).lower().lstrip(".") for f in formats)
    bad = [f for f in normalised if f not in APP_IMAGE_FORMATS]
    if bad:
        raise ConfigError(
            f"Unsupported format(s) in 'input.supported_formats': {', '.join(bad)}. "
            f"Allowed: {', '.join(APP_IMAGE_FORMATS)}."
        )
    input_cfg = InputConfig(
        max_file_size_mb=float(_number(i.get("max_file_size_mb", 10.0), "input.max_file_size_mb",
                                       low=0, low_inclusive=False, high=500)),
        supported_formats=normalised,
        max_image_pixels=int(_number(i.get("max_image_pixels", 40_000_000),
                                     "input.max_image_pixels", low=1, integer=True)),
        min_dimension=int(_number(i.get("min_dimension", 16), "input.min_dimension",
                                  low=1, integer=True)),
    )

    inf = _as_dict(data.get("inference"), "inference")
    _check_keys(inf, {"confidence_threshold", "top_k"}, "inference")
    top_k = inf.get("top_k")
    inference = InferenceSettings(
        confidence_threshold=float(_number(inf.get("confidence_threshold", 0.5),
                                           "inference.confidence_threshold", low=0, high=1)),
        top_k=None if top_k is None else int(_number(top_k, "inference.top_k", low=1,
                                                     integer=True)),
    )

    s = _as_dict(data.get("storage"), "storage")
    _check_keys(s, {"database_path", "export_directory"}, "storage")
    storage = StorageConfig(
        database_path=_resolve(s.get("database_path", "outputs/inspections.db"), base,
                               "storage.database_path"),
        export_directory=_resolve(s.get("export_directory", "outputs/exports"), base,
                                  "storage.export_directory"),
    )
    return AppConfig(model=model, input=input_cfg, inference=inference, storage=storage,
                     project_root=base)


# --------------------------------------------------------------------------- training
@dataclass(frozen=True)
class DatasetSection:
    manifest_path: Path
    images_root: Path
    class_mapping_path: Path | None = None
    dataset_info_path: Path | None = None


@dataclass(frozen=True)
class ModelSection:
    architecture: str = "resnet18"
    pretrained: bool = True
    pretrained_weights: str = "IMAGENET1K_V1"
    freeze_backbone: bool = True


@dataclass(frozen=True)
class TrainSection:
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    selection_metric: str = "val_macro_f1"
    class_weighting: str = "none"
    num_workers: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class OutputSection:
    directory: Path
    experiment_name: str = "resnet18-baseline"


@dataclass(frozen=True)
class TrainingConfig:
    dataset: DatasetSection
    model: ModelSection
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    training: TrainSection
    output: OutputSection

    def to_dict(self) -> dict:
        """Plain-data form used for snapshots (``training_config.yaml``) and hashing."""
        t = self.training
        return {
            "dataset": {
                "manifest_path": str(self.dataset.manifest_path),
                "images_root": str(self.dataset.images_root),
                "class_mapping_path": _opt_str(self.dataset.class_mapping_path),
                "dataset_info_path": _opt_str(self.dataset.dataset_info_path),
            },
            "model": {
                "architecture": self.model.architecture,
                "pretrained": self.model.pretrained,
                "pretrained_weights": self.model.pretrained_weights,
                "freeze_backbone": self.model.freeze_backbone,
            },
            "preprocessing": self.preprocessing.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "training": {
                "seed": t.seed,
                "batch_size": t.batch_size,
                "max_epochs": t.max_epochs,
                "learning_rate": t.learning_rate,
                "backbone_learning_rate": t.backbone_learning_rate,
                "weight_decay": t.weight_decay,
                "early_stopping_patience": t.early_stopping_patience,
                "selection_metric": t.selection_metric,
                "class_weighting": t.class_weighting,
                "num_workers": t.num_workers,
                "device": t.device,
            },
            "output": {
                "directory": str(self.output.directory),
                "experiment_name": self.output.experiment_name,
            },
        }


def _opt_str(value: Path | None) -> str | None:
    return None if value is None else str(value)


def training_config_from_dict(data: dict, base: Path) -> TrainingConfig:
    """Build a validated :class:`TrainingConfig` from a mapping (paths relative to ``base``)."""
    _check_keys(data, {"dataset", "model", "preprocessing", "augmentation", "training",
                       "output"}, "config")

    d = _as_dict(data.get("dataset"), "dataset")
    _check_keys(d, {"manifest_path", "images_root", "class_mapping_path",
                    "dataset_info_path"}, "dataset")
    if "manifest_path" not in d or "images_root" not in d:
        raise ConfigError("'dataset.manifest_path' and 'dataset.images_root' are required.")
    manifest = _resolve(d["manifest_path"], base, "dataset.manifest_path")
    mapping = d.get("class_mapping_path")
    info = d.get("dataset_info_path")
    dataset = DatasetSection(
        manifest_path=manifest,
        images_root=_resolve(d["images_root"], base, "dataset.images_root"),
        class_mapping_path=(_resolve(mapping, base, "dataset.class_mapping_path")
                            if mapping else manifest.parent / "class_mapping.json"),
        dataset_info_path=(_resolve(info, base, "dataset.dataset_info_path")
                           if info else manifest.parent / "dataset_manifest.json"),
    )

    mo = _as_dict(data.get("model"), "model")
    _check_keys(mo, {"architecture", "pretrained", "pretrained_weights", "freeze_backbone"},
                "model")
    md = ModelSection()
    model = ModelSection(
        architecture=_choice(mo.get("architecture", md.architecture), "model.architecture",
                             SUPPORTED_ARCHITECTURES),
        pretrained=_boolean(mo.get("pretrained", md.pretrained), "model.pretrained"),
        pretrained_weights=_choice(mo.get("pretrained_weights", md.pretrained_weights),
                                   "model.pretrained_weights", ("IMAGENET1K_V1",)),
        freeze_backbone=_boolean(mo.get("freeze_backbone", md.freeze_backbone),
                                 "model.freeze_backbone"),
    )

    t = _as_dict(data.get("training"), "training")
    _check_keys(t, {"seed", "batch_size", "max_epochs", "learning_rate",
                    "backbone_learning_rate", "weight_decay", "early_stopping_patience",
                    "selection_metric", "class_weighting", "num_workers", "device"}, "training")
    td = TrainSection()
    training = TrainSection(
        seed=int(_number(t.get("seed", td.seed), "training.seed", low=0, integer=True)),
        batch_size=int(_number(t.get("batch_size", td.batch_size), "training.batch_size",
                               low=1, integer=True)),
        max_epochs=int(_number(t.get("max_epochs", td.max_epochs), "training.max_epochs",
                               low=1, integer=True)),
        learning_rate=float(_number(t.get("learning_rate", td.learning_rate),
                                    "training.learning_rate", low=0, low_inclusive=False)),
        backbone_learning_rate=float(_number(
            t.get("backbone_learning_rate", td.backbone_learning_rate),
            "training.backbone_learning_rate", low=0, low_inclusive=False)),
        weight_decay=float(_number(t.get("weight_decay", td.weight_decay),
                                   "training.weight_decay", low=0)),
        early_stopping_patience=int(_number(
            t.get("early_stopping_patience", td.early_stopping_patience),
            "training.early_stopping_patience", low=1, integer=True)),
        selection_metric=_choice(t.get("selection_metric", td.selection_metric),
                                 "training.selection_metric", ("val_macro_f1", "val_loss")),
        class_weighting=_choice(t.get("class_weighting", td.class_weighting),
                                "training.class_weighting", ("none", "balanced")),
        num_workers=int(_number(t.get("num_workers", td.num_workers), "training.num_workers",
                                low=0, integer=True)),
        device=_choice(t.get("device", td.device), "training.device", ("auto", "cpu", "cuda")),
    )

    o = _as_dict(data.get("output"), "output")
    _check_keys(o, {"directory", "experiment_name"}, "output")
    name = o.get("experiment_name", "resnet18-baseline")
    if not isinstance(name, str) or not name or not all(
        c.isalnum() or c in "-_" for c in name
    ):
        raise ConfigError("'output.experiment_name' may only contain letters, digits, '-' and '_'.")
    output = OutputSection(
        directory=_resolve(o.get("directory", "outputs/experiments"), base, "output.directory"),
        experiment_name=name,
    )

    return TrainingConfig(
        dataset=dataset,
        model=model,
        preprocessing=PreprocessingConfig.from_dict(data.get("preprocessing")),
        augmentation=AugmentationConfig.from_dict(data.get("augmentation")),
        training=training,
        output=output,
    )


def load_training_config(path: str | Path, base_dir: str | Path | None = None) -> TrainingConfig:
    """Load and validate ``configs/training.yaml`` (or a ``training_config.yaml`` snapshot)."""
    path = Path(path)
    data = load_yaml_mapping(path)
    base = Path(base_dir) if base_dir else project_root_for(path)
    return training_config_from_dict(data, base)
