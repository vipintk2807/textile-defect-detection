"""Model metadata: class mapping, preprocessing and provenance that travel with the weights.

The weights file only contains tensors. Everything needed to use them correctly (class order,
preprocessing, architecture) lives in ``model_metadata.json`` and is validated at load time.
The metadata stores the SHA-256 of the weights file, so a mismatching or swapped weights file
is rejected instead of silently producing wrong labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import SUPPORTED_ARCHITECTURES, PreprocessingConfig
from .errors import ConfigError, ModelArtifactError
from .provenance import sha256_file

SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    class_names: tuple[str, ...]
    preprocessing: PreprocessingConfig
    weights_sha256: str
    architecture: str = "resnet18"
    task: str = "classification"
    scoring: str = "softmax"
    weights_file: str = "model_weights.pth"
    created_at: str = ""
    framework: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    evaluation_report: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_id or not isinstance(self.model_id, str):
            raise ModelArtifactError("Model metadata is invalid: 'model_id' is missing.")
        if self.task != "classification":
            raise ModelArtifactError(
                f"Model metadata task '{self.task}' is not supported (classification only).")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ModelArtifactError(
                f"Model architecture '{self.architecture}' is not supported.")
        if self.scoring != "softmax":
            raise ModelArtifactError(f"Scoring method '{self.scoring}' is not supported.")
        names = tuple(self.class_names)
        if len(names) < 2:
            raise ModelArtifactError("Model metadata must list at least two classes.")
        if any((not isinstance(n, str)) or not n.strip() for n in names):
            raise ModelArtifactError("Model metadata contains an empty class name.")
        if len(set(names)) != len(names):
            raise ModelArtifactError("Model metadata contains duplicate class names.")
        if not isinstance(self.weights_sha256, str) or not _SHA256_RE.match(self.weights_sha256):
            raise ModelArtifactError("Model metadata has an invalid 'weights_sha256'.")
        if self.schema_version != SCHEMA_VERSION:
            raise ModelArtifactError(
                f"Model metadata schema version {self.schema_version} is not supported "
                f"(expected {SCHEMA_VERSION}).")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def trained_on_synthetic_data(self) -> bool:
        return bool(self.dataset.get("synthetic", False))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "task": self.task,
            "architecture": self.architecture,
            "scoring": self.scoring,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "preprocessing": self.preprocessing.to_dict(),
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "created_at": self.created_at,
            "framework": self.framework,
            "training": self.training,
            "dataset": self.dataset,
            "evaluation_report": self.evaluation_report,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        if not isinstance(data, dict):
            raise ModelArtifactError("Model metadata must be a JSON object.")
        required = ("model_id", "class_names", "preprocessing", "weights_sha256")
        missing = [k for k in required if k not in data]
        if missing:
            raise ModelArtifactError(
                f"Model metadata is incomplete (missing: {', '.join(missing)}).")
        class_names = data["class_names"]
        if not isinstance(class_names, list):
            raise ModelArtifactError("Model metadata 'class_names' must be a list.")
        declared = data.get("num_classes")
        if declared is not None and declared != len(class_names):
            raise ModelArtifactError(
                "Model metadata 'num_classes' does not match the class list.")
        try:
            preprocessing = PreprocessingConfig.from_dict(data["preprocessing"],
                                                          "metadata.preprocessing")
        except ConfigError as exc:
            raise ModelArtifactError(f"Model metadata preprocessing is invalid: {exc}") from exc
        return cls(
            model_id=data["model_id"],
            class_names=tuple(class_names),
            preprocessing=preprocessing,
            weights_sha256=data["weights_sha256"],
            architecture=data.get("architecture", "resnet18"),
            task=data.get("task", "classification"),
            scoring=data.get("scoring", "softmax"),
            weights_file=data.get("weights_file", "model_weights.pth"),
            created_at=data.get("created_at", ""),
            framework=dict(data.get("framework") or {}),
            training=dict(data.get("training") or {}),
            dataset=dict(data.get("dataset") or {}),
            evaluation_report=data.get("evaluation_report"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def save_metadata(metadata: ModelMetadata, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise ModelArtifactError(
            f"Refusing to overwrite existing metadata file '{path.name}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def load_metadata(path: str | Path) -> ModelMetadata:
    path = Path(path)
    if not path.is_file():
        raise ModelArtifactError(
            f"Model metadata file '{path.name}' was not found. Train a model or copy the "
            "exported model files into the 'models' folder (see README).",
            detail=str(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"Model metadata file '{path.name}' is not valid JSON.",
                                 detail=repr(exc)) from exc
    return ModelMetadata.from_dict(data)


def verify_weights_file(weights_path: str | Path, metadata: ModelMetadata) -> None:
    """Check that the weights file exists and matches the SHA-256 recorded in the metadata."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise ModelArtifactError(
            f"Model weights file '{weights_path.name}' was not found. Train a model or copy "
            "the exported model files into the 'models' folder (see README).",
            detail=str(weights_path))
    actual = sha256_file(weights_path)
    if actual != metadata.weights_sha256:
        raise ModelArtifactError(
            "The model weights file does not match its metadata (checksum mismatch). "
            "Use the weights and metadata files exported together.",
            detail=f"expected {metadata.weights_sha256}, got {actual}")
