"""Inference engine: validated image -> preprocessing -> model -> structured result.

The engine depends on a *predictor* callable (batch ``float32[N,3,H,W]`` -> logits ``[N,C]``),
not on PyTorch directly. :meth:`InferenceEngine.from_config` builds the real PyTorch predictor;
tests may inject a stand-in predictor to exercise everything else. The engine never falls back
to a different model, random weights or a default class mapping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AppConfig, InputConfig
from .errors import InferenceError, ModelArtifactError, TextileDefectError
from .image_processing import (LoadedImage, decode_image, load_image_file, preprocess_image,
                               sanitize_filename)
from .model_metadata import ModelMetadata, load_metadata, verify_weights_file
from .predictions import InspectionResult, build_result, logits_to_prediction

logger = logging.getLogger(__name__)

Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModelInfo:
    """Facts about the loaded model for display in the interface."""

    model_id: str
    architecture: str
    class_names: tuple[str, ...]
    device: str
    created_at: str
    trained_on_synthetic_data: bool
    dataset_name: str
    weights_sha256_short: str


class InferenceEngine:
    def __init__(self, metadata: ModelMetadata, predictor: Predictor, *, input_config: InputConfig,
                 threshold: float, top_k: int | None = None, device: str = "cpu") -> None:
        self.metadata = metadata
        self._predictor = predictor
        self.input_config = input_config
        self.threshold = threshold
        self.top_k = top_k
        self.device = device

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: AppConfig) -> "InferenceEngine":
        """Load metadata and weights named in the config; raise ModelArtifactError on any problem."""
        metadata = load_metadata(config.model.metadata_path)
        if metadata.architecture != config.model.architecture:
            raise ModelArtifactError(
                "The model metadata architecture does not match the configuration "
                f"('{metadata.architecture}' vs '{config.model.architecture}').")
        verify_weights_file(config.model.weights_path, metadata)
        from .model import load_torch_predictor  # imported lazily: needs PyTorch

        predictor = load_torch_predictor(config.model.weights_path, metadata,
                                         device=config.model.device)
        return cls(metadata, predictor, input_config=config.input,
                   threshold=config.inference.confidence_threshold,
                   top_k=config.inference.top_k, device=config.model.device)

    # ------------------------------------------------------------------ information
    @property
    def info(self) -> ModelInfo:
        m = self.metadata
        return ModelInfo(
            model_id=m.model_id,
            architecture=m.architecture,
            class_names=m.class_names,
            device=self.device,
            created_at=m.created_at,
            trained_on_synthetic_data=m.trained_on_synthetic_data,
            dataset_name=str(m.dataset.get("dataset_name", "unknown")),
            weights_sha256_short=m.weights_sha256[:12],
        )

    # ------------------------------------------------------------------ inference
    def inspect_loaded(self, loaded: LoadedImage, filename: str) -> InspectionResult:
        """Run inference on an already validated image."""
        batch = preprocess_image(loaded.image, self.metadata.preprocessing)[np.newaxis, ...]
        start = time.perf_counter()
        try:
            logits = self._predictor(batch)
        except TextileDefectError:
            raise
        except MemoryError as exc:
            raise InferenceError(
                "Not enough memory to analyse this image. Try a smaller image.",
                detail=repr(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert any runtime failure into a safe error
            logger.exception("Predictor raised an unexpected exception")
            raise InferenceError(
                "Analysis could not be completed. Try another image or check the application "
                "logs.", detail=repr(exc)) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prediction = logits_to_prediction(logits, self.metadata.class_names, self.top_k)
        return build_result(
            prediction,
            image_name=sanitize_filename(filename),
            model_id=self.metadata.model_id,
            threshold=self.threshold,
            processing_time_ms=elapsed_ms,
            image_width=loaded.width,
            image_height=loaded.height,
        )

    def inspect_bytes(self, data: bytes, filename: str) -> InspectionResult:
        loaded = decode_image(
            data, filename,
            allowed_extensions=self.input_config.supported_formats,
            max_bytes=self.input_config.max_bytes,
            max_pixels=self.input_config.max_image_pixels,
            min_dimension=self.input_config.min_dimension,
        )
        return self.inspect_loaded(loaded, filename)

    def inspect_path(self, path: str | Path) -> InspectionResult:
        path = Path(path)
        loaded = load_image_file(
            path,
            allowed_extensions=self.input_config.supported_formats,
            max_bytes=self.input_config.max_bytes,
            max_pixels=self.input_config.max_image_pixels,
            min_dimension=self.input_config.min_dimension,
        )
        return self.inspect_loaded(loaded, path.name)
