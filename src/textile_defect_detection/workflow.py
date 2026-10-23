"""UI-independent application workflow (everything the Streamlit app does, minus rendering).

Keeping this separate from ``app.py`` means the upload -> validate -> infer -> (optionally) save
logic is covered by ordinary automated tests, and the presentation layer contains no model logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, InputConfig, load_app_config
from .errors import ConfigError, StorageError, TextileDefectError
from .image_processing import LoadedImage, decode_image, sanitize_filename
from .inference import InferenceEngine
from .inspection_store import InspectionStore
from .predictions import InspectionResult

logger = logging.getLogger(__name__)

SETUP_HINT = ("Train a model (see README, 'Training') or copy an exported model's "
              "'model_weights.pth' and 'model_metadata.json' into the 'models' folder, then "
              "reload the app.")


@dataclass(frozen=True)
class ModelState:
    """Outcome of trying to bring the model up. ``ready`` is only True with a loaded engine."""

    ready: bool
    message: str
    config: AppConfig | None = None
    engine: InferenceEngine | None = None


def load_model_state(config_path: str | Path) -> ModelState:
    """Load config + model. Never raises for expected problems and never substitutes a model."""
    try:
        config = load_app_config(config_path)
    except ConfigError as exc:
        logger.error("Configuration problem: %s (%s)", exc.user_message, exc.detail)
        return ModelState(False, f"Configuration problem: {exc.user_message}")
    try:
        engine = InferenceEngine.from_config(config)
    except TextileDefectError as exc:
        logger.error("Model could not be loaded: %s (%s)", exc.user_message, exc.detail)
        return ModelState(False, f"{exc.user_message} {SETUP_HINT}", config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while loading the model")
        return ModelState(False, "A compatible trained model could not be loaded. Check the "
                          "model configuration and the application logs.", config=config)
    return ModelState(True, "Model ready", config=config, engine=engine)


def validate_upload(data: bytes | None, filename: str | None,
                    input_config: InputConfig) -> LoadedImage:
    """Validate an uploaded file for preview/analysis (raises ImageValidationError)."""
    return decode_image(
        data or b"", sanitize_filename(filename),
        allowed_extensions=input_config.supported_formats,
        max_bytes=input_config.max_bytes,
        max_pixels=input_config.max_image_pixels,
        min_dimension=input_config.min_dimension,
    )


@dataclass(frozen=True)
class AnalysisOutcome:
    """Result of one analysis attempt. Exactly one of ``result`` / ``error_message`` is set."""

    result: InspectionResult | None = None
    error_message: str | None = None
    saved: bool | None = None          # None = history not requested
    save_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


def run_analysis(engine: InferenceEngine | None, data: bytes, filename: str, *,
                 store: InspectionStore | None = None, save_history: bool = False,
                 loaded: LoadedImage | None = None) -> AnalysisOutcome:
    """Analyse one image. Expected failures become an error outcome - never a fake prediction."""
    if engine is None:
        return AnalysisOutcome(error_message="Inference is unavailable because no compatible "
                                              "trained model is loaded.")
    safe_name = sanitize_filename(filename)
    try:
        image = loaded or validate_upload(data, safe_name, engine.input_config)
        result = engine.inspect_loaded(image, safe_name)
    except TextileDefectError as exc:
        logger.warning("Analysis failed: %s (%s)", exc.user_message, exc.detail)
        outcome = AnalysisOutcome(error_message=exc.user_message)
        if save_history and store is not None:
            try:
                store.add_failure(image_name=safe_name, model_id=engine.metadata.model_id,
                                  error_message=exc.user_message)
            except StorageError:
                pass  # the failure itself is already reported to the user
        return outcome
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected analysis failure")
        return AnalysisOutcome(error_message="Analysis could not be completed. Try another "
                                              "image or check the application logs.")

    if not save_history:
        return AnalysisOutcome(result=result)
    if store is None:
        return AnalysisOutcome(result=result, saved=False,
                               save_error="Inspection history is unavailable.")
    try:
        store.add(result)
    except StorageError as exc:
        logger.error("History write failed: %s", exc.detail)
        return AnalysisOutcome(result=result, saved=False, save_error=exc.user_message)
    return AnalysisOutcome(result=result, saved=True)


def open_store(config: AppConfig) -> tuple[InspectionStore | None, str | None]:
    """Open the history store; returns (store, None) or (None, user-facing problem)."""
    try:
        return InspectionStore(config.storage.database_path), None
    except StorageError as exc:
        logger.error("History unavailable: %s", exc.detail)
        return None, exc.user_message
