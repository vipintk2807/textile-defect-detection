"""Exception hierarchy.

``str(exc)`` / ``exc.user_message`` is written to be safe to show in the user interface
(no stack traces, no absolute paths). ``exc.detail`` carries extra technical information
that is meant for developer logs only.
"""

from __future__ import annotations


class TextileDefectError(Exception):
    """Base class for all expected, user-reportable errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.user_message = message
        self.detail = detail


class ConfigError(TextileDefectError):
    """Invalid or missing configuration."""


class ImageValidationError(TextileDefectError):
    """An input image is missing, unsupported, corrupt or unusable."""


class ModelArtifactError(TextileDefectError):
    """Model weights/metadata are missing, incompatible or cannot be loaded."""


class InferenceError(TextileDefectError):
    """Inference could not be completed or produced invalid output."""


class DatasetError(TextileDefectError):
    """Dataset files, manifests or labels are invalid."""


class TrainingError(TextileDefectError):
    """Training could not be completed."""


class EvaluationError(TextileDefectError):
    """Evaluation inputs or outputs are invalid."""


class StorageError(TextileDefectError):
    """Inspection history or export could not be read/written."""
