"""PyTorch model construction, safe weight I/O and the CPU/GPU predictor.

This is the only module (besides ``training.py``) that imports PyTorch. Importing it without
PyTorch installed raises nothing; the functions raise :class:`ModelArtifactError` with
installation guidance when called.

Weights are stored as a plain ``state_dict`` of tensors and loaded with
``torch.load(..., weights_only=True)`` so a malicious pickle cannot execute code on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import ModelArtifactError
from .model_metadata import ModelMetadata

_TORCH_HELP = ("PyTorch/torchvision are not installed. Run 'pip install -r requirements.txt' "
               "inside the project's virtual environment (see README).")


def require_torch():
    """Import and return the ``torch`` module or raise an actionable error."""
    try:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ModelArtifactError(_TORCH_HELP, detail=repr(exc)) from exc
    return torch


def resolve_device(requested: str):
    """Return a ``torch.device``. ``auto`` picks CUDA when available; an explicit ``cuda`` request
    that cannot be met is an error (no silent fallback)."""
    torch = require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelArtifactError(
            "CUDA was requested but is not available. Use device 'cpu' (or 'auto').")
    if requested not in {"cpu", "cuda"}:
        raise ModelArtifactError(f"Unsupported device '{requested}'.")
    return torch.device(requested)


def build_model(architecture: str, num_classes: int, *, pretrained: bool = False,
                pretrained_weights: str = "IMAGENET1K_V1"):
    """Create a ResNet-18 whose final layer has ``num_classes`` outputs."""
    torch = require_torch()
    from torchvision import models  # noqa: PLC0415

    if architecture != "resnet18":
        raise ModelArtifactError(f"Unsupported architecture '{architecture}'.")
    if num_classes < 2:
        raise ModelArtifactError("At least two classes are required.")
    weights = None
    if pretrained:
        if pretrained_weights != "IMAGENET1K_V1":
            raise ModelArtifactError(f"Unknown pretrained weights '{pretrained_weights}'.")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:  # noqa: BLE001 - typically a download failure when offline
        if pretrained:
            raise ModelArtifactError(
                "The pretrained ResNet-18 weights could not be loaded (they are downloaded "
                "from the PyTorch servers on first use and need internet access). Connect to "
                "the internet or set 'model.pretrained: false' to train from scratch.",
                detail=repr(exc)) from exc
        raise
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze everything except the final classification layer."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def set_train_mode(model, freeze_backbone: bool) -> None:
    """Training mode, but keep a frozen backbone (incl. BatchNorm statistics) in eval mode."""
    model.train()
    if freeze_backbone:
        for name, child in model.named_children():
            if name != "fc":
                child.eval()


def save_weights(model, path: str | Path, *, overwrite: bool = False) -> Path:
    torch = require_torch()
    path = Path(path)
    if path.exists() and not overwrite:
        raise ModelArtifactError(f"Refusing to overwrite existing weights file '{path.name}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)
    return path


def load_state_dict_safely(path: str | Path):
    """Load a tensor-only checkpoint; pickled arbitrary objects are rejected."""
    torch = require_torch()
    path = Path(path)
    if not path.is_file():
        raise ModelArtifactError(f"Checkpoint file '{path.name}' was not found.",
                                 detail=str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelArtifactError(
            f"Checkpoint file '{path.name}' could not be loaded safely. It may be corrupt or "
            "not a weights file produced by this project.", detail=repr(exc)) from exc


def load_torch_predictor(weights_path: str | Path, metadata: ModelMetadata, device: str = "cpu"):
    """Build the predictor callable used by :class:`~textile_defect_detection.inference.InferenceEngine`."""
    torch = require_torch()
    torch_device = resolve_device(device)
    state = load_state_dict_safely(weights_path)
    if not isinstance(state, dict):
        raise ModelArtifactError("The weights file does not contain a state dictionary.")
    model = build_model(metadata.architecture, metadata.num_classes, pretrained=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelArtifactError(
            "The weights are incompatible with the model description (architecture or class "
            "count mismatch).", detail=repr(exc)) from exc
    model.to(torch_device)
    model.eval()

    def predict(batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(torch_device)
        with torch.inference_mode():
            logits = model(x)
        return logits.detach().cpu().numpy()

    return predict
