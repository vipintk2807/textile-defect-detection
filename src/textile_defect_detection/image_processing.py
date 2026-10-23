"""Image validation, decoding, preprocessing and training augmentation.

Everything here uses Pillow and NumPy only (no PyTorch), so the exact same code path is used by
training, evaluation and the application. Preprocessing never modifies the original image.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from .config import AugmentationConfig, PreprocessingConfig
from .errors import ImageValidationError

# extension -> Pillow format name. The app only exposes the first four (see config.APP_IMAGE_FORMATS).
FORMAT_BY_EXTENSION = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tif": "TIFF",
    "tiff": "TIFF",
}
DATASET_EXTENSIONS = tuple(FORMAT_BY_EXTENSION)
DATASET_MAX_PIXELS = 200_000_000

_SUPPORTED_MODES = {"1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK", "I;16"}
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class LoadedImage:
    """A validated image converted to RGB, plus facts about the original file."""

    image: Image.Image
    width: int
    height: int
    original_mode: str
    original_format: str


def sanitize_filename(name: str | None, max_length: int = 255) -> str:
    """Return a safe display/storage name: base name only, no control characters."""
    if not name:
        return "unnamed"
    base = re.split(r"[\\/]", str(name))[-1]
    base = _CONTROL_CHARS.sub("", base).strip()
    if base in {"", ".", ".."}:
        return "unnamed"
    return base[:max_length]


def _format_list(extensions) -> str:
    unique = []
    for ext in extensions:
        label = ext.upper()
        if label not in unique:
            unique.append(label)
    return ", ".join(unique)


def decode_image(
    data: bytes,
    filename: str,
    *,
    allowed_extensions: tuple[str, ...],
    max_bytes: int | None,
    max_pixels: int,
    min_dimension: int = 1,
) -> LoadedImage:
    """Validate ``data`` by decoding it and return an RGB image.

    The *content* is decoded (the file extension alone is never trusted). Raises
    :class:`ImageValidationError` with a message that is safe to show to users.
    """
    if not data:
        raise ImageValidationError("No image data was provided. Please select an image file.")
    if max_bytes is not None and len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ImageValidationError(f"The file is larger than the {limit_mb:g} MB limit.")

    extension = Path(sanitize_filename(filename)).suffix.lower().lstrip(".")
    if extension not in allowed_extensions:
        raise ImageValidationError(
            "Unsupported file type. Please select a supported image "
            f"({_format_list(allowed_extensions)})."
        )
    allowed_formats = {FORMAT_BY_EXTENSION[e] for e in allowed_extensions}

    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as exc:
        raise ImageValidationError("The selected file could not be decoded as an image.",
                                   detail=repr(exc)) from exc

    if img.format not in allowed_formats:
        raise ImageValidationError(
            "The file content is not a supported image type "
            f"({_format_list(allowed_extensions)})."
        )
    width, height = img.size
    if width <= 0 or height <= 0 or min(width, height) < min_dimension:
        raise ImageValidationError(
            f"The image is too small (minimum {min_dimension} pixels on each side)."
        )
    if width * height > max_pixels:
        raise ImageValidationError(
            "The image dimensions are too large to process safely "
            f"(limit {max_pixels:,} pixels)."
        )
    if img.mode not in _SUPPORTED_MODES:
        raise ImageValidationError(f"Unsupported image channel configuration ('{img.mode}').")

    original_mode, original_format = img.mode, str(img.format)
    try:
        img.load()  # full decode: raises for truncated/corrupt data
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The selected file could not be decoded as an image.",
                                   detail=repr(exc)) from exc

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - malformed EXIF must not block an otherwise valid image
        pass

    rgb = _to_rgb(img)
    return LoadedImage(rgb, rgb.width, rgb.height, original_mode, original_format)


def _to_rgb(img: Image.Image) -> Image.Image:
    """Explicit colour handling: transparency is composited onto white, 16-bit scaled to 8-bit."""
    mode = img.mode
    if mode == "RGB":
        return img.copy()
    has_transparency = mode in {"RGBA", "LA", "PA"} or (mode == "P" and "transparency" in img.info)
    if has_transparency:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if mode == "I;16":
        arr = np.asarray(img, dtype=np.float64) / 257.0
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    return img.convert("RGB")


def load_image_file(path: str | Path, *, allowed_extensions: tuple[str, ...] = DATASET_EXTENSIONS,
                    max_bytes: int | None = None, max_pixels: int = DATASET_MAX_PIXELS,
                    min_dimension: int = 1) -> LoadedImage:
    """Read and validate an image from disk (used by dataset tools, training and evaluation)."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageValidationError(f"Image file '{path.name}' could not be read.",
                                   detail=repr(exc)) from exc
    return decode_image(data, path.name, allowed_extensions=allowed_extensions,
                        max_bytes=max_bytes, max_pixels=max_pixels, min_dimension=min_dimension)


def preprocess_image(image: Image.Image, cfg: PreprocessingConfig) -> np.ndarray:
    """Deterministic preprocessing -> float32 array of shape (3, crop, crop).

    Follows the documented ResNet-18 ImageNet recipe (bilinear resize, centre crop, scale to
    [0, 1], mean/std normalisation) implemented with Pillow/NumPy. It is not claimed to be
    bit-identical to torchvision's transforms.
    """
    img = image if image.mode == "RGB" else image.convert("RGB")
    if cfg.resize_mode == "squash":
        img = img.resize((cfg.crop_size, cfg.crop_size), Image.Resampling.BILINEAR)
    else:
        width, height = img.size
        short, long_ = (width, height) if width <= height else (height, width)
        new_long = max(cfg.resize_size, int(cfg.resize_size * long_ / short))
        new_size = (cfg.resize_size, new_long) if width <= height else (new_long, cfg.resize_size)
        img = img.resize(new_size, Image.Resampling.BILINEAR)
        new_w, new_h = img.size
        left = int(round((new_w - cfg.crop_size) / 2.0))
        top = int(round((new_h - cfg.crop_size) / 2.0))
        img = img.crop((left, top, left + cfg.crop_size, top + cfg.crop_size))

    array = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.asarray(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(cfg.std, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    result = np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
    expected = (3, cfg.crop_size, cfg.crop_size)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ImageValidationError("The image could not be converted to the model input format.")
    return result


def augment_image(image: Image.Image, cfg: AugmentationConfig, rng: random.Random) -> Image.Image:
    """Random training augmentation. Returns a new image; the input is never modified."""
    img = image.copy()
    if not cfg.enabled:
        return img
    if cfg.horizontal_flip and rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if cfg.vertical_flip and rng.random() < 0.5:
        img = ImageOps.flip(img)
    if cfg.rotation_degrees > 0:
        angle = rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
        mean_colour = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(axis=0))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=mean_colour)
    if cfg.brightness > 0:
        img = ImageEnhance.Brightness(img).enhance(
            rng.uniform(1 - cfg.brightness, 1 + cfg.brightness))
    if cfg.contrast > 0:
        img = ImageEnhance.Contrast(img).enhance(
            rng.uniform(1 - cfg.contrast, 1 + cfg.contrast))
    return img
