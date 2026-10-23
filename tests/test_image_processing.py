import io
import random

import numpy as np
import pytest
from PIL import Image

from helpers import image_bytes, noisy_image_bytes
from textile_defect_detection.config import AugmentationConfig, PreprocessingConfig
from textile_defect_detection.errors import ImageValidationError
from textile_defect_detection.image_processing import (augment_image, decode_image,
                                                       preprocess_image, sanitize_filename)

ALLOWED = ("jpg", "jpeg", "png", "webp")


def decode(data, name="x.png", **kw):
    params = dict(allowed_extensions=ALLOWED, max_bytes=1_000_000, max_pixels=1_000_000)
    params.update(kw)
    return decode_image(data, name, **params)


@pytest.mark.parametrize("fmt,name", [("PNG", "a.png"), ("JPEG", "a.jpg"), ("WEBP", "a.webp")])
def test_valid_formats_decode_to_rgb(fmt, name):
    loaded = decode(image_bytes(fmt, size=(64, 48)), name)
    assert loaded.image.mode == "RGB" and (loaded.width, loaded.height) == (64, 48)
    assert loaded.original_format == fmt


def test_empty_oversized_corrupt_and_wrong_type_are_rejected():
    with pytest.raises(ImageValidationError, match="No image data"):
        decode(b"")
    with pytest.raises(ImageValidationError, match="larger than"):
        decode(image_bytes(), max_bytes=10)
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        decode(b"this is not an image")
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        decode(image_bytes(), "x.gif")
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        decode(image_bytes(), "noextension")


def test_truncated_image_is_rejected():
    data = noisy_image_bytes(seed=1, size=(128, 128))
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        decode(data[: len(data) // 2])


def test_content_is_checked_not_just_extension():
    # a BMP renamed to .png: content format BMP is not in the allowed set
    with pytest.raises(ImageValidationError, match="content is not a supported"):
        decode(image_bytes("BMP"), "fake.png")


def test_dimension_limits():
    with pytest.raises(ImageValidationError, match="too large"):
        decode(image_bytes(size=(200, 200)), max_pixels=10_000)
    with pytest.raises(ImageValidationError, match="too small"):
        decode(image_bytes(size=(8, 8)), min_dimension=16)


def test_grayscale_and_transparency_are_handled_explicitly():
    gray = decode(image_bytes("PNG", mode="L"))
    assert gray.image.mode == "RGB" and gray.original_mode == "L"
    rgba = Image.new("RGBA", (20, 20), (255, 0, 0, 0))  # fully transparent red
    buffer = io.BytesIO()
    rgba.save(buffer, "PNG")
    loaded = decode(buffer.getvalue())
    assert loaded.image.getpixel((5, 5)) == (255, 255, 255)  # composited onto white
    assert loaded.original_mode == "RGBA"


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\x\\a.png") == "a.png"
    assert sanitize_filename("a\x00b\n.png") == "ab.png"
    assert sanitize_filename("") == "unnamed" and sanitize_filename(None) == "unnamed"
    assert len(sanitize_filename("a" * 1000)) == 255


def test_preprocess_shape_dtype_and_determinism():
    cfg = PreprocessingConfig()
    img = decode(noisy_image_bytes(2, (300, 120))).image
    first, second = preprocess_image(img, cfg), preprocess_image(img, cfg)
    assert first.shape == (3, cfg.crop_size, cfg.crop_size) and first.dtype == np.float32
    assert np.all(np.isfinite(first)) and np.array_equal(first, second)


def test_preprocess_normalisation_matches_manual_computation():
    cfg = PreprocessingConfig(resize_size=16, crop_size=16, resize_mode="squash")
    img = Image.new("RGB", (16, 16), (255, 0, 51))
    out = preprocess_image(img, cfg)
    expected = [(c / 255.0 - m) / s for c, m, s in zip((255, 0, 51), cfg.mean, cfg.std)]
    for channel in range(3):
        assert abs(float(out[channel, 3, 3]) - expected[channel]) < 1e-5


def test_center_crop_keeps_centre_and_squash_keeps_everything():
    img = Image.new("RGB", (200, 50), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 20, 50))  # white strip at the far left edge only
    crop = preprocess_image(img, PreprocessingConfig(resize_size=32, crop_size=32))
    squash = preprocess_image(img, PreprocessingConfig(resize_size=32, crop_size=32,
                                                       resize_mode="squash"))
    assert np.ptp(crop, axis=(1, 2)).max() < 1e-4   # centre crop lost the edge strip entirely
    assert np.ptp(squash, axis=(1, 2)).min() > 1.0  # squash kept it


def test_augmentation_is_seeded_does_not_mutate_and_keeps_size():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8))
    before = np.asarray(img).copy()
    cfg = AugmentationConfig(rotation_degrees=10, brightness=0.3, contrast=0.3, vertical_flip=True)
    a = augment_image(img, cfg, random.Random("s"))
    b = augment_image(img, cfg, random.Random("s"))
    assert np.array_equal(np.asarray(a), np.asarray(b)) and a.size == img.size
    assert np.array_equal(np.asarray(img), before)  # original untouched
    off = augment_image(img, AugmentationConfig(enabled=False), random.Random(0))
    assert np.array_equal(np.asarray(off), before)


def test_evaluation_preprocessing_has_no_randomness():
    cfg = PreprocessingConfig(resize_size=40, crop_size=32)
    img = decode(noisy_image_bytes(5)).image
    outputs = [preprocess_image(img, cfg) for _ in range(3)]
    assert all(np.array_equal(outputs[0], o) for o in outputs[1:])
