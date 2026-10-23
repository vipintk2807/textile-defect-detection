import shutil

import pytest
from PIL import Image

from helpers import prepared_synthetic_dataset
from textile_defect_detection.dataset_validation import (dhash64, find_near_duplicates,
                                                         validate_dataset)
from textile_defect_detection.errors import DatasetError


def codes(report, severity=None):
    items = report.issues if severity is None else [i for i in report.issues if i.severity == severity]
    return {i.code for i in items}


def test_clean_synthetic_dataset_passes(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    report = validate_dataset(samples, root, mapping.class_names)
    assert report.ok and report.valid_images == len(samples) == 48
    assert report.class_counts["train"]["synthetic_blob"] > 0
    assert "PASS" in report.to_markdown()


def test_missing_file_is_a_confirmed_error_and_reported_with_id(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    (root / samples[0].relative_image_path).unlink()
    report = validate_dataset(samples, root, mapping.class_names)
    assert not report.ok and "missing_file" in codes(report, "error")
    assert samples[0].sample_id in report.invalid_sample_ids
    assert report.valid_images == 47  # invalid samples are counted, not silently dropped


def test_corrupt_image_is_reported(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    (root / samples[3].relative_image_path).write_bytes(b"not an image")
    report = validate_dataset(samples, root, mapping.class_names)
    assert "undecodable_image" in codes(report, "error")


def test_invalid_label_is_reported(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    samples[1].class_label = "mystery_defect"
    report = validate_dataset(samples, root, mapping.class_names)
    assert "invalid_label" in codes(report, "error")


def test_unsafe_manifest_path_is_reported(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    samples[0].relative_image_path = "../../etc/passwd"
    report = validate_dataset(samples, root, mapping.class_names)
    assert "unsafe_path" in codes(report, "error")


def test_exact_duplicate_across_splits_is_a_leakage_error(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    a = next(s for s in samples if s.split == "train")
    b = next(s for s in samples if s.split == "test" and s.class_label == a.class_label)
    shutil.copyfile(root / a.relative_image_path, root / b.relative_image_path)
    report = validate_dataset(samples, root, mapping.class_names)
    assert "duplicate_across_splits" in codes(report, "error")
    assert report.exact_duplicate_groups


def test_duplicate_with_conflicting_labels_is_an_error(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    a = samples[0]
    b = next(s for s in samples if s.class_label != a.class_label)
    shutil.copyfile(root / a.relative_image_path, root / b.relative_image_path)
    report = validate_dataset(samples, root, mapping.class_names)
    assert "duplicate_label_conflict" in codes(report, "error")


def test_near_duplicate_is_only_a_warning(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    a = next(s for s in samples if s.split == "train")
    b = next(s for s in samples if s.split == "test")
    img = Image.open(root / a.relative_image_path).convert("RGB")
    img.putpixel((0, 0), (0, 0, 0))  # not byte-identical, visually the same
    img.save(root / b.relative_image_path)
    report = validate_dataset(samples, root, mapping.class_names)
    assert "near_duplicate_across_splits" in codes(report, "warning")
    assert "near_duplicate_across_splits" not in codes(report, "error")


def test_near_duplicate_check_can_be_disabled(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    report = validate_dataset(samples, root, mapping.class_names, near_duplicates=False)
    assert report.near_duplicates_checked is False


def test_dhash_and_pairing_primitives():
    import numpy as np
    flat = Image.new("RGB", (32, 32), (10, 10, 10))
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    assert dhash64(flat) == dhash64(flat.copy())
    hashes = {"a": dhash64(noise), "b": dhash64(noise.copy()), "c": dhash64(flat)}
    pairs = find_near_duplicates(hashes, max_distance=2)
    assert ("a", "b", 0) in pairs and all("c" not in (p[0], p[1]) for p in pairs)


def test_wide_images_trigger_aspect_ratio_warning(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path, images_per_class=6)
    Image.new("RGB", (400, 40), (50, 60, 70)).save(root / samples[0].relative_image_path)
    report = validate_dataset(samples, root, mapping.class_names)
    assert "extreme_aspect_ratio" in codes(report, "warning")


def test_missing_group_metadata_is_flagged(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path, images_per_class=6)
    for s in samples:
        s.group_id = ""
    report = validate_dataset(samples, root, mapping.class_names)
    assert "no_group_metadata" in codes(report, "warning")


def test_empty_manifest_raises():
    with pytest.raises(DatasetError):
        validate_dataset([], ".", ["a", "b"])


def test_unsplit_manifest_is_flagged_as_invalid_split(tmp_path):
    root, samples, mapping = prepared_synthetic_dataset(tmp_path, images_per_class=6)
    for s in samples:
        s.split = ""
    assert "invalid_split" in codes(validate_dataset(samples, root, mapping.class_names), "error")
