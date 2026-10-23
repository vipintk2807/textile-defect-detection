from collections import Counter

import pytest

from helpers import prepared_synthetic_dataset
from textile_defect_detection.dataset import (ClassMapping, Sample, build_class_mapping,
                                              load_class_mapping, read_samples_csv,
                                              resolve_sample_path, save_class_mapping,
                                              scan_class_folders, scan_split_folders,
                                              write_samples_csv)
from textile_defect_detection.errors import DatasetError
from textile_defect_detection.splits import assign_splits, check_split_integrity


def sample(sid, label="a", split="", group="", ):
    return Sample(sid, sid, split, label, "src", group)


def test_manifest_round_trip_and_no_overwrite(tmp_path):
    items = [sample("x/1.png", "a", "train", "g1"), sample("x/2.png", "b", "val")]
    write_samples_csv(items, tmp_path / "s.csv")
    assert read_samples_csv(tmp_path / "s.csv") == items
    with pytest.raises(DatasetError, match="overwrite"):
        write_samples_csv(items, tmp_path / "s.csv")


def test_manifest_missing_columns_and_file(tmp_path):
    (tmp_path / "bad.csv").write_text("sample_id,split\n1,train\n")
    with pytest.raises(DatasetError, match="missing required column"):
        read_samples_csv(tmp_path / "bad.csv")
    with pytest.raises(DatasetError, match="not found"):
        read_samples_csv(tmp_path / "none.csv")


def test_class_mapping_is_deterministic_never_merges_and_round_trips(tmp_path):
    mapping = build_class_mapping(["b_stain", "a_hole", "b_stain", "c_thread"])
    assert mapping.class_names == ["a_hole", "b_stain", "c_thread"]
    assert {e["mapping_rule"] for e in mapping.entries} == {"identity (not merged)"}
    save_class_mapping(mapping, tmp_path / "m.json")
    assert load_class_mapping(tmp_path / "m.json").class_names == mapping.class_names
    merged = build_class_mapping(["x", "y"], {"y": "x"})
    assert merged.class_names == ["x"]  # merging happens only through an explicit map
    with pytest.raises(DatasetError):
        ClassMapping(["a", "a"])


def test_path_traversal_is_refused(tmp_path):
    for bad in ("../outside.png", "/etc/passwd", "a/../../b.png"):
        with pytest.raises(DatasetError):
            resolve_sample_path(tmp_path, bad)
    assert resolve_sample_path(tmp_path, "a/b.png").name == "b.png"


def test_scan_class_folders_and_group_regex(tmp_path):
    root, samples, _ = prepared_synthetic_dataset(tmp_path, images_per_class=6)
    assert len(samples) == 24 and all(s.group_id.startswith("g") for s in samples)
    with pytest.raises(DatasetError, match="No images"):
        (tmp_path / "empty").mkdir()
        scan_class_folders(tmp_path / "empty", source_dataset="x")


def test_scan_split_folders_preserves_official_split(tmp_path):
    for split, cls in (("train", "a"), ("val", "a"), ("test", "b"), ("test", "a")):
        d = tmp_path / split / cls
        d.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.new("RGB", (8, 8)).save(d / f"{split}_{cls}.png")
    samples = scan_split_folders(tmp_path, source_dataset="x")
    assert Counter(s.split for s in samples) == {"train": 1, "val": 1, "test": 2}
    (tmp_path / "weird").mkdir()
    with pytest.raises(DatasetError, match="Unexpected folder"):
        scan_split_folders(tmp_path, source_dataset="x")


# ------------------------------------------------------------------ splits
def groups_of(n_groups, per_group, label):
    return [sample(f"{label}{g}_{k}", label, group=f"{label}{g}")
            for g in range(n_groups) for k in range(per_group)]


def test_groups_never_span_splits_and_every_class_appears_everywhere():
    samples = groups_of(10, 3, "a") + groups_of(10, 3, "b")
    result = assign_splits(samples, seed=3)
    by_group = {}
    for s in result.samples:
        by_group.setdefault(s.group_id, set()).add(s.split)
    assert all(len(v) == 1 for v in by_group.values())
    for label in "ab":
        assert {s.split for s in result.samples if s.class_label == label} == {"train", "val", "test"}
    assert not [i for i in check_split_integrity(result.samples) if i.severity == "error"]


def test_split_is_deterministic_and_seed_dependent():
    samples = groups_of(12, 2, "a") + groups_of(12, 2, "b")
    a = assign_splits(samples, seed=1).samples
    b = assign_splits(samples, seed=1).samples
    c = assign_splits(samples, seed=2).samples
    assert [s.split for s in a] == [s.split for s in b]
    assert [s.split for s in a] != [s.split for s in c]


def test_approximate_proportions():
    result = assign_splits(groups_of(40, 1, "a"), seed=0)
    counts = Counter(s.split for s in result.samples)
    assert counts["train"] == 28 and counts["val"] == 6 and counts["test"] == 6


def test_tiny_classes_are_reported_not_padded():
    samples = groups_of(1, 2, "rare") + groups_of(10, 2, "common")
    result = assign_splits(samples, seed=0)
    rare = {s.split for s in result.samples if s.class_label == "rare"}
    assert rare == {"train"} and any("rare" in w for w in result.warnings)


def test_identical_files_are_kept_together_via_hashes():
    samples = [sample(f"s{i}", "a") for i in range(12)]
    hashes = {s.sample_id: f"{i % 6:064x}" for i, s in enumerate(samples)}  # pairs identical
    result = assign_splits(samples, seed=5, hashes=hashes)
    split_of = {s.sample_id: s.split for s in result.samples}
    for i in range(6):
        assert split_of[f"s{i}"] == split_of[f"s{i + 6}"]
    assert not [i for i in check_split_integrity(result.samples, hashes) if i.severity == "error"]


def test_integrity_checker_detects_leakage_and_structure_problems():
    leaky = [sample("1", "a", "train", "g"), sample("2", "a", "test", "g")]
    assert "group_spans_splits" in {i.code for i in check_split_integrity(leaky)}
    dup_content = [sample("1", "a", "train"), sample("2", "a", "test")]
    issues = check_split_integrity(dup_content, {"1": "h", "2": "h"})
    assert "duplicate_across_splits" in {i.code for i in issues}
    bad = [sample("1", "a", "training"), sample("1", "a", "test")]
    codes = {i.code for i in check_split_integrity(bad)}
    assert {"invalid_split", "sample_in_multiple_splits"} <= codes
    missing = [sample("1", "a", "train"), sample("2", "b", "train"), sample("3", "a", "val"),
               sample("4", "a", "test")]
    assert "class_missing_from_split" in {i.code for i in check_split_integrity(missing)}


def test_bad_ratios_and_empty_input():
    with pytest.raises(DatasetError):
        assign_splits(groups_of(5, 1, "a"), ratios=(0.5, 0.5, 0.5))
    with pytest.raises(DatasetError):
        assign_splits([])


def test_mixed_label_group_is_kept_together_with_warning():
    samples = [sample("1", "a", group="g"), sample("2", "b", group="g")] + groups_of(8, 1, "a") \
        + groups_of(8, 1, "b")
    result = assign_splits(samples, seed=0)
    assert any("multiple class labels" in w for w in result.warnings)
    split_of = {s.sample_id: s.split for s in result.samples}
    assert split_of["1"] == split_of["2"]
