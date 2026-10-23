"""Dataset validation: file integrity, labels, duplicates, split integrity, class counts.

Confirmed errors (``severity == "error"``) are kept separate from heuristic warnings (for
example perceptual-hash matches, which are evidence of visual similarity, not proof of identity).
Nothing is deleted or modified; invalid samples are reported with a reason.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import (VALID_SPLITS, Issue, Sample, count_by_split_and_class,
                      resolve_sample_path)
from .errors import DatasetError, ImageValidationError
from .image_processing import DATASET_EXTENSIONS, DATASET_MAX_PIXELS, decode_image
from .provenance import sha256_bytes
from .splits import check_split_integrity

logger = logging.getLogger(__name__)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
NEAR_DUPLICATE_MAX_IMAGES = 5000
WIDE_ASPECT_RATIO = 2.0

# Issue codes that mean "this file itself is unusable" (used to build a clean manifest).
FILE_LEVEL_ERROR_CODES = {"missing_file", "unsafe_path", "undecodable_image", "invalid_label"}


@dataclass
class ValidationReport:
    images_examined: int = 0
    valid_images: int = 0
    issues: list[Issue] = field(default_factory=list)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    image_modes: dict[str, int] = field(default_factory=dict)
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    near_duplicate_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    near_duplicates_checked: bool = False

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def invalid_sample_ids(self) -> set[str]:
        return {i.sample_id for i in self.errors if i.code in FILE_LEVEL_ERROR_CODES and i.sample_id}

    def to_dict(self) -> dict:
        return {
            "images_examined": self.images_examined,
            "valid_images": self.valid_images,
            "invalid_images": self.images_examined - self.valid_images,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "class_counts_by_split": self.class_counts,
            "image_modes": self.image_modes,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "near_duplicates_checked": self.near_duplicates_checked,
            "near_duplicate_pairs": [list(p) for p in self.near_duplicate_pairs],
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self, title: str = "Dataset validation report") -> str:
        lines = [f"# {title}", "",
                 f"- Images examined: {self.images_examined}",
                 f"- Valid images: {self.valid_images}",
                 f"- Invalid images: {self.images_examined - self.valid_images}",
                 f"- Confirmed errors: {len(self.errors)}",
                 f"- Heuristic warnings: {len(self.warnings)}",
                 f"- Result: {'PASS (no confirmed errors)' if self.ok else 'FAIL (see errors)'}",
                 "", "## Samples per class and split", ""]
        splits = [s for s in (*VALID_SPLITS, "unassigned") if s in self.class_counts]
        classes = sorted({c for per in self.class_counts.values() for c in per})
        lines.append("| class | " + " | ".join(splits) + " |")
        lines.append("|---|" + "---|" * len(splits))
        for c in classes:
            lines.append(f"| {c} | " + " | ".join(
                str(self.class_counts[s].get(c, 0)) for s in splits) + " |")
        lines += ["", "## Confirmed errors", ""]
        lines += [f"- [{i.code}] {i.message}" + (f" (`{i.sample_id}`)" if i.sample_id else "")
                  for i in self.errors] or ["None."]
        lines += ["", "## Warnings (heuristic - review manually)", ""]
        lines += [f"- [{i.code}] {i.message}" + (f" (`{i.sample_id}`)" if i.sample_id else "")
                  for i in self.warnings] or ["None."]
        lines += ["", "Note: a perceptual-hash match is evidence of visual similarity, not proof "
                  "that two files show the same physical sample. Passing validation does not "
                  "establish that the data is representative of factory conditions.", ""]
        return "\n".join(lines)


def dhash64(image: Image.Image) -> int:
    """64-bit difference hash (visual-similarity heuristic)."""
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def find_near_duplicates(hashes: dict[str, int], max_distance: int) -> list[tuple[str, str, int]]:
    """All pairs with Hamming distance <= ``max_distance`` (O(n^2); caller limits n)."""
    ids = sorted(hashes)
    if len(ids) < 2:
        return []
    arr = np.array([hashes[i] for i in ids], dtype=np.uint64)
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(ids) - 1):
        xor = arr[i + 1:] ^ arr[i]
        distance = _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for offset in np.nonzero(distance <= max_distance)[0]:
            pairs.append((ids[i], ids[i + 1 + int(offset)], int(distance[offset])))
    return pairs


def validate_dataset(samples: list[Sample], images_root: str | Path, class_names: list[str], *,
                     near_duplicates: bool = True, near_duplicate_distance: int = 4,
                     extensions: tuple[str, ...] = DATASET_EXTENSIONS,
                     max_pixels: int = DATASET_MAX_PIXELS) -> ValidationReport:
    """Validate every sample and the split structure; never raises for bad *data* (it reports)."""
    if not samples:
        raise DatasetError("The manifest contains no samples.")
    report = ValidationReport(images_examined=len(samples))
    report.class_counts = count_by_split_and_class(samples)
    known = set(class_names)
    modes: Counter = Counter()
    dhashes: dict[str, int] = {}
    wide: list[str] = []
    id_counts = Counter(s.sample_id for s in samples)

    for s in samples:
        if s.class_label not in known:
            report.issues.append(Issue(
                "error", "invalid_label",
                f"Label '{s.class_label}' is not in the class mapping.", s.sample_id))
        try:
            path = resolve_sample_path(images_root, s.relative_image_path)
        except DatasetError as exc:
            report.issues.append(Issue("error", "unsafe_path", exc.user_message, s.sample_id))
            continue
        if not path.is_file():
            report.issues.append(Issue("error", "missing_file",
                                       "Referenced image file does not exist.", s.sample_id))
            continue
        try:
            data = path.read_bytes()
            loaded = decode_image(data, path.name, allowed_extensions=extensions,
                                  max_bytes=None, max_pixels=max_pixels)
        except (OSError, ImageValidationError) as exc:
            msg = exc.user_message if isinstance(exc, ImageValidationError) else \
                "File could not be read."
            report.issues.append(Issue("error", "undecodable_image", msg, s.sample_id))
            continue

        label_valid = s.class_label in known
        if label_valid:
            report.valid_images += 1
        report.hashes[s.sample_id] = sha256_bytes(data)
        modes[loaded.original_mode] += 1
        if max(loaded.width, loaded.height) / min(loaded.width, loaded.height) > WIDE_ASPECT_RATIO:
            wide.append(s.sample_id)
        if near_duplicates:
            dhashes[s.sample_id] = dhash64(loaded.image)

    report.image_modes = dict(modes)
    for sid, count in id_counts.items():
        if count > 1:
            report.issues.append(Issue("error", "duplicate_sample_id",
                                       f"sample_id appears {count} times in the manifest.", sid))

    _exact_duplicates(report, samples)
    _near_duplicates(report, samples, dhashes, near_duplicates, near_duplicate_distance)
    report.issues.extend(check_split_integrity(samples, report.hashes))

    if wide:
        examples = ", ".join(wide[:3])
        report.issues.append(Issue(
            "warning", "extreme_aspect_ratio",
            f"{len(wide)} image(s) have an aspect ratio above {WIDE_ASPECT_RATIO:g}:1 (e.g. "
            f"{examples}). The default 'center_crop' preprocessing keeps only the central square "
            "and may discard most of such images; consider 'resize_mode: squash' or tiling, and "
            "compare against the baseline."))
    no_groups = sum(1 for s in samples if not s.group_id)
    if no_groups == len(samples):
        report.issues.append(Issue(
            "warning", "no_group_metadata",
            "No group_id (specimen/session) metadata is available; only byte-identical "
            "duplicates are kept together. Near-identical images of the same physical sample "
            "could still land in different splits."))
    return report


def _exact_duplicates(report: ValidationReport, samples: list[Sample]) -> None:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        digest = report.hashes.get(s.sample_id)
        if digest:
            by_hash[digest].append(s)
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        ids = sorted(s.sample_id for s in group)
        report.exact_duplicate_groups.append(ids)
        labels = {s.class_label for s in group}
        if len(labels) > 1:
            report.issues.append(Issue(
                "error", "duplicate_label_conflict",
                f"Identical images carry different labels {sorted(labels)}: {', '.join(ids[:4])}"))
        else:
            report.issues.append(Issue(
                "warning", "exact_duplicates",
                f"{len(ids)} byte-identical images (sha256 {digest[:12]}...): "
                f"{', '.join(ids[:4])}"))


def _near_duplicates(report: ValidationReport, samples: list[Sample], dhashes: dict[str, int],
                     enabled: bool, max_distance: int) -> None:
    if not enabled:
        return
    if len(dhashes) > NEAR_DUPLICATE_MAX_IMAGES:
        report.issues.append(Issue(
            "warning", "near_duplicate_check_skipped",
            f"Near-duplicate check skipped: more than {NEAR_DUPLICATE_MAX_IMAGES} images."))
        return
    report.near_duplicates_checked = True
    split_of = {s.sample_id: s.split for s in samples}
    exact = {tuple(g) for g in report.exact_duplicate_groups}
    exact_pairs = {(a, b) for g in exact for a in g for b in g if a < b}
    pairs = [p for p in find_near_duplicates(dhashes, max_distance)
             if (p[0], p[1]) not in exact_pairs]
    report.near_duplicate_pairs = pairs
    cross = [p for p in pairs if split_of.get(p[0]) != split_of.get(p[1])]
    if pairs:
        report.issues.append(Issue(
            "warning", "near_duplicates",
            f"{len(pairs)} visually similar pair(s) (perceptual hash distance <= {max_distance}); "
            "repeated fabric texture can cause false matches, so review manually."))
    for a, b, d in cross[:20]:
        report.issues.append(Issue(
            "warning", "near_duplicate_across_splits",
            f"Visually similar images in different splits (distance {d}): {a} [{split_of.get(a)}] "
            f"vs {b} [{split_of.get(b)}]. Possible leakage - review."))
