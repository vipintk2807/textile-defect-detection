"""Dataset manifests, class mappings, provenance records and folder scanning.

Source images and annotations are never modified. A prepared dataset is a set of *manifest*
files that reference images by path relative to a documented images root.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from .errors import DatasetError
from .image_processing import DATASET_EXTENSIONS

SAMPLE_FIELDS = ["sample_id", "relative_image_path", "split", "class_label", "source_dataset",
                 "group_id", "source_annotation_reference"]
VALID_SPLITS = ("train", "val", "test")
_SPLIT_FOLDER_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                       "validation": "val", "test": "test", "testing": "test"}


@dataclass
class Sample:
    sample_id: str
    relative_image_path: str
    split: str
    class_label: str
    source_dataset: str = ""
    group_id: str = ""
    source_annotation_reference: str = ""

    @property
    def group_key(self) -> str:
        """Samples that share a group key must never be split apart."""
        return self.group_id or self.sample_id


@dataclass
class Issue:
    """A validation finding. ``error`` = confirmed problem; ``warning`` = heuristic/needs review."""

    severity: str
    code: str
    message: str
    sample_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PREPARATION_RECORD_FIELDS = ("prepared_at", "layout", "split_seed", "split_ratios",
                             "samples_csv_sha256", "class_mapping_sha256",
                             "counts_by_split_and_class")


@dataclass
class DatasetInfo:
    """Provenance record (dataset specification 6.1). Unknown values stay ``"unknown"``."""

    dataset_id: str = "unknown"
    dataset_name: str = "unknown"
    source_url: str = "unknown"
    publisher: str = "unknown"
    version: str = "unknown"
    download_date: str = "unknown"
    license: str = "unknown"
    citation: str = "unknown"
    usage_restrictions: str = "unknown"
    annotation_format: str = "image-level class label (folder name)"
    synthetic: bool = False
    preparation_steps: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset_info(path: str | Path) -> DatasetInfo:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset info file '{path.name}' was not found.", detail=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Dataset info file '{path.name}' is not valid JSON.",
                           detail=repr(exc)) from exc
    if not isinstance(data, dict):
        raise DatasetError(f"Dataset info file '{path.name}' must contain a JSON object.")
    allowed = set(DatasetInfo.__dataclass_fields__)
    # Bookkeeping written by scripts/prepare_dataset.py into dataset_manifest.json. These are
    # recorded for traceability but are not part of the provenance record, so they are ignored.
    prepared_by_tool = set(PREPARATION_RECORD_FIELDS)
    unknown = set(data) - allowed - prepared_by_tool
    if unknown:
        raise DatasetError(f"Unknown field(s) in dataset info: {', '.join(sorted(unknown))}.")
    return DatasetInfo(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------- class mapping
@dataclass
class ClassMapping:
    """Ordered class list (list position = model output index) plus source-label mapping rows."""

    class_names: list[str]
    entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise DatasetError("Class mapping contains duplicate class names.")
        if not self.class_names:
            raise DatasetError("Class mapping is empty.")

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.class_names)}

    def to_dict(self) -> dict:
        return {"classes": [{"index": i, "name": n} for i, n in enumerate(self.class_names)],
                "mappings": self.entries}


def build_class_mapping(source_labels: list[str], label_map: dict[str, str] | None = None,
                        annotation_type: str = "image-level label",
                        source_reference: str = "") -> ClassMapping:
    """Build a deterministic mapping (project labels sorted alphabetically).

    Without ``label_map`` every source label becomes a project label unchanged; categories are
    never merged unless the caller supplies an explicit map.
    """
    label_map = label_map or {}
    entries = []
    for source in sorted(set(source_labels)):
        project = label_map.get(source, source)
        entries.append({
            "source_label": source,
            "project_label": project,
            "definition": "",
            "mapping_rule": "explicit label map" if source in label_map else "identity (not merged)",
            "annotation_type": annotation_type,
            "source_reference": source_reference,
        })
    names = sorted({e["project_label"] for e in entries})
    return ClassMapping(names, entries)


def save_class_mapping(mapping: ClassMapping, path: str | Path, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise DatasetError(f"Refusing to overwrite existing file '{path.name}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping.to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def load_class_mapping(path: str | Path) -> ClassMapping:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Class mapping file '{path.name}' was not found.", detail=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        classes = sorted(data["classes"], key=lambda c: c["index"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(f"Class mapping file '{path.name}' is malformed.",
                           detail=repr(exc)) from exc
    if [c["index"] for c in classes] != list(range(len(classes))):
        raise DatasetError("Class mapping indices must be 0..N-1 without gaps.")
    return ClassMapping([c["name"] for c in classes], list(data.get("mappings", [])))


# --------------------------------------------------------------------------- manifests
def read_samples_csv(path: str | Path) -> list[Sample]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Manifest file '{path.name}' was not found.", detail=str(path))
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [f for f in ("sample_id", "relative_image_path", "split", "class_label")
                       if f not in (reader.fieldnames or [])]
            if missing:
                raise DatasetError(
                    f"Manifest '{path.name}' is missing required column(s): {', '.join(missing)}.")
            samples = []
            for line_no, row in enumerate(reader, start=2):
                values = {k: (row.get(k) or "").strip() for k in SAMPLE_FIELDS}
                if not values["sample_id"] or not values["relative_image_path"]:
                    raise DatasetError(
                        f"Manifest '{path.name}' line {line_no}: sample_id and "
                        "relative_image_path are required.")
                samples.append(Sample(**values))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise DatasetError(f"Manifest '{path.name}' could not be read.", detail=repr(exc)) from exc
    return samples


def write_samples_csv(samples: list[Sample], path: str | Path, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise DatasetError(f"Refusing to overwrite existing file '{path.name}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: getattr(s, k) for k in SAMPLE_FIELDS})


def resolve_sample_path(images_root: str | Path, relative_path: str) -> Path:
    """Join ``relative_path`` to the images root and refuse paths that escape it."""
    root = Path(images_root).resolve()
    rel = PurePosixPath(relative_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise DatasetError("A manifest path is absolute or contains '..'.",
                           detail=relative_path)
    candidate = (root / Path(*rel.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DatasetError("A manifest path points outside the images root.",
                           detail=relative_path) from exc
    return candidate


# --------------------------------------------------------------------------- scanning folders
def _iter_images(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    wanted = {f".{e}" for e in extensions}
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in wanted)


def _group_for(stem: str, pattern: re.Pattern | None) -> str:
    if pattern is None:
        return ""
    match = pattern.search(stem)
    if not match:
        return ""
    return match.group(1) if pattern.groups else match.group(0)


def scan_class_folders(root: str | Path, *, source_dataset: str,
                       extensions: tuple[str, ...] = DATASET_EXTENSIONS,
                       group_regex: str | None = None) -> list[Sample]:
    """Scan ``root/<class>/**/image`` into samples without a split (split is assigned later)."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset folder '{root.name}' was not found.", detail=str(root))
    pattern = re.compile(group_regex) if group_regex else None
    samples: list[Sample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in _iter_images(class_dir, extensions):
            rel = image.relative_to(root).as_posix()
            samples.append(Sample(
                sample_id=rel, relative_image_path=rel, split="",
                class_label=class_dir.name, source_dataset=source_dataset,
                group_id=_group_for(image.stem, pattern),
                source_annotation_reference=f"folder:{class_dir.name}"))
    if not samples:
        raise DatasetError(f"No images found under '{root.name}' (expected <class>/<image> files).")
    return samples


def scan_split_folders(root: str | Path, *, source_dataset: str,
                       extensions: tuple[str, ...] = DATASET_EXTENSIONS,
                       group_regex: str | None = None) -> list[Sample]:
    """Scan an official-split layout ``root/{train,val,test}/<class>/image``."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset folder '{root.name}' was not found.", detail=str(root))
    pattern = re.compile(group_regex) if group_regex else None
    samples: list[Sample] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = _SPLIT_FOLDER_NAMES.get(split_dir.name.lower())
        if split is None:
            raise DatasetError(
                f"Unexpected folder '{split_dir.name}' in a split-folders dataset "
                "(expected train/val/test).")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image in _iter_images(class_dir, extensions):
                rel = image.relative_to(root).as_posix()
                samples.append(Sample(
                    sample_id=rel, relative_image_path=rel, split=split,
                    class_label=class_dir.name, source_dataset=source_dataset,
                    group_id=_group_for(image.stem, pattern),
                    source_annotation_reference=f"folder:{split_dir.name}/{class_dir.name}"))
    if not samples:
        raise DatasetError(f"No images found under '{root.name}'.")
    return samples


def count_by_split_and_class(samples: list[Sample]) -> dict[str, dict[str, int]]:
    """``{split: {class_label: count}}`` including every split present."""
    counts: dict[str, dict[str, int]] = {}
    for s in samples:
        counts.setdefault(s.split or "unassigned", {})
        counts[s.split or "unassigned"][s.class_label] = \
            counts[s.split or "unassigned"].get(s.class_label, 0) + 1
    return counts
