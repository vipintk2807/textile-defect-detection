"""Build manifests (samples.csv, class_mapping.json, dataset_manifest.json) from image folders.

Source images/annotations are only read. Layouts:
  class-folders : <root>/<class>/<image>            (this tool assigns group-aware splits)
  split-folders : <root>/{train,val,test}/<class>/<image>   (official split is preserved)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _common
from textile_defect_detection.dataset import (DatasetInfo, build_class_mapping, count_by_split_and_class,
                                              load_dataset_info, save_class_mapping,
                                              scan_class_folders, scan_split_folders,
                                              write_samples_csv)
from textile_defect_detection.errors import TextileDefectError
from textile_defect_detection.provenance import sha256_bytes, sha256_file, utc_now_iso
from textile_defect_detection.splits import assign_splits, check_split_integrity
from textile_defect_detection.dataset import resolve_sample_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", required=True, help="folder with the class (or split) folders")
    p.add_argument("--output-dir", required=True, help="NEW folder for the manifests")
    p.add_argument("--layout", choices=["class-folders", "split-folders"], default="class-folders")
    p.add_argument("--dataset-info", help="JSON provenance file (default: <source-root>/dataset_info.json if present)")
    p.add_argument("--group-regex", help="regex whose 1st group extracts a specimen/group id from the file name stem")
    p.add_argument("--label-map", help="JSON {source_label: project_label}; categories are never merged unless listed")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    _common.setup_logging(a.verbose)

    try:
        root = Path(a.source_root)
        info_path = Path(a.dataset_info) if a.dataset_info else root / "dataset_info.json"
        info = load_dataset_info(info_path) if info_path.is_file() else DatasetInfo()
        if not info_path.is_file():
            print("WARNING: no dataset info file; provenance/licence fields are 'unknown'. "
                  "Fill them in before publishing results.")
        scan = scan_split_folders if a.layout == "split-folders" else scan_class_folders
        samples = scan(root, source_dataset=info.dataset_id, group_regex=a.group_regex)
        label_map = json.loads(Path(a.label_map).read_text(encoding="utf-8")) if a.label_map else None
        if label_map:
            for s in samples:
                s.class_label = label_map.get(s.class_label, s.class_label)
        mapping = build_class_mapping([s.class_label for s in samples],
                                      source_reference=info.dataset_name)

        hashes = {s.sample_id: sha256_bytes(resolve_sample_path(root, s.relative_image_path).read_bytes())
                  for s in samples}
        warnings = []
        if a.layout == "class-folders":
            result = assign_splits(samples, ratios=tuple(a.ratios), seed=a.seed, hashes=hashes)
            samples, warnings = result.samples, result.warnings
        if not a.group_regex:
            warnings.append("No --group-regex given: only byte-identical duplicates are grouped. "
                            "Near-identical images of one physical sample may leak across splits.")
        errors = [i for i in check_split_integrity(samples, hashes) if i.severity == "error"]
        if errors:
            for i in errors:
                print(f"ERROR: {i.message}", file=sys.stderr)
            return 1

        out = Path(a.output_dir)
        if out.exists() and any(out.iterdir()):
            print(f"ERROR: output folder '{out}' is not empty; refusing to overwrite.", file=sys.stderr)
            return 1
        write_samples_csv(samples, out / "samples.csv")
        save_class_mapping(mapping, out / "class_mapping.json")
        record = info.to_dict()
        record.update({"prepared_at": utc_now_iso(), "layout": a.layout, "split_seed": a.seed,
                       "split_ratios": a.ratios if a.layout == "class-folders" else "official",
                       "samples_csv_sha256": sha256_file(out / "samples.csv"),
                       "class_mapping_sha256": sha256_file(out / "class_mapping.json"),
                       "counts_by_split_and_class": count_by_split_and_class(samples)})
        record["preparation_steps"] = list(info.preparation_steps) + [
            f"scanned {a.layout} layout", "assigned splits" if a.layout == "class-folders"
            else "kept official split"]
        (out / "dataset_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except TextileDefectError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1

    print(f"Prepared {len(samples)} samples -> {out}")
    for split, per in sorted(count_by_split_and_class(samples).items()):
        print(f"  {split}: {dict(sorted(per.items()))}")
    for w in warnings:
        print(f"WARNING: {w}")
    print("Next: python scripts/validate_dataset.py --manifest "
          f"{out / 'samples.csv'} --images-root {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
