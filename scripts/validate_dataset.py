"""Validate a prepared dataset (files, labels, duplicates, split integrity, class counts).

Exit codes: 0 = no confirmed errors, 1 = confirmed errors found, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _common
from textile_defect_detection.dataset import (build_class_mapping, load_class_mapping,
                                              read_samples_csv, write_samples_csv)
from textile_defect_detection.dataset_validation import validate_dataset
from textile_defect_detection.errors import TextileDefectError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="samples.csv")
    p.add_argument("--images-root", required=True)
    p.add_argument("--class-mapping", help="class_mapping.json (default: next to the manifest)")
    p.add_argument("--report-dir", help="write validation_report.json/.md here (new folder)")
    p.add_argument("--no-near-duplicates", action="store_true")
    p.add_argument("--near-duplicate-distance", type=int, default=4)
    p.add_argument("--write-clean-manifest", metavar="DIR",
                   help="write a NEW manifest without unusable files + excluded_samples.csv")
    a = p.parse_args(argv)
    _common.setup_logging()
    try:
        manifest = Path(a.manifest)
        samples = read_samples_csv(manifest)
        mapping_path = Path(a.class_mapping) if a.class_mapping else manifest.parent / "class_mapping.json"
        mapping = (load_class_mapping(mapping_path) if mapping_path.is_file()
                   else build_class_mapping([s.class_label for s in samples]))
        report = validate_dataset(samples, a.images_root, mapping.class_names,
                                  near_duplicates=not a.no_near_duplicates,
                                  near_duplicate_distance=a.near_duplicate_distance)
        if a.report_dir:
            out = Path(a.report_dir)
            if out.exists() and any(out.iterdir()):
                print(f"ERROR: report folder '{out}' is not empty.", file=sys.stderr)
                return 2
            out.mkdir(parents=True, exist_ok=True)
            (out / "validation_report.json").write_text(
                json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
            (out / "validation_report.md").write_text(report.to_markdown(), encoding="utf-8")
        if a.write_clean_manifest:
            out = Path(a.write_clean_manifest)
            bad = report.invalid_sample_ids
            write_samples_csv([s for s in samples if s.sample_id not in bad], out / "samples.csv")
            reasons = {i.sample_id: i.message for i in report.errors if i.sample_id in bad}
            with open(out / "excluded_samples.csv", "w", encoding="utf-8", newline="") as h:
                import csv
                w = csv.writer(h)
                w.writerow(["sample_id", "reason"])
                w.writerows(sorted(reasons.items()))
            print(f"Clean manifest written to {out} ({len(bad)} sample(s) excluded, see excluded_samples.csv).")
    except TextileDefectError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 2

    print(report.to_markdown())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
