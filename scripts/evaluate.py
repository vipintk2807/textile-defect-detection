"""Evaluate a trained model on one manifest split and write a traceable report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common
from textile_defect_detection.dataset import load_dataset_info, read_samples_csv
from textile_defect_detection.errors import TextileDefectError
from textile_defect_detection.evaluation import run_evaluation
from textile_defect_detection.model_metadata import load_metadata, verify_weights_file


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, help="folder with model_weights.pth + model_metadata.json")
    p.add_argument("--manifest", required=True, help="samples.csv")
    p.add_argument("--images-root", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--evaluations-dir", help="default: <model-dir>/../evaluations")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args(argv)
    _common.setup_logging()
    try:
        from textile_defect_detection.model import load_torch_predictor

        model_dir = Path(a.model_dir)
        metadata = load_metadata(model_dir / "model_metadata.json")
        weights = model_dir / "model_weights.pth"
        verify_weights_file(weights, metadata)
        predictor = load_torch_predictor(weights, metadata, device=a.device)
        manifest = Path(a.manifest)
        info_path = manifest.parent / "dataset_manifest.json"
        info = load_dataset_info(info_path).to_dict() if info_path.is_file() else {}
        report = run_evaluation(
            predictor, metadata, read_samples_csv(manifest), a.images_root, split=a.split,
            evaluations_dir=a.evaluations_dir or model_dir.parent / "evaluations",
            manifest_path=manifest, batch_size=a.batch_size, dataset_info=info,
            weights_path=weights)
    except TextileDefectError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1
    m = report["metrics"]
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"Split '{a.split}': {report['sample_count']} samples, accuracy {m['accuracy']:.4f}, "
          f"macro F1 {m['macro_f1'] if m['macro_f1'] is None else round(m['macro_f1'], 4)}")
    print(f"Report written to {report['output_directory']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
