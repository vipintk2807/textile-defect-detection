"""Copy a trained model into models/ and verify it loads in a fresh process-style check."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _common
import numpy as np
from textile_defect_detection.errors import TextileDefectError
from textile_defect_detection.model_metadata import load_metadata, verify_weights_file


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, help="e.g. outputs/experiments/<id>/model")
    p.add_argument("--dest", default="models", help="destination folder (default: models)")
    p.add_argument("--evaluation-dir", help="optionally copy this evaluation report folder too")
    a = p.parse_args(argv)
    _common.setup_logging()
    try:
        from textile_defect_detection.model import load_torch_predictor

        src = Path(a.model_dir)
        metadata = load_metadata(src / "model_metadata.json")
        verify_weights_file(src / "model_weights.pth", metadata)
        dest = Path(a.dest)
        targets = [dest / "model_weights.pth", dest / "model_metadata.json"]
        existing = [t.name for t in targets if t.exists()]
        if existing:
            print(f"ERROR: {', '.join(existing)} already exist in '{dest}'. Move or rename them "
                  "first (nothing is overwritten).", file=sys.stderr)
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "model_weights.pth", targets[0])
        shutil.copyfile(src / "model_metadata.json", targets[1])
        if a.evaluation_dir:
            shutil.copytree(a.evaluation_dir, dest / "evaluation_report", dirs_exist_ok=False)
        predictor = load_torch_predictor(targets[0], load_metadata(targets[1]))
        size = metadata.preprocessing.crop_size
        out = predictor(np.zeros((1, 3, size, size), dtype=np.float32))
        if out.shape != (1, metadata.num_classes) or not np.all(np.isfinite(out)):
            raise TextileDefectError("Exported model produced invalid output on a test input.")
    except TextileDefectError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1
    print(f"Exported model '{metadata.model_id}' to {dest} and verified it loads "
          f"({metadata.num_classes} classes).")
    if metadata.trained_on_synthetic_data:
        print("NOTE: this model was trained on SYNTHETIC data (software demo only).")
    print("Start the app with: python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
