"""Train the ResNet-18 baseline (CPU for small runs; use the Colab notebook for real training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common
from textile_defect_detection.config import load_training_config
from textile_defect_detection.errors import TextileDefectError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--resume", metavar="EXPERIMENT_DIR",
                   help="continue an interrupted run (uses that folder's training_config.yaml)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="override training.device")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    _common.setup_logging(a.verbose)
    try:
        from textile_defect_detection.training import run_training

        if a.resume:
            cfg = load_training_config(Path(a.resume) / "training_config.yaml")
        else:
            cfg = load_training_config(a.config)
        if a.device:
            from dataclasses import replace
            cfg = replace(cfg, training=replace(cfg.training, device=a.device))
        result = run_training(cfg, resume_dir=Path(a.resume) if a.resume else None)
    except TextileDefectError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1
    print(f"Training finished: {result.stopping_reason} after {result.epochs_completed} epoch(s); "
          f"best epoch {result.best_epoch} ({result.selection_metric} = {result.best_score:.4f}, "
          f"validation only).")
    print(f"Experiment folder : {result.experiment_dir}")
    print(f"Exportable model  : {result.model_dir}")
    print("The test split has NOT been evaluated. When all decisions are final run:")
    print(f"  python scripts/evaluate.py --model-dir \"{result.model_dir}\" "
          "--manifest <samples.csv> --images-root <images> --split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
