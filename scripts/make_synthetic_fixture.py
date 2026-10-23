"""Generate a small SYNTHETIC (not real textile) dataset for smoke tests and demos."""

from __future__ import annotations

import argparse
import sys

import _common  # noqa: F401  (sets up sys.path)
from textile_defect_detection.synthetic import generate_synthetic_dataset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic_demo", help="output folder (must be new/empty)")
    p.add_argument("--images-per-class", type=int, default=24)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    try:
        out = generate_synthetic_dataset(a.out, images_per_class=a.images_per_class,
                                         image_size=a.image_size, seed=a.seed)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote SYNTHETIC dataset to {out} (not real textile images; software testing only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
