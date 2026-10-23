"""Shared helpers for the command-line scripts (run them from the project root)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    import textile_defect_detection  # noqa: F401
except ImportError:  # allow running without 'pip install -e .'
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
