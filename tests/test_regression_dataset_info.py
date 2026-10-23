"""Regression: the file written by prepare_dataset.py must be loadable by training/evaluation."""

import subprocess
import sys
from pathlib import Path

import pytest

from textile_defect_detection.dataset import load_dataset_info
from textile_defect_detection.errors import DatasetError

ROOT = Path(__file__).resolve().parent.parent


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=300)


def test_dataset_manifest_written_by_prepare_script_can_be_loaded(tmp_path):
    syn, proc = tmp_path / "syn", tmp_path / "proc"
    assert run("make_synthetic_fixture.py", "--out", str(syn), "--images-per-class", "12").returncode == 0
    assert run("prepare_dataset.py", "--source-root", str(syn), "--output-dir", str(proc),
               "--group-regex", r"_(g\d+)_").returncode == 0
    info = load_dataset_info(proc / "dataset_manifest.json")
    assert info.synthetic is True and info.dataset_id == "synthetic-demo"


def test_genuinely_unknown_fields_are_still_rejected(tmp_path):
    path = tmp_path / "info.json"
    path.write_text('{"dataset_id": "x", "licence": "typo for license"}')
    with pytest.raises(DatasetError, match="licence"):
        load_dataset_info(path)
