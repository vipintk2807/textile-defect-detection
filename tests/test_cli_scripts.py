"""Run the real command-line scripts in subprocesses (torch-free ones end to end)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=300)


def test_generate_prepare_validate_pipeline(tmp_path):
    syn, proc, rep = tmp_path / "syn", tmp_path / "proc", tmp_path / "rep"
    assert run("make_synthetic_fixture.py", "--out", str(syn), "--images-per-class", "12").returncode == 0
    prep = run("prepare_dataset.py", "--source-root", str(syn), "--output-dir", str(proc),
               "--group-regex", r"_(g\d+)_")
    assert prep.returncode == 0, prep.stderr
    for name in ("samples.csv", "class_mapping.json", "dataset_manifest.json"):
        assert (proc / name).is_file()
    val = run("validate_dataset.py", "--manifest", str(proc / "samples.csv"), "--images-root",
              str(syn), "--report-dir", str(rep))
    assert val.returncode == 0, val.stdout + val.stderr
    assert (rep / "validation_report.json").is_file() and "PASS" in val.stdout


def test_prepare_refuses_to_overwrite_and_validate_reports_errors(tmp_path):
    syn, proc = tmp_path / "syn", tmp_path / "proc"
    run("make_synthetic_fixture.py", "--out", str(syn), "--images-per-class", "12")
    assert run("prepare_dataset.py", "--source-root", str(syn), "--output-dir", str(proc)).returncode == 0
    again = run("prepare_dataset.py", "--source-root", str(syn), "--output-dir", str(proc))
    assert again.returncode == 1 and "refusing to overwrite" in again.stderr
    next(syn.rglob("*.png")).unlink()
    bad = run("validate_dataset.py", "--manifest", str(proc / "samples.csv"), "--images-root", str(syn))
    assert bad.returncode == 1 and "missing_file" in bad.stdout


def test_validate_with_unreadable_manifest_exits_2(tmp_path):
    res = run("validate_dataset.py", "--manifest", str(tmp_path / "none.csv"), "--images-root", str(tmp_path))
    assert res.returncode == 2 and "not found" in res.stderr


@pytest.mark.parametrize("script", ["train.py", "evaluate.py", "export_model.py", "prepare_dataset.py",
                                    "validate_dataset.py", "make_synthetic_fixture.py"])
def test_every_script_has_working_help(script):
    res = run(script, "--help")
    assert res.returncode == 0 and "usage" in res.stdout.lower()


def test_train_without_pytorch_or_dataset_fails_with_clear_message(tmp_path):
    res = run("train.py", "--config", str(ROOT / "configs" / "training.yaml"))
    assert res.returncode == 1 and res.stderr.startswith("ERROR:")
