"""Static guarantees: app compiles, no hard-coded predictions, no secrets, no network calls."""

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = list((ROOT / "src").rglob("*.py")) + [ROOT / "app.py"] + list((ROOT / "scripts").glob("*.py"))


def test_all_python_files_compile():
    for path in SOURCES + list((ROOT / "tests").glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_app_delegates_to_workflow_and_has_no_fake_predictions():
    text = (ROOT / "app.py").read_text()
    assert "run_analysis" in text and "load_model_state" in text
    assert not re.search(r"^\s*(import|from)\s+(random|torch)\b", text, re.M)
    lowered = text.lower()
    for forbidden in ("certified", "100% accurate", "defect confirmed", "fabric certified"):
        assert forbidden not in lowered.replace("not a confirmed", "")


def test_no_network_client_libraries_are_used_by_the_application():
    for path in SOURCES:
        text = path.read_text()
        assert not re.search(r"^\s*(import|from)\s+(requests|urllib\.request|http\.client|socket)\b",
                             text, re.M), path.name


def test_no_obvious_secrets_in_repository_text_files():
    patterns = [r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
                r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----", r"AKIA[0-9A-Z]{16}"]
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".ipynb"}:
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            text = path.read_text(errors="ignore")
            for pat in patterns:
                assert not re.search(pat, text), f"possible secret in {path.name}"


def test_gitignore_excludes_data_models_outputs_and_environments():
    text = (ROOT / ".gitignore").read_text()
    for needle in (".venv/", "__pycache__/", "data/raw/", "*.pth", "outputs/*", "*.db"):
        assert needle in text
