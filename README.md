# Textile defect detection

An open-source **image-level classification baseline** for textile/garment images, with a reproducible data → training → evaluation → inference pipeline and a local Streamlit inspection app. It runs on a Windows laptop **without a GPU**; GPU-heavy training is intended for Google Colab.

> **Inspection-assistance prototype.** Predictions are model outputs, not confirmed inspection results. The classifier labels the *whole image*; it does not locate defects, measure them, or decide accept/reject. No accuracy, speed or industrial-readiness claim is made — see [Current status](#current-status).

Repository: <https://github.com/vipintk2807/textile-defect-detection> (owner `vipintk2807`)

---

## Current status

| Item | Status |
|---|---|
| Dataset tools (prepare / validate / group-aware split / duplicate + leakage checks) | Implemented and **executed** against a synthetic fixture |
| Preprocessing, augmentation, prediction contract, inference engine, metrics, evaluation reports, SQLite history, CSV/JSON export, app workflow logic | Implemented and **executed** (177 automated test cases pass, see [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)) |
| PyTorch model, training loop, checkpointing, safe weight loading, Colab notebook, Streamlit UI | Implemented, **not executed in the build environment** (no PyTorch/Streamlit could be installed there). Their tests exist and run on your machine. |
| A trained model | **None is included.** No real dataset was downloaded; no training or evaluation on real textile data has been done. |
| Real dataset | Not selected or verified. See [docs/DATASETS.md](docs/DATASETS.md). |

Until you train a model, the app shows a clear "model unavailable" message. It never substitutes random weights or fake predictions.

## What is in the box

```
app.py                       Streamlit app (presentation only)
src/textile_defect_detection/
  config.py                  validated YAML configuration
  image_processing.py        validation, decoding, preprocessing, augmentation (Pillow/NumPy)
  model_metadata.py          class mapping + preprocessing + checksum travelling with the weights
  predictions.py             softmax, threshold status, prediction contract
  inference.py               InferenceEngine (validated image -> structured result)
  model.py / training.py     PyTorch ResNet-18, safe weight I/O, training loop  (need PyTorch)
  dataset.py / splits.py / dataset_validation.py   manifests, group-aware splits, integrity checks
  metrics.py / evaluation.py accuracy, per-class P/R/F1, macro F1, confusion matrix, reports
  inspection_store.py / export.py   local SQLite history, CSV/JSON export
  workflow.py                UI-independent app logic (fully tested)
  synthetic.py               generator for a clearly labelled SYNTHETIC demo dataset
scripts/                     prepare_dataset, validate_dataset, train, evaluate, export_model, make_synthetic_fixture
configs/                     inference.yaml, training.yaml, training_synthetic_smoke.yaml
notebooks/textile_defect_training.ipynb   Google Colab training notebook
tests/                       automated tests
docs/                        datasets, spec notes, verification report, GitHub publishing, UI checklist
```

## Windows setup (Command Prompt)

Requirements: Windows 10/11, **Python 3.10–3.12** from <https://www.python.org/downloads/> (tick "Add python.exe to PATH"), and Git if you want to publish. No GPU needed. Everything is free/open-source.

Open **Command Prompt** (CMD), then:

```bat
cd C:\path\to\textile-defect-detection
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
python -m pytest -q
```

* After activation your prompt starts with `(.venv)`. Re-run `.venv\Scripts\activate.bat` in every new CMD window.
* `torch` from PyPI on Windows is the CPU build. If installation of torch fails or is huge, use the official CPU wheels: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements-dev.txt`.
* Expected test result: all tests pass (the torch tests run only once PyTorch is installed). If something fails, see [Troubleshooting](#troubleshooting).

## Try the whole pipeline with synthetic data (≈ a few minutes on CPU)

This proves the software works end to end. The data are **generated, not real textiles**; the resulting model says nothing about real defects (the app shows a warning banner for such models).

```bat
python scripts\make_synthetic_fixture.py --out data\synthetic_demo --images-per-class 24
python scripts\prepare_dataset.py --source-root data\synthetic_demo --output-dir data\processed\synthetic_demo --group-regex "_(g\d+)_"
python scripts\validate_dataset.py --manifest data\processed\synthetic_demo\samples.csv --images-root data\synthetic_demo
python scripts\train.py --config configs\training_synthetic_smoke.yaml
```

`train.py` prints an experiment folder such as `outputs\experiments\20260101-120000-synthetic-smoke`. Then (replace `<EXP>`):

```bat
python scripts\evaluate.py --model-dir outputs\experiments\<EXP>\model --manifest data\processed\synthetic_demo\samples.csv --images-root data\synthetic_demo --split test
python scripts\export_model.py --model-dir outputs\experiments\<EXP>\model
python -m streamlit run app.py
```

Streamlit prints a local address (usually <http://localhost:8501>); open it, upload any JPG/PNG/WebP image, press **Analyze image**.

## Using a real dataset

1. **Choose and verify a dataset** — read [docs/DATASETS.md](docs/DATASETS.md) first (licence, labels, whether it supports image-level classification). Nothing in this repo has verified a dataset for you.
2. Put images in `data\raw\<name>\<class>\image.png` (or `train/val/test` sub-folders for an official split) and create `data\raw\<name>\dataset_info.json` from the template in [data/README.md](data/README.md) — fill in the real source URL, licence and citation.
3. Prepare, validate, then fix errors:
   ```bat
   python scripts\prepare_dataset.py --source-root data\raw\<name> --output-dir data\processed\<name>-v1 --group-regex "<regex for specimen id>"
   python scripts\validate_dataset.py --manifest data\processed\<name>-v1\samples.csv --images-root data\raw\<name> --report-dir outputs\validation\<name>-v1
   ```
   Use `--layout split-folders` to keep an official train/val/test split. `validate_dataset.py` exits with code 1 on confirmed errors (missing/corrupt files, invalid labels, identical images across splits, conflicting labels) and prints heuristic warnings (near-duplicates, very wide images, missing specimen ids) separately.
4. Copy `configs\training.yaml`, set `dataset.manifest_path` / `dataset.images_root`, and train on **Colab** (GPU) using the notebook, or locally on CPU for small experiments:
   ```bat
   python scripts\train.py --config configs\training.yaml
   ```
   The baseline is ResNet-18 with ImageNet-pretrained weights (downloaded by torchvision on first use; internet needed once), a frozen backbone + new classification head, Adam, early stopping on **validation** macro-F1.
5. Choose settings using validation results only. When finished, evaluate the **test** split once:
   ```bat
   python scripts\evaluate.py --model-dir outputs\experiments\<EXP>\model --manifest <samples.csv> --images-root <images> --split test
   ```
   The report (JSON + Markdown + per-sample CSV) lists the model checksum, dataset manifest checksum, split, per-class precision/recall/F1/support, macro-F1, accuracy, confusion matrix and warnings, and counts earlier evaluations of the same split.
6. Export and run the app: `python scripts\export_model.py --model-dir outputs\experiments\<EXP>\model` then `python -m streamlit run app.py`.

### Google Colab

Open `notebooks\textile_defect_training.ipynb` in Colab (File → Upload notebook, or from GitHub once published), upload the project ZIP, and run the cells top to bottom. The notebook detects whether a GPU is present, validates data, trains, evaluates, and downloads a results ZIP (Colab's `/content` disk is temporary). Colab's free-tier limits change; check its current documentation.

## The app

* Upload → validate (real content decode, size/pixel limits) → preview → **Analyze image** → predicted class + ranked softmax scores (labelled "not a calibrated probability") + notice. **Clear image** resets everything; a new image never shows an old result.
* Below the confidence threshold (default 0.5, a software placeholder — choose yours from validation data) the result is shown as *review required*, never as "defect-free".
* Optional **Save results to local history** (off by default): stores metadata and scores — never the image — in `outputs\inspections.db`; filter and download CSV/JSON in the expander.
* Privacy: images are processed locally; the app makes no network requests and Streamlit usage statistics are disabled in `.streamlit\config.toml`. The app does not write uploaded images to disk; Streamlit itself holds uploads in server memory for the session (framework behaviour — not independently verified in this build).

## Configuration

`configs\inference.yaml` — model paths, input limits, threshold, storage. `configs\training.yaml` — dataset paths, preprocessing, augmentation, optimiser, early stopping. Unknown keys are rejected, so typos fail loudly. Relative paths are relative to the project root.

## Tests

```bat
python -m pytest -q
```

Unit tests cover configuration, image validation/preprocessing/augmentation, softmax and thresholds, metadata and checksum checks, inference error handling, dataset validation and leakage detection, splits, metrics (cross-checked against scikit-learn), evaluation reports, SQLite history (incl. injection attempts), exports (incl. CSV formula injection), theme contrast, and the command-line scripts. `tests\test_torch_pipeline.py` exercises model dimensions, frozen-backbone behaviour, safe checkpoint loading (including rejecting a malicious pickle), a synthetic training smoke run, evaluation and resume — it is skipped automatically if PyTorch is not installed. Synthetic fixtures verify software only, never accuracy.

Manual UI checks: [docs/MANUAL_UI_CHECKLIST.md](docs/MANUAL_UI_CHECKLIST.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'py' is not recognized` | Install Python from python.org with "Add to PATH", or use `python -m venv .venv`. |
| `.venv\Scripts\activate.bat` not found | You are not in the project folder, or the venv was not created. |
| `ModuleNotFoundError: textile_defect_detection` | Run `pip install -e .` inside the activated venv from the project root. |
| App says **Model unavailable** | No trained model in `models\`. Run the synthetic demo or train, then `scripts\export_model.py`, then click *Reload model*. |
| "checksum mismatch" | `model_weights.pth` and `model_metadata.json` come from different runs; copy both from the same `model\` folder. |
| "pretrained weights could not be loaded" | First training run needs internet to download ImageNet weights; or set `model.pretrained: false`. |
| `CUDA was requested but is not available` | Use `device: cpu` (inference) or `auto` (training). |
| Training is slow | Expected on CPU; use Colab and/or fewer epochs / smaller `batch_size`. |
| Streamlit port busy | `python -m streamlit run app.py --server.port 8502` |
| PowerShell blocks activation | Use CMD (`activate.bat`) or call `.venv\Scripts\python.exe` directly. Do not weaken system policy. |

## Known limitations

* No trained model or real-data results ship with this project.
* Classification only: no localisation/segmentation, no severity, no real-time/camera input, no accept/reject decisions.
* Closed-set classifier: an unfamiliar defect is forced into a known class. Softmax scores are not calibrated.
* Public-dataset performance does not establish factory performance (fabric, camera, lighting, defect mix differ).
* `center_crop` preprocessing discards most of very wide images (e.g. 4096×256 strips); `resize_mode: squash` is available, but tiling is **not** implemented. The validator warns about such images.
* Dependency versions are ranges, not tested pins (see `requirements.txt`).
* PyTorch/Streamlit code paths were written and reviewed but not run in the build environment.

## Contributing & releases

See [CONTRIBUTING.md](CONTRIBUTING.md). To publish to GitHub with the correct account on a multi-account Windows machine, follow [docs/PUBLISHING_WINDOWS.md](docs/PUBLISHING_WINDOWS.md).

## Licence

Source code: MIT (`LICENSE`) — please confirm this choice suits you before publishing. The licence does **not** cover any dataset or pretrained weights, which keep their own terms (ImageNet-pretrained torchvision weights and any dataset you use must be checked separately).
