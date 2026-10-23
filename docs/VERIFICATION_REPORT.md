# Verification report (build environment)

This report lists only what was **actually executed** while building the project, and what was **not**. It is not evidence of model accuracy or industrial suitability. Please add your own results (Windows, real pytest, PyTorch, Streamlit) before a public release.

## Environment used for the checks
Linux sandbox (kernel 6.18), Python 3.12.3, NumPy 2.4.4, Pillow 12.1.1, scikit-learn 1.8.0, PyYAML 6.0.3. **No internet access.** PyTorch, torchvision, Streamlit, pytest and ruff were **not installed** and could not be installed. Date of build: 2026-10-19.

## Test execution
Real `pytest` was unavailable. The test files are written for pytest, and were executed with a minimal stand-in runner that implements only what they use (`tmp_path`, `parametrize`, `raises`, `importorskip`). The stand-in is not shipped.

| Item | Result |
|---|---|
| Command (stand-in runner, all `tests/test_*.py`) | **177 passed, 0 failed, 1 module skipped** |
| Skipped | `tests/test_torch_pipeline.py` (9 tests): needs PyTorch |
| Covered | config validation, image validation/preprocessing/augmentation, softmax/thresholds/prediction contract, metadata + checksum, inference error handling (with a stand-in predictor), workflow logic, SQLite history, CSV/JSON export, dataset validation + leakage detection, group-aware splits, metrics (cross-checked with scikit-learn), evaluation reports, theme contrast, synthetic generator, command-line scripts in subprocesses, static checks (compile, no network imports, no secrets, `.gitignore`) |
| Test-double caveat | Inference/evaluation tests use stand-in predictors, so they verify software wiring, not any model's behaviour. |

### Sanity check that the tests detect bugs
Seven deliberate defects were injected one at a time into a copy of the code: inverted class ranking, broken F1 formula, test-split assignment leaking into train, copy-less augmentation, CSV formula-injection guard removed, confidence threshold ignored, checksum verification disabled. **Six were caught by failing tests.** The augmentation one was not caught because it is an equivalent mutant (every Pillow operation used already returns a new image).

### Defects found and fixed during the build
* Synthetic generator reused group ids across classes and produced byte-identical images; fixed (found by running the pipeline).
* `app.py`: duplicate Streamlit button keys and a disabled Analyze button after a failed attempt; fixed by code review (the app itself was not run).
* Four test mistakes (arithmetic, per-channel comparison, a degenerate dhash fixture, runner limitation) fixed; no source bug was behind them.

## Commands run end to end (torch-free)
`make_synthetic_fixture.py` → `prepare_dataset.py` → `validate_dataset.py` on a generated synthetic dataset: validation PASS (48/48 images, 0 errors, 0 warnings, split 24/12/12 with every class in every split). Failure paths exercised: missing file → exit code 1 and an `excluded_samples.csv` with the reason; re-running `prepare_dataset.py` into a non-empty folder → refused; unreadable manifest → exit code 2. Every script's `--help` works.

## Measured design checks
WCAG contrast of the specified palette: body text on page 11.37:1, on white 12.23:1; navy on page 12.21:1; white on teal button 4.83:1; teal on white 4.83:1; success 5.26, warning 5.24, error 6.57, info 7.01 (all on white); white on navy 13.13:1; **muted slate on white 4.61:1 but on the page background 4.28:1 (below 4.5:1)** — so it must not be used for essential text on the page background.

## NOT verified (you must run these)
| Item | Why not | How |
|---|---|---|
| `tests/test_torch_pipeline.py` (model dimensions, frozen backbone, safe loading incl. malicious pickle, synthetic training smoke run, resume, evaluation with a real model) | PyTorch not installable offline | `pip install -r requirements-dev.txt` then `python -m pytest -q` |
| Real `pytest` run of the whole suite on Windows | no pytest/Windows in sandbox | `python -m pytest -q` |
| Streamlit app startup and all UI states | Streamlit not installable | `docs/MANUAL_UI_CHECKLIST.md` |
| Training/evaluation on any real dataset; pretrained-weight download | no network, no dataset | README → "Using a real dataset" |
| Colab notebook execution | needs Colab (notebook JSON and code cells were only syntax-checked) | run it in Colab |
| Windows CMD commands in the README | sandbox is Linux; commands written but not run on Windows | follow README |
| `ruff` lint | not installed (an AST scan for unused imports was run and clean-ups applied) | `ruff check .` |
| Inference latency, memory, CPU/GPU comparison | no model/hardware | not measured; no numbers are claimed |
| Dataset licences/contents (AITEX, MVTec AD) | no network | docs/DATASETS.md |
| Exact dependency versions for torch/torchvision/streamlit | not installable | ranges in `requirements.txt`; freeze after your first green run |

## Claims deliberately NOT made
No model is included or claimed to be trained; no accuracy, F1, speed, or factory-readiness figure exists in this project. The synthetic dataset is generated and labelled `synthetic_*`; any model trained on it is flagged in its metadata and in the app.
