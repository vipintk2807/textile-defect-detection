# Contributing

1. Set up the environment (README → Windows setup) and run `python -m pytest -q` before and after your change.
2. Keep presentation (`app.py`), logic (`workflow.py`, `inference.py`, …) and PyTorch code (`model.py`, `training.py`) separate; preprocessing exists in one place (`image_processing.py`) and is reused everywhere.
3. Never fabricate results: no hard-coded predictions, no placeholder metrics. Synthetic data must stay labelled `synthetic_*` and never be used for accuracy claims.
4. Do not tune against the test split. Changing preprocessing, class mapping, architecture or dataset invalidates earlier reported numbers — regenerate evaluation reports.
5. Do not commit datasets, model weights, databases, credentials or private images.
6. Add tests for new behaviour, including failure paths. Optional lint: `ruff check .` (not run in the original build).

## Preparing a release
Run the full tests, the synthetic smoke pipeline, the manual UI checklist, update `docs/VERIFICATION_REPORT.md` with *your* actual results, verify dataset/weights licences, then tag.
