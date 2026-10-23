# How the specification documents were reconciled

The project was built from eight documents: `master_prompt`, `project_explanation`, `technical_specification`, `dataset_and_annotation_specification`, `model_training_and_evaluation`, `app_workflow`, `app_theme`, `testing_requirements` (supplied as `.md`; the master prompt names them `.docx`). No document was blocking; the following judgement calls were made.

| Topic | Conflict / gap | Decision |
|---|---|---|
| Inspection history & export | `project_explanation` and `technical_specification` require SQLite history and CSV/JSON export; `app_workflow` says no history/database "unless explicitly approved" and to avoid retaining uploads. | Implemented, but **opt-in per session** (checkbox, off by default), metadata and scores only, never images. The export/history code is covered by tests. Delete `inspection_store.py`/`export.py` and the expander if you do not want it. |
| Task type | Technical spec: classification vs detection depends on annotations; training spec: classification baseline. | Classification only (ResNet-18). No boxes/annotation rendering, so the "annotated image" parts of the explanation/spec (which apply only to detection models) are not implemented, as those documents allow. |
| Input formats | Tech spec lists JPEG/PNG/WebP; workflow says JPEG/PNG are "reasonable". | JPEG, PNG, WebP (configurable). |
| OpenCV | Listed in the technology table. | Not used; Pillow + NumPy cover everything a classification baseline needs, avoiding an unnecessary dependency. |
| Pydantic | "Where useful". | Not used; frozen dataclasses with explicit validation avoid a dependency and keep errors user-friendly. |
| Streamlit theme | Spec asks for contrast verification. | Measured with a WCAG calculator in the tests. Result: muted slate `#667788` is 4.28:1 on the page background (fails 4.5:1) but 4.61:1 on white. It is therefore used only on white surfaces (currently not used at all). |
| Preprocessing | Spec: use pretrained weights' documented preprocessing. | ImageNet recipe (resize 256 bilinear, centre-crop 224, mean/std) implemented in Pillow/NumPy; **not claimed bit-identical to torchvision**. Stored in model metadata and reused by training, evaluation and inference. |
| Test-set discipline | Final test evaluated "once". | `train.py` never reads the test split. `evaluate.py` counts earlier evaluations of the same split and writes a warning; non-test splits are labelled "not independent". |
| Checkpoint format | State-dict, trusted files. | Tensor-only `model_weights.pth` + JSON metadata with SHA-256; loaded with `weights_only=True` (a malicious pickle is rejected — tested when PyTorch is installed). |
| Default threshold 0.5 | Spec: placeholder only. | Kept as placeholder, documented; below-threshold = "review required". |
| Python / dependency versions | Spec: pin tested versions. | Not possible offline for torch/streamlit; compatible ranges given. Freeze after your first successful run. |

Out of scope (per documents): camera/video input, production-line integration, accept/reject decisions, localisation/segmentation, authentication, cloud hosting.
