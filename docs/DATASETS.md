# Dataset notes

**Nothing here has been verified by this build.** The sandbox that produced this project had no internet access, so no dataset was downloaded, inspected, or licence-checked. The notes below are *candidates and cautions*; items marked "recalled" come from general knowledge and must be confirmed on the official pages before you rely on them.

## Required checks before using any dataset

1. Find the original publisher's page and read the licence/terms (training use, modification, redistribution, commercial use, attribution).
2. Confirm the labels are **image-level classes** (this baseline is a single-label image classifier) — not only masks/boxes.
3. Count images per class and per specimen; check that every class can fill train/val/test.
4. Record everything in `dataset_info.json` (template in `data/README.md`).
5. If redistribution is unclear, do not publish the images; publish instructions only.

## Candidates named in the project specification

### AITEX Textile Fabric Image Database — first candidate to investigate
* Source: <https://www.aitex.es/afid/> (URL from the specification; availability and terms not checked).
* Recalled, unverified: a few hundred high-resolution fabric images (very wide strips, about 4096×256 px) of several fabric types, with defect-free images and defect images accompanied by defect masks.
* Consequences to check: (a) with only about a hundred defective images spread over many defect types, per-class counts may be too small for a meaningful validation/test split — the split tool will warn instead of padding classes; (b) very wide strips are mostly discarded by the default centre crop — the validator warns, `resize_mode: squash` is available, and **tiling into square patches (not implemented)** would likely be the better next step; (c) if the data only provide masks, image-level labels must be derived by a documented rule — do that explicitly and record it in `dataset_info.json`, never silently.

### MVTec Anomaly Detection (MVTec AD) — only as a separate anomaly-detection experiment
* Source: <https://www.mvtec.com/company/research/datasets/mvtec-ad> (not checked).
* Recalled, unverified: licensed for **non-commercial** use (CC BY-NC-SA 4.0) — confirm, because that may conflict with commercial use of derived models.
* It is an industrial-object/texture benchmark, not a garment-defect dataset, and its task (anomaly detection) differs from this repo's classification baseline. Do not merge it with other datasets.

### Others
Use original academic/institutional sources only; do not use mirrors when the original exists.

## What the tools do for you
* `prepare_dataset.py`: builds manifests, assigns **group-aware stratified** splits (70/15/15 default, seed recorded), merges byte-identical duplicates into one group, preserves official splits, never overwrites.
* `validate_dataset.py`: confirmed errors (missing/corrupt files, invalid labels, identical images across splits or with conflicting labels, split structure) vs heuristic warnings (perceptual-hash near-duplicates, extreme aspect ratios, no specimen metadata).
* Classes with fewer than three independent groups cannot appear in every split; this is reported, not hidden.

## Public-data limits
Even a clean public benchmark result would not show factory suitability: fabric types, defect mix, cameras, lighting and defect prevalence differ. Factory-specific images would need permission, a documented labelling procedure, and a separate untouched evaluation set.
