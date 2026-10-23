# data/

Local datasets live here and are **never committed** (see `.gitignore`). Nothing in this repository downloads data for you.

```
data/
  raw/<dataset_name>/<class>/<image>          class-folder layout  (this tool assigns the split)
  raw/<dataset_name>/{train,val,test}/<class>/<image>   official-split layout (--layout split-folders)
  processed/<dataset_version>/                manifests written by scripts/prepare_dataset.py
      samples.csv  class_mapping.json  dataset_manifest.json
  synthetic_demo/                             SYNTHETIC generated demo data (not real textiles)
```

Source images and annotations are only ever read. Manifests reference images by a path relative to the images root you pass to the scripts.

## dataset_info.json template (copy to `raw/<dataset_name>/dataset_info.json`)

Fill in every field from the dataset's **official** page. Leave `"unknown"` rather than guessing.

```json
{
  "dataset_id": "my-dataset-v1",
  "dataset_name": "Official dataset name",
  "source_url": "https://...",
  "publisher": "Publisher / institution",
  "version": "published version or download date",
  "download_date": "YYYY-MM-DD",
  "license": "exact licence identifier or terms",
  "citation": "required citation",
  "usage_restrictions": "redistribution / commercial-use limits",
  "annotation_format": "image-level class label (folder name)",
  "synthetic": false,
  "preparation_steps": [],
  "notes": ""
}
```

## Manifest columns

`sample_id, relative_image_path, split, class_label, source_dataset, group_id, source_annotation_reference`.
`group_id` identifies a specimen/session: all images sharing a group id stay in the same split. Provide it with `--group-regex` (first capture group of the file-name stem) whenever the file names encode the specimen.
