# models/

Place the **exported model** here (never committed to Git):

* `model_weights.pth` — tensors only (loaded with `torch.load(weights_only=True)`).
* `model_metadata.json` — class names in output order, preprocessing, SHA-256 of the weights, training/dataset provenance.

Create both with `python scripts\export_model.py --model-dir outputs\experiments\<EXP>\model` (it refuses to overwrite and checks the model loads). The app refuses to run if either file is missing, malformed, or their checksums disagree.

Weights trained from ImageNet-pretrained torchvision ResNet-18 inherit that checkpoint's terms; check them (and your dataset's licence) before redistributing any model.
