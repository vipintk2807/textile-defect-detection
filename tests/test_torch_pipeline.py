"""PyTorch-dependent tests (skipped automatically when PyTorch/torchvision are not installed).

They train on the SYNTHETIC fixture only to prove the pipeline runs end to end. Nothing here
says anything about real-world accuracy.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from helpers import image_bytes, make_metadata, prepared_synthetic_dataset  # noqa: E402
from textile_defect_detection.config import (AugmentationConfig, DatasetSection, ModelSection,  # noqa: E402
                                             OutputSection, PreprocessingConfig, TrainingConfig,
                                             TrainSection)
from textile_defect_detection.errors import ModelArtifactError  # noqa: E402
from textile_defect_detection.model import (build_model, load_state_dict_safely,  # noqa: E402
                                            load_torch_predictor, resolve_device, save_weights,
                                            set_backbone_trainable, set_train_mode)
from textile_defect_detection.model_metadata import (ModelMetadata, load_metadata,  # noqa: E402
                                                     verify_weights_file)
from textile_defect_detection.provenance import sha256_file  # noqa: E402


def test_model_output_dimension_matches_class_count():
    model = build_model("resnet18", 5, pretrained=False)
    out = model(torch.zeros(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 5)


def test_frozen_backbone_is_not_updated_but_head_is():
    model = build_model("resnet18", 3, pretrained=False)
    set_backbone_trainable(model, False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    set_train_mode(model, freeze_backbone=True)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 64, 64)),
                                             torch.tensor([0, 1, 2, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    after = model.state_dict()
    assert not torch.equal(before["fc.weight"], after["fc.weight"])
    assert torch.equal(before["conv1.weight"], after["conv1.weight"])
    # frozen BatchNorm running statistics must not drift either
    assert torch.equal(before["bn1.running_mean"], after["bn1.running_mean"])


def test_cuda_request_without_cuda_is_an_error():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(ModelArtifactError, match="CUDA"):
        resolve_device("cuda")


def export_random_model(tmp_path, num_classes=3):
    """Random-weight model (software test only - NOT a trained model)."""
    model = build_model("resnet18", num_classes, pretrained=False)
    weights = save_weights(model, tmp_path / "w.pth")
    meta = ModelMetadata.from_dict({**make_metadata().to_dict(),
                                    "weights_sha256": sha256_file(weights)})
    return weights, meta


def test_safe_checkpoint_round_trip_and_predictor_output(tmp_path):
    weights, meta = export_random_model(tmp_path)
    predictor = load_torch_predictor(weights, meta)
    batch = np.random.default_rng(0).normal(size=(2, 3, 32, 32)).astype(np.float32)
    first, second = predictor(batch), predictor(batch)
    assert first.shape == (2, 3) and np.all(np.isfinite(first))
    assert np.allclose(first, second, atol=1e-6)  # eval mode, deterministic


def test_incompatible_and_missing_and_malicious_checkpoints_are_rejected(tmp_path):
    weights, meta = export_random_model(tmp_path, num_classes=3)
    four_class = ModelMetadata.from_dict({**meta.to_dict(), "class_names": list("abcd"),
                                          "num_classes": 4})
    with pytest.raises(ModelArtifactError, match="incompatible"):
        load_torch_predictor(weights, four_class)
    with pytest.raises(ModelArtifactError, match="not found"):
        load_state_dict_safely(tmp_path / "missing.pth")
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("code executed during load",))

    evil = tmp_path / "evil.pth"
    evil.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ModelArtifactError, match="safely"):
        load_state_dict_safely(evil)


def test_tampered_weights_fail_checksum(tmp_path):
    weights, meta = export_random_model(tmp_path)
    verify_weights_file(weights, meta)
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(ModelArtifactError, match="checksum"):
        verify_weights_file(weights, meta)


def smoke_config(tmp_path, root, epochs=2):
    return TrainingConfig(
        dataset=DatasetSection(tmp_path / "proc" / "samples.csv", root,
                               tmp_path / "proc" / "class_mapping.json",
                               tmp_path / "proc" / "dataset_manifest.json"),
        model=ModelSection(pretrained=False, freeze_backbone=False),
        preprocessing=PreprocessingConfig(resize_size=40, crop_size=32, resize_mode="squash"),
        augmentation=AugmentationConfig(),
        training=TrainSection(seed=3, batch_size=8, max_epochs=epochs, learning_rate=1e-3,
                              backbone_learning_rate=1e-3, early_stopping_patience=5,
                              device="cpu"),
        output=OutputSection(tmp_path / "experiments", "smoke"))


def test_training_smoke_run_then_clean_load_inference_and_evaluation(tmp_path):
    from textile_defect_detection.dataset import read_samples_csv
    from textile_defect_detection.evaluation import run_evaluation
    from textile_defect_detection.inference import InferenceEngine
    from textile_defect_detection.config import InputConfig
    from textile_defect_detection.training import run_training

    root, samples, mapping = prepared_synthetic_dataset(tmp_path)
    result = run_training(smoke_config(tmp_path, root))
    assert result.epochs_completed == 2 and result.best_epoch in (1, 2)
    exp = result.experiment_dir
    for name in ("training_config.yaml", "class_mapping.json", "samples_manifest.csv",
                 "environment.json", "training_log.csv", "run_summary.json",
                 "best_weights.pth", "last_checkpoint.pth"):
        assert (exp / name).is_file(), name
    assert '"test_split_evaluated": false' in (exp / "run_summary.json").read_text()

    meta = load_metadata(result.model_dir / "model_metadata.json")
    assert list(meta.class_names) == mapping.class_names  # class order preserved
    verify_weights_file(result.model_dir / "model_weights.pth", meta)
    assert meta.trained_on_synthetic_data and meta.training["experiment_id"] == exp.name

    predictor = load_torch_predictor(result.model_dir / "model_weights.pth", meta)
    engine = InferenceEngine(meta, predictor, input_config=InputConfig(), threshold=0.0)
    res = engine.inspect_bytes(image_bytes(size=(80, 80)), "probe.png")
    assert res.prediction.predicted_class in meta.class_names

    report = run_evaluation(predictor, meta, read_samples_csv(tmp_path / "proc" / "samples.csv"),
                            root, split="test", evaluations_dir=exp / "evaluations",
                            manifest_path=tmp_path / "proc" / "samples.csv")
    assert report["sample_count"] == 12 and 0.0 <= report["metrics"]["accuracy"] <= 1.0


def test_training_validates_inputs_before_creating_any_output(tmp_path):
    from textile_defect_detection.errors import DatasetError
    from textile_defect_detection.training import run_training
    root, samples, _ = prepared_synthetic_dataset(tmp_path)
    cfg = smoke_config(tmp_path, root, epochs=1)
    first = run_training(cfg)
    assert first.experiment_dir.is_dir()
    from dataclasses import replace
    bad = replace(cfg, dataset=replace(cfg.dataset, manifest_path=tmp_path / "missing.csv"))
    before = sorted(p.name for p in (tmp_path / "experiments").iterdir())
    with pytest.raises(DatasetError):
        run_training(bad)
    assert sorted(p.name for p in (tmp_path / "experiments").iterdir()) == before


def test_resume_continues_from_last_checkpoint(tmp_path):
    from dataclasses import replace
    from textile_defect_detection.training import run_training
    root, _, _ = prepared_synthetic_dataset(tmp_path)
    cfg = smoke_config(tmp_path, root, epochs=1)
    first = run_training(cfg)
    longer = replace(cfg, training=replace(cfg.training, max_epochs=2))
    resumed = run_training(longer, resume_dir=first.experiment_dir)
    assert resumed.epochs_completed == 2 and len(resumed.history) == 2
