"""Training pipeline: manifest dataset, transfer-learning loop, checkpoints, run records.

Requires PyTorch. The loop follows model_training_and_evaluation.md: train on the ``train`` split,
select the checkpoint on the ``val`` split only, apply early stopping, then write the final
artifact (weights + metadata). The ``test`` split is never read here; evaluate it afterwards
with ``scripts/evaluate.py`` once all model-selection decisions are finished.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .config import TrainingConfig
from .dataset import (DatasetInfo, Sample, build_class_mapping, load_class_mapping,
                      load_dataset_info, read_samples_csv, resolve_sample_path,
                      save_class_mapping, count_by_split_and_class)
from .errors import DatasetError, TrainingError
from .image_processing import augment_image, load_image_file, preprocess_image
from .metrics import compute_classification_metrics
from .model import (build_model, load_state_dict_safely, require_torch, resolve_device,
                    save_weights, set_backbone_trainable, set_train_mode)
from .model_metadata import ModelMetadata, save_metadata
from .provenance import environment_info, git_revision, sha256_file, utc_now_iso
from .splits import check_split_integrity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data
class ManifestImageDataset:
    """Map-style dataset over manifest rows (works with ``torch.utils.data.DataLoader``).

    Augmentation is applied only when ``augmentation`` is given (training split) and uses a
    per-item RNG seeded from (seed, epoch, index), so runs are repeatable. Source files are only
    read, never written.
    """

    def __init__(self, samples: list[Sample], images_root: Path, class_to_index: dict[str, int],
                 preprocessing, augmentation=None, seed: int = 0) -> None:
        self.samples = samples
        self.images_root = Path(images_root)
        self.class_to_index = class_to_index
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = require_torch()
        sample = self.samples[index]
        path = resolve_sample_path(self.images_root, sample.relative_image_path)
        loaded = load_image_file(path)
        image = loaded.image
        if self.augmentation is not None and self.augmentation.enabled:
            rng = random.Random(f"{self.seed}-{self.epoch}-{index}")
            image = augment_image(image, self.augmentation, rng)
        array = preprocess_image(image, self.preprocessing)
        return torch.from_numpy(array), self.class_to_index[sample.class_label]


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_counts: dict[str, int], class_names: list[str]) -> list[float]:
    """'Balanced' weights: total / (n_classes * count). Classes must all have training samples."""
    total = sum(train_counts.get(n, 0) for n in class_names)
    return [total / (len(class_names) * train_counts[n]) for n in class_names]


# --------------------------------------------------------------------------- epochs
def train_one_epoch(model, loader, optimizer, criterion, device, freeze_backbone: bool) -> float:
    torch = require_torch()
    set_train_mode(model, freeze_backbone)
    running, seen = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        if not torch.isfinite(loss):
            raise TrainingError("The training loss became non-finite (NaN/inf). Lower the "
                                "learning rate or check the data.")
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * targets.size(0)
        seen += targets.size(0)
    if seen == 0:
        raise TrainingError("The training split produced no batches.")
    return running / seen


def evaluate_loader(model, loader, device) -> tuple[float, list[int], list[int]]:
    """Validation pass: eval mode, no gradients, no augmentation. Returns (loss, y_true, y_pred)."""
    torch = require_torch()
    model.eval()
    total_loss, seen = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += float(torch.nn.functional.cross_entropy(
                outputs, targets, reduction="sum").item())
            seen += targets.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    if seen == 0:
        raise TrainingError("The validation split produced no batches.")
    return total_loss / seen, y_true, y_pred


# --------------------------------------------------------------------------- run
@dataclass
class RunResult:
    experiment_dir: Path
    model_dir: Path
    best_epoch: int
    best_score: float
    selection_metric: str
    epochs_completed: int
    stopping_reason: str
    device: str
    history: list[dict] = field(default_factory=list)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _load_inputs(cfg: TrainingConfig):
    samples = read_samples_csv(cfg.dataset.manifest_path)
    if cfg.dataset.class_mapping_path and cfg.dataset.class_mapping_path.is_file():
        mapping = load_class_mapping(cfg.dataset.class_mapping_path)
    else:
        mapping = build_class_mapping([s.class_label for s in samples])
        logger.warning("No class_mapping.json found; derived class order alphabetically.")
    if cfg.dataset.dataset_info_path and cfg.dataset.dataset_info_path.is_file():
        info = load_dataset_info(cfg.dataset.dataset_info_path)
    else:
        info = DatasetInfo()
        logger.warning("No dataset_manifest.json found; dataset provenance is 'unknown'.")

    unknown = sorted({s.class_label for s in samples} - set(mapping.class_names))
    if unknown:
        raise DatasetError(f"The manifest contains labels not in the class mapping: "
                           f"{', '.join(unknown)}.")
    errors = [i for i in check_split_integrity(samples) if i.severity == "error"]
    if errors:
        raise DatasetError("Split integrity errors must be fixed before training: "
                           + "; ".join(i.message for i in errors[:3]))
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    if not train or not val:
        raise DatasetError("Training needs non-empty 'train' and 'val' splits.")
    train_counts = Counter(s.class_label for s in train)
    missing = [n for n in mapping.class_names if train_counts.get(n, 0) == 0]
    if missing:
        raise DatasetError("Classes without any training samples cannot be learned: "
                           f"{', '.join(missing)}. Remove them from the class mapping or add data.")
    return samples, mapping, info, train, val, train_counts


def run_training(cfg: TrainingConfig, *, resume_dir: Path | None = None) -> RunResult:
    """Run (or resume) one experiment. Writes everything under a new experiment directory."""
    torch = require_torch()
    from torch.utils.data import DataLoader  # noqa: PLC0415

    samples, mapping, info, train_samples, val_samples, train_counts = _load_inputs(cfg)
    class_names = list(mapping.class_names)
    class_to_index = mapping.class_to_index
    t = cfg.training

    if resume_dir is not None:
        exp_dir = Path(resume_dir)
        if not (exp_dir / "last_checkpoint.pth").is_file():
            raise TrainingError("Cannot resume: 'last_checkpoint.pth' was not found in that "
                                "experiment folder.")
        experiment_id = exp_dir.name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        experiment_id = f"{stamp}-{cfg.output.experiment_name}"
        exp_dir = cfg.output.directory / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
        (exp_dir / "training_config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        save_class_mapping(mapping, exp_dir / "class_mapping.json")
        shutil.copyfile(cfg.dataset.manifest_path, exp_dir / "samples_manifest.csv")
        _write_json(exp_dir / "dataset_info.json", info.to_dict())
        _write_json(exp_dir / "environment.json", environment_info())

    seed_everything(t.seed)
    device = resolve_device(t.device)
    logger.info("Experiment %s on device %s", experiment_id, device)

    train_ds = ManifestImageDataset(train_samples, cfg.dataset.images_root, class_to_index,
                                    cfg.preprocessing, cfg.augmentation, seed=t.seed)
    val_ds = ManifestImageDataset(val_samples, cfg.dataset.images_root, class_to_index,
                                  cfg.preprocessing, None, seed=t.seed)
    generator = torch.Generator()
    generator.manual_seed(t.seed)
    train_loader = DataLoader(train_ds, batch_size=t.batch_size, shuffle=True,
                              num_workers=t.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=t.batch_size, shuffle=False,
                            num_workers=t.num_workers)

    model = build_model(cfg.model.architecture, len(class_names), pretrained=cfg.model.pretrained,
                        pretrained_weights=cfg.model.pretrained_weights)
    freeze = cfg.model.freeze_backbone
    if freeze:
        set_backbone_trainable(model, False)
    model.to(device)

    head_params = list(model.fc.parameters())
    if freeze:
        param_groups = [{"params": head_params, "lr": t.learning_rate}]
    else:
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        param_groups = [{"params": backbone, "lr": t.backbone_learning_rate},
                        {"params": head_params, "lr": t.learning_rate}]
    optimizer = torch.optim.Adam(param_groups, weight_decay=t.weight_decay)

    weight_tensor = None
    if t.class_weighting == "balanced":
        weight_tensor = torch.tensor(compute_class_weights(train_counts, class_names),
                                     dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    history: list[dict] = []
    best_score, best_epoch, without_improvement, start_epoch = -float("inf"), 0, 0, 1
    if resume_dir is not None:
        ckpt = load_state_dict_safely(exp_dir / "last_checkpoint.pth")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = list(ckpt["history"])
        best_score, best_epoch = float(ckpt["best_score"]), int(ckpt["best_epoch"])
        without_improvement = int(ckpt["epochs_without_improvement"])
        start_epoch = int(ckpt["epoch"]) + 1

    started = time.perf_counter()
    stopping_reason = "max_epochs_reached"
    epochs_completed = start_epoch - 1
    try:
        for epoch in range(start_epoch, t.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_ds.epoch = epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, freeze)
            val_loss, y_true, y_pred = evaluate_loader(model, val_loader, device)
            metrics = compute_classification_metrics(y_true, y_pred, class_names)
            score = (metrics.macro_f1 if t.selection_metric == "val_macro_f1" else -val_loss)
            if score is None:
                raise TrainingError("The selection metric is undefined on the validation split.")
            improved = score > best_score + 1e-9
            if improved:
                best_score, best_epoch, without_improvement = float(score), epoch, 0
                save_weights(model, exp_dir / "best_weights.pth", overwrite=True)
            else:
                without_improvement += 1
            history.append({
                "epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
                "val_accuracy": round(metrics.accuracy, 6),
                "val_macro_f1": None if metrics.macro_f1 is None else round(metrics.macro_f1, 6),
                "learning_rate": optimizer.param_groups[-1]["lr"],
                "epoch_seconds": round(time.perf_counter() - epoch_start, 2),
                "checkpoint_updated": improved,
            })
            epochs_completed = epoch
            torch.save({
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score,
                "best_epoch": best_epoch, "epochs_without_improvement": without_improvement,
                "history": history}, exp_dir / "last_checkpoint.pth")
            _write_history(exp_dir / "training_log.csv", history)
            logger.info("epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f val_macro_f1=%s%s",
                        epoch, train_loss, val_loss, metrics.accuracy,
                        history[-1]["val_macro_f1"], " *" if improved else "")
            if without_improvement >= t.early_stopping_patience:
                stopping_reason = "early_stopping"
                break
    except BaseException as exc:
        _write_json(exp_dir / "run_summary.json", {
            "experiment_id": experiment_id, "status": "failed",
            "last_completed_epoch": epochs_completed, "error": repr(exc),
            "finished_at": utc_now_iso()})
        raise

    if best_epoch == 0 or not (exp_dir / "best_weights.pth").is_file():
        raise TrainingError("No checkpoint was selected (no epoch completed).")

    # Final artifact = the best validation-selected weights (reloaded, not the last epoch).
    model_dir = exp_dir / "model"
    model_dir.mkdir(exist_ok=True)
    final_weights = model_dir / "model_weights.pth"
    shutil.copyfile(exp_dir / "best_weights.pth", final_weights)
    import torchvision  # noqa: PLC0415

    metadata = ModelMetadata(
        model_id=f"{cfg.model.architecture}-{experiment_id}",
        class_names=tuple(class_names),
        preprocessing=cfg.preprocessing,
        weights_sha256=sha256_file(final_weights),
        architecture=cfg.model.architecture,
        created_at=utc_now_iso(),
        framework={"torch": torch.__version__, "torchvision": torchvision.__version__},
        training={
            "experiment_id": experiment_id, "seed": t.seed, "device": str(device),
            "pretrained_init": cfg.model.pretrained,
            "pretrained_weights": cfg.model.pretrained_weights if cfg.model.pretrained else None,
            "freeze_backbone": freeze, "class_weighting": t.class_weighting,
            "selection_metric": t.selection_metric, "best_epoch": best_epoch,
            "best_validation_score": best_score, "epochs_completed": epochs_completed,
            "stopping_reason": stopping_reason,
            "config_sha256": sha256_file(exp_dir / "training_config.yaml"),
            "git_revision": git_revision(),
        },
        dataset={
            **info.to_dict(),
            "manifest_sha256": sha256_file(exp_dir / "samples_manifest.csv"),
            "counts_by_split_and_class": count_by_split_and_class(samples),
        },
        evaluation_report=None,
    )
    save_metadata(metadata, model_dir / "model_metadata.json", overwrite=True)
    _write_json(exp_dir / "run_summary.json", {
        "experiment_id": experiment_id, "status": "completed", "device": str(device),
        "epochs_completed": epochs_completed, "stopping_reason": stopping_reason,
        "best_epoch": best_epoch, "selection_metric": t.selection_metric,
        "best_validation_score": best_score,
        "training_seconds": round(time.perf_counter() - started, 1),
        "test_split_evaluated": False,
        "note": "Test split NOT evaluated by training. Run scripts/evaluate.py once decisions "
                "are final.", "finished_at": utc_now_iso()})
    return RunResult(exp_dir, model_dir, best_epoch, best_score, t.selection_metric,
                     epochs_completed, stopping_reason, str(device), history)
