"""Pytest suite for MAYA Phase 3 Sprint 2 training engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataloader import create_dataloader
from ai.datasets.dataset_config import DatasetConfig, SplitBudget, SplitName
from ai.datasets.pipeline import run_dataset_pipeline
from ai.engine.checkpoint import load_checkpoint, save_checkpoint
from ai.engine.experiment import ExperimentTracker
from ai.engine.history import TrainingHistory
from ai.engine.trainer import Trainer
from ai.models.model_config import ModelConfig
from ai.models.model_factory import ModelFactory
from ai.training.artifacts import write_sprint2_artifacts
from ai.training.training_config import (
    TrainingConfig,
    TrainingProfile,
    apply_profile,
    list_profiles,
)
from ai.training.validate import run_validation
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@pytest.fixture(scope="module")
def tiny_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny processed corpus once for the module."""

    root = tmp_path_factory.mktemp("maya_train")
    raw_real = root / "dataset" / "raw" / "REAL"
    raw_fake = root / "dataset" / "raw" / "FAKE"
    raw_real.mkdir(parents=True)
    raw_fake.mkdir(parents=True)

    for index in range(12):
        Image.new("RGB", (64, 48), color=(index * 10, 40, 80)).save(
            raw_real / f"real_{index}.jpg"
        )
        Image.new("RGB", (80, 60), color=(80, 40, index * 10)).save(
            raw_fake / f"fake_{index}.jpg"
        )

    from ai.datasets.dataset_config import DatasetConfig as DC

    ds_cfg = DC(
        project_root=root,
        random_seed=7,
        image_size=64,
        train_budget=SplitBudget(4, 4),
        validation_budget=SplitBudget(4, 4),
        test_budget=SplitBudget(4, 4),
        sample_grid_per_class=1,
        batch_size=2,
    )
    result = run_dataset_pipeline(ds_cfg)
    assert result.validation_passed
    (root / "dataset" / "versions").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "versions" / "CURRENT").write_text("v1\n", encoding="utf-8")
    return root


def _make_loaders(root: Path, batch_size: int = 2):
    ds_cfg = DatasetConfig(
        project_root=root,
        image_size=64,
        batch_size=batch_size,
        num_workers=0,
    )
    train_loader = create_dataloader(
        SplitName.TRAIN,
        config=ds_cfg,
        batch_size=batch_size,
        transform_name="train",
        image_size=64,
        num_workers=0,
    )
    val_loader = create_dataloader(
        SplitName.VALIDATION,
        config=ds_cfg,
        batch_size=batch_size,
        shuffle=False,
        transform_name="eval",
        image_size=64,
        num_workers=0,
    )
    return train_loader, val_loader


def _training_config(root: Path, **kwargs) -> TrainingConfig:
    model = ModelConfig(
        project_root=root,
        image_size=64,
        pretrained_weights=True,
        freeze_backbone=True,
        device_preference="cpu",
        batch_size=2,
    )
    defaults = dict(
        model=model,
        project_root=root,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        early_stopping_patience=5,
        max_batches_per_epoch=2,
        checkpoint_dir=root / "artifacts" / "checkpoints",
        experiment_dir=root / "artifacts" / "experiments",
        artifact_dir=root / "artifacts" / "phase3" / "sprint2",
        log_dir=root / "logs",
        profile="debug",
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def test_profile_loading() -> None:
    assert "debug" in list_profiles()
    debug = apply_profile(TrainingProfile.DEBUG.value)
    assert debug.epochs == 2
    assert debug.batch_size == 4
    prod = apply_profile("production")
    assert prod.epochs == 25
    assert prod.batch_size == 8


def test_trainer_initialization(tiny_project: Path) -> None:
    train_loader, val_loader = _make_loaders(tiny_project)
    config = _training_config(tiny_project)
    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)
    assert trainer.model is not None
    assert trainer.device.type == "cpu"
    assert trainer.start_epoch == 1


def test_single_epoch_training_and_validation(tiny_project: Path) -> None:
    train_loader, val_loader = _make_loaders(tiny_project)
    config = _training_config(tiny_project, epochs=1)
    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)
    result = trainer.fit()
    assert result.epochs_ran == 1
    assert "train_loss" in result.metrics
    assert "val_loss" in result.metrics
    assert "val_accuracy" in result.metrics


def test_validate_metrics_only(tiny_project: Path) -> None:
    _, val_loader = _make_loaders(tiny_project)
    config = _training_config(tiny_project)
    model = ModelFactory.create(config=config.model).to("cpu")
    metrics = run_validation(model, val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics


def test_checkpoint_save_load_and_resume(tiny_project: Path) -> None:
    train_loader, val_loader = _make_loaders(tiny_project)
    config = _training_config(tiny_project, epochs=1)
    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)
    result = trainer.fit()
    assert result.best_checkpoint is not None or (config.checkpoint_dir / "last.pt").exists()

    ckpt = config.checkpoint_dir / "last.pt"
    assert ckpt.exists()

    model = ModelFactory.create(config=config.model)
    # Match Trainer: only trainable parameters are in the AdamW group
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=2)
    payload = load_checkpoint(
        ckpt, model=model, optimizer=optimizer, scheduler=scheduler, map_location="cpu"
    )
    assert "epoch" in payload
    assert "metrics" in payload
    assert "config" in payload

    # Resume for one more epoch
    config2 = _training_config(tiny_project, epochs=2)
    trainer2 = Trainer(
        config=config2,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=ckpt,
    )
    assert trainer2.start_epoch == int(payload["epoch"]) + 1
    result2 = trainer2.fit()
    assert result2.epochs_ran >= 1


def test_experiment_logging(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments_isolated")
    path1 = tracker.write(
        profile="debug",
        dataset_version="v1",
        model_name="efficientnet_b0",
        hyperparameters={"epochs": 1},
        metrics={"val_loss": 0.5},
        duration_seconds=1.23,
        notes="unit-test",
    )
    path2 = tracker.write(
        profile="debug",
        dataset_version="v1",
        model_name="efficientnet_b0",
        hyperparameters={"epochs": 1},
        metrics={"val_loss": 0.4},
        duration_seconds=1.0,
    )
    assert path1.name == "experiment_001.json"
    assert path2.name == "experiment_002.json"
    data = json.loads(path1.read_text(encoding="utf-8"))
    assert data["dataset_version"] == "v1"
    assert data["model"] == "efficientnet_b0"


def test_history_csv_and_artifacts(tiny_project: Path) -> None:
    history = TrainingHistory()
    history.append(1, {"train_loss": 1.0, "val_loss": 0.9, "val_accuracy": 0.5})
    history.append(2, {"train_loss": 0.8, "val_loss": 0.7, "val_accuracy": 0.6})
    csv_path = history.export_csv(tiny_project / "artifacts" / "phase3" / "sprint2" / "training_history.csv")
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "train_loss" in text

    train_loader, val_loader = _make_loaders(tiny_project)
    config = _training_config(tiny_project, epochs=1)
    result = Trainer(config=config, train_loader=train_loader, val_loader=val_loader).fit()
    paths = write_sprint2_artifacts(result, config)
    assert paths["metrics_json"].exists()
    assert paths["experiment_summary_md"].exists()
    assert paths["checkpoint_report_md"].exists()
