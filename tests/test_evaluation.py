"""Pytest suite for MAYA Phase 3 Sprint 3 evaluation engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataloader import create_dataloader
from ai.datasets.dataset_config import DatasetConfig, SplitBudget, SplitName
from ai.datasets.pipeline import run_dataset_pipeline
from ai.evaluation.evaluation_config import EvaluationConfig
from ai.evaluation.evaluator import Evaluator, apply_threshold
from ai.evaluation.metrics import accuracy, f1_score, precision, recall, roc_auc
from ai.evaluation.metrics_registry import MetricRegistry, register_default_metrics
from ai.evaluation.reporting import write_metrics_json
from ai.evaluation.visualization import plot_confusion_matrix, plot_roc_curve
from ai.models.model_config import ModelConfig
from ai.models.model_factory import ModelFactory


@pytest.fixture(scope="module")
def tiny_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("maya_eval")
    raw_real = root / "dataset" / "raw" / "REAL"
    raw_fake = root / "dataset" / "raw" / "FAKE"
    raw_real.mkdir(parents=True)
    raw_fake.mkdir(parents=True)
    for index in range(12):
        Image.new("RGB", (64, 48), color=(index * 10, 40, 80)).save(raw_real / f"r_{index}.jpg")
        Image.new("RGB", (80, 60), color=(80, 40, index * 10)).save(raw_fake / f"f_{index}.jpg")

    ds_cfg = DatasetConfig(
        project_root=root,
        random_seed=3,
        image_size=64,
        train_budget=SplitBudget(4, 4),
        validation_budget=SplitBudget(4, 4),
        test_budget=SplitBudget(4, 4),
        sample_grid_per_class=1,
        batch_size=2,
    )
    assert run_dataset_pipeline(ds_cfg).validation_passed
    (root / "dataset" / "versions").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "versions" / "CURRENT").write_text("v1\n", encoding="utf-8")
    # Synthetic training history for curve plots
    hist = root / "artifacts" / "phase3" / "sprint2" / "training_history.csv"
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(
        "epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate\n"
        "1,1.0,0.5,0.9,0.55,0.0001\n"
        "2,0.8,0.6,0.7,0.65,5e-05\n",
        encoding="utf-8",
    )
    return root


def test_metric_correctness() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert precision(y_true, y_pred, positive_class=1) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred, positive_class=1) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred, positive_class=1) == pytest.approx(0.8)


def test_roc_auc_and_registry() -> None:
    register_default_metrics()
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    y_pred = (y_score >= 0.5).astype(int)
    auc = roc_auc(y_true, y_score=y_score, positive_class=1)
    assert 0.0 <= auc <= 1.0
    all_metrics = MetricRegistry.calculate_all(
        y_true, y_pred, y_score=y_score, positive_class=1
    )
    assert "accuracy" in all_metrics
    assert "roc_auc" in all_metrics
    assert "matthews_corrcoef" in all_metrics


def test_registry_extensibility() -> None:
    register_default_metrics()

    @MetricRegistry.register("always_half")
    def always_half(y_true, y_pred, **kwargs):  # noqa: ANN001
        return 0.5

    assert MetricRegistry.calculate("always_half", np.array([0]), np.array([0])) == 0.5
    MetricRegistry._registry.pop("always_half", None)


def test_threshold_logic() -> None:
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.55, 0.45]])
    preds = apply_threshold(probs, threshold=0.5, positive_class=1)
    assert preds.tolist() == [0, 1, 0]
    preds_hi = apply_threshold(probs, threshold=0.7, positive_class=1)
    assert preds_hi.tolist() == [0, 0, 0]


def test_confusion_and_roc_plots(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_score = np.array([0.2, 0.6, 0.7, 0.9, 0.4])
    cm = plot_confusion_matrix(y_true, y_pred, tmp_path / "confusion_matrix.png")
    roc = plot_roc_curve(y_true, y_score, tmp_path / "roc_curve.png")
    assert cm.exists() and roc.exists()


def test_json_export_and_confidence(tiny_project: Path) -> None:
    model_cfg = ModelConfig(
        project_root=tiny_project,
        image_size=64,
        pretrained_weights=True,
        freeze_backbone=True,
        device_preference="cpu",
    )
    ds_cfg = DatasetConfig(project_root=tiny_project, image_size=64, batch_size=2, num_workers=0)
    loader = create_dataloader(
        SplitName.TEST,
        config=ds_cfg,
        batch_size=2,
        shuffle=False,
        transform_name="eval",
        image_size=64,
        num_workers=0,
    )
    eval_cfg = EvaluationConfig(
        project_root=tiny_project,
        threshold=0.5,
        artifact_dir=tiny_project / "artifacts" / "phase3" / "sprint3",
        history_csv=tiny_project / "artifacts" / "phase3" / "sprint2" / "training_history.csv",
        device_preference="cpu",
        max_batches=2,
    )
    model = ModelFactory.create(config=model_cfg)
    result = Evaluator(model=model, test_loader=loader, config=eval_cfg, model_config=model_cfg).evaluate()

    assert result.n_samples > 0
    assert "accuracy" in result.metrics
    assert (eval_cfg.artifact_dir / "metrics.json").exists()
    assert (eval_cfg.artifact_dir / "evaluation_result.json").exists()
    assert (eval_cfg.artifact_dir / "confusion_matrix.png").exists()
    assert (eval_cfg.artifact_dir / "roc_curve.png").exists()
    assert (eval_cfg.artifact_dir / "precision_recall_curve.png").exists()
    assert (eval_cfg.artifact_dir / "confidence_distribution.png").exists()
    assert (eval_cfg.artifact_dir / "accuracy_curve.png").exists()
    assert (eval_cfg.artifact_dir / "loss_curve.png").exists()
    assert (eval_cfg.artifact_dir / "classification_report.txt").exists()
    assert (eval_cfg.artifact_dir / "evaluation_summary.md").exists()
    assert (eval_cfg.artifact_dir / "phase3_report.md").exists()

    payload = json.loads((eval_cfg.artifact_dir / "evaluation_result.json").read_text(encoding="utf-8"))
    assert payload["threshold"] == 0.5
    sample = payload["predictions"][0]
    assert "predicted_class" in sample
    assert "confidence_percentage" in sample
    assert "raw_probability" in sample
    assert "threshold_used" in sample


def test_write_metrics_json(tmp_path: Path) -> None:
    path = write_metrics_json({"accuracy": 0.9}, tmp_path / "metrics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["accuracy"] == 0.9
