"""Visualization-only helpers for MAYA evaluation artefacts.

Purpose
-------
Generate matplotlib figures for QA/intelligence reporting. No metric math
beyond what's required for axes.

Architecture
------------
Visualization Layer. Called by ``Evaluator`` after predictions exist.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai.evaluation.metrics import (
    confusion_counts,
    precision_recall_curve_points,
    roc_curve_points,
)

logger = logging.getLogger("maya.ai.evaluation.visualization")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Wrote figure → %s", path)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    *,
    class_names: tuple[str, ...] = ("REAL", "FAKE"),
    positive_class: int = 1,
) -> Path:
    counts = confusion_counts(y_true, y_pred, positive_class=positive_class)
    # Matrix layout [[TN, FP], [FN, TP]] for neg=0 / pos=1 when positive_class=1
    if positive_class == 1:
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        labels = list(class_names)
    else:
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        labels = list(reversed(class_names)) if len(class_names) == 2 else list(class_names)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(int(value)), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    *,
    positive_class: int = 1,
) -> Path:
    fpr, tpr, _ = roc_curve_points(y_true, y_score, positive_class=positive_class)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label="ROC", color="#3d8bfd")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _save(fig, output_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    *,
    positive_class: int = 1,
) -> Path:
    precision, recall, _ = precision_recall_curve_points(
        y_true, y_score, positive_class=positive_class
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#d9485f")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    return _save(fig, output_path)


def plot_confidence_distribution(
    confidences: np.ndarray,
    output_path: Path,
    *,
    threshold: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(confidences, bins=20, color="#3d8bfd", edgecolor="white")
    ax.axvline(threshold, color="#d9485f", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Confidence (max class probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    return _save(fig, output_path)


def plot_accuracy_curve(history_csv: Path, output_path: Path) -> Path | None:
    rows = _read_history(history_csv)
    if not rows:
        logger.warning("No history rows for accuracy curve")
        return None
    epochs = [int(r["epoch"]) for r in rows]
    train_acc = [float(r.get("train_accuracy", float("nan"))) for r in rows]
    val_acc = [float(r.get("val_accuracy", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(epochs, train_acc, marker="o", label="train_accuracy")
    ax.plot(epochs, val_acc, marker="o", label="val_accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.legend()
    return _save(fig, output_path)


def plot_loss_curve(history_csv: Path, output_path: Path) -> Path | None:
    rows = _read_history(history_csv)
    if not rows:
        logger.warning("No history rows for loss curve")
        return None
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r.get("train_loss", float("nan"))) for r in rows]
    val_loss = [float(r.get("val_loss", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="train_loss")
    ax.plot(epochs, val_loss, marker="o", label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    return _save(fig, output_path)


def _read_history(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        logger.warning("History CSV missing: %s", path)
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_all_plots(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    confidences: np.ndarray,
    output_dir: Path,
    class_names: tuple[str, ...],
    positive_class: int,
    threshold: float,
    history_csv: Path | None,
) -> dict[str, Path]:
    """Write all Sprint 3.3 figures into ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["confusion_matrix"] = plot_confusion_matrix(
        y_true,
        y_pred,
        output_dir / "confusion_matrix.png",
        class_names=class_names,
        positive_class=positive_class,
    )
    paths["roc_curve"] = plot_roc_curve(
        y_true, y_score, output_dir / "roc_curve.png", positive_class=positive_class
    )
    paths["precision_recall_curve"] = plot_precision_recall_curve(
        y_true,
        y_score,
        output_dir / "precision_recall_curve.png",
        positive_class=positive_class,
    )
    paths["confidence_distribution"] = plot_confidence_distribution(
        confidences,
        output_dir / "confidence_distribution.png",
        threshold=threshold,
    )
    if history_csv is not None:
        acc = plot_accuracy_curve(history_csv, output_dir / "accuracy_curve.png")
        loss = plot_loss_curve(history_csv, output_dir / "loss_curve.png")
        if acc is not None:
            paths["accuracy_curve"] = acc
        if loss is not None:
            paths["loss_curve"] = loss
    else:
        logger.warning("Skipping accuracy/loss curves — history CSV not provided")
    return paths
