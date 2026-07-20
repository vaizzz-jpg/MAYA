"""Evaluation coordinator for MAYA Sprint 3.3.

Purpose
-------
Own orchestration only: gather predictions → registry metrics → plots → reports.
Does not compute metrics inline and does not train models.

Architecture
------------
Coordinator between Phase 2 DataLoader, Sprint 3.1 model, and evaluation modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai.engine.device import resolve_device
from ai.engine.experiment import read_dataset_version
from ai.evaluation.evaluation_config import EvaluationConfig
from ai.evaluation.metrics import confusion_counts
from ai.evaluation.metrics_registry import MetricRegistry, register_default_metrics
from ai.evaluation.reporting import (
    build_prediction_records,
    write_classification_report,
    write_evaluation_result_json,
    write_evaluation_summary_md,
    write_metrics_json,
    write_phase3_report_md,
)
from ai.evaluation.visualization import generate_all_plots
from ai.models.model_config import ModelConfig

logger = logging.getLogger("maya.ai.evaluation.evaluator")


@dataclass
class EvaluationResult:
    """Structured outcome of a full evaluation run."""

    metrics: dict[str, float]
    predictions: list[dict[str, Any]]
    figure_paths: dict[str, Path]
    report_paths: dict[str, Path]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    threshold: float
    n_samples: int


def apply_threshold(
    probabilities: np.ndarray,
    *,
    threshold: float,
    positive_class: int,
) -> np.ndarray:
    """Map class probabilities to hard labels using a configurable threshold.

    For binary tasks: predict ``positive_class`` when P(pos) >= threshold,
    otherwise the complementary class.
    """

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        pos_prob = probs
    else:
        pos_prob = probs[:, int(positive_class)]
    pos = int(positive_class)
    neg = 0 if pos == 1 else 1
    return np.where(pos_prob >= threshold, pos, neg).astype(int)


@dataclass
class Evaluator:
    """Coordinate offline evaluation without embedding metric formulas."""

    model: nn.Module
    test_loader: DataLoader
    config: EvaluationConfig
    model_config: ModelConfig | None = None

    def __post_init__(self) -> None:
        register_default_metrics()
        preference = self.config.device_preference
        if preference == "auto" and self.model_config is not None:
            preference = self.model_config.device_preference
        self.device = resolve_device(preference)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _collect_predictions(
        self,
    ) -> tuple[list[int], list[int], list[list[float]], np.ndarray, np.ndarray]:
        """Stream the loader once; apply configurable threshold."""

        y_true: list[int] = []
        probabilities: list[list[float]] = []
        confidences: list[float] = []
        pos = self.config.positive_class
        threshold = self.config.threshold

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if self.config.max_batches is not None and batch_idx >= self.config.max_batches:
                break

            images = images.to(self.device)
            labels_cpu = labels.detach().cpu().numpy().astype(int)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for label, prob_row in zip(labels_cpu, probs):
                y_true.append(int(label))
                probabilities.append([float(p) for p in prob_row])
                confidences.append(float(np.max(prob_row)))

            del images, logits, probs

        prob_matrix = np.asarray(probabilities, dtype=float)
        y_pred = apply_threshold(
            prob_matrix, threshold=threshold, positive_class=pos
        ).tolist()
        y_score = prob_matrix[:, pos] if prob_matrix.size else np.asarray([], dtype=float)
        return y_true, y_pred, probabilities, np.asarray(confidences), y_score

    def evaluate(self) -> EvaluationResult:
        """Run the full evaluation pipeline and write Sprint 3 artefacts."""

        logger.info(
            "Evaluation start threshold=%.3f positive_class=%s device=%s",
            self.config.threshold,
            self.config.positive_class,
            self.device,
        )

        y_true_list, y_pred_list, probabilities, confidences, y_score = (
            self._collect_predictions()
        )
        y_true = np.asarray(y_true_list, dtype=int)
        y_pred = np.asarray(y_pred_list, dtype=int)

        metrics = MetricRegistry.calculate_all(
            y_true,
            y_pred,
            y_score=y_score,
            positive_class=self.config.positive_class,
        )
        counts = confusion_counts(
            y_true, y_pred, positive_class=self.config.positive_class
        )
        metrics_with_counts = {
            **metrics,
            **{f"count_{k}": float(v) for k, v in counts.items()},
        }

        out = Path(self.config.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)

        figures = generate_all_plots(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            confidences=confidences,
            output_dir=out,
            class_names=self.config.class_names,
            positive_class=self.config.positive_class,
            threshold=self.config.threshold,
            history_csv=self.config.history_csv,
        )

        predictions = build_prediction_records(
            y_true_list,
            y_pred_list,
            probabilities,
            class_names=self.config.class_names,
            threshold=self.config.threshold,
            positive_class=self.config.positive_class,
        )

        dataset_version = read_dataset_version(self.config.project_root)
        checkpoint = (
            str(self.config.checkpoint_path) if self.config.checkpoint_path else "in-memory"
        )

        eval_payload = {
            "threshold": self.config.threshold,
            "positive_class": self.config.positive_class,
            "class_names": list(self.config.class_names),
            "dataset_version": dataset_version,
            "checkpoint": checkpoint,
            "n_samples": len(predictions),
            "metrics": metrics,
            "confusion": counts,
            "predictions": predictions,
        }

        reports = {
            "classification_report": write_classification_report(
                metrics,
                counts,
                out / "classification_report.txt",
                class_names=self.config.class_names,
                threshold=self.config.threshold,
            ),
            "metrics_json": write_metrics_json(metrics_with_counts, out / "metrics.json"),
            "evaluation_result_json": write_evaluation_result_json(
                eval_payload, out / "evaluation_result.json"
            ),
            "evaluation_summary_md": write_evaluation_summary_md(
                metrics,
                out / "evaluation_summary.md",
                threshold=self.config.threshold,
                n_samples=len(predictions),
                figure_names=[p.name for p in figures.values()],
            ),
            "phase3_report_md": write_phase3_report_md(
                metrics,
                self.config,
                out / "phase3_report.md",
                checkpoint=checkpoint,
                dataset_version=dataset_version,
            ),
        }

        logger.info("Evaluation complete n=%s metrics=%s", len(predictions), metrics)
        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            figure_paths=figures,
            report_paths=reports,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            threshold=self.config.threshold,
            n_samples=len(predictions),
        )
