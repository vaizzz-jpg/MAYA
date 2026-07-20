"""Text / JSON / Markdown reporting for MAYA evaluation (no graphs).

Purpose
-------
Serialize metrics, predictions, and summaries for investigators and
appendices without coupling to matplotlib.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.evaluation.evaluation_config import EvaluationConfig

logger = logging.getLogger("maya.ai.evaluation.reporting")


def write_classification_report(
    metrics: dict[str, float],
    counts: dict[str, int],
    output_path: Path,
    *,
    class_names: tuple[str, ...],
    threshold: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MAYA Classification Report",
        "==========================",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Threshold: {threshold}",
        f"Classes: {', '.join(class_names)}",
        "",
        "Confusion counts:",
        f"  TP={counts.get('tp')} TN={counts.get('tn')} FP={counts.get('fp')} FN={counts.get('fn')}",
        "",
        "Metrics:",
    ]
    for key in sorted(metrics.keys()):
        value = metrics[key]
        lines.append(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote classification report → %s", output_path)
    return output_path


def write_metrics_json(metrics: dict[str, float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote metrics JSON → %s", output_path)
    return output_path


def write_evaluation_result_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation_result JSON → %s", output_path)
    return output_path


def write_evaluation_summary_md(
    metrics: dict[str, float],
    output_path: Path,
    *,
    threshold: float,
    n_samples: int,
    figure_names: list[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAYA Evaluation Summary",
        "",
        f"- Samples: **{n_samples}**",
        f"- Decision threshold: **{threshold}**",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- F1: `{metrics.get('f1')}`",
        f"- ROC AUC: `{metrics.get('roc_auc')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        "",
        "## Figures",
        "",
    ]
    for name in figure_names:
        lines.append(f"- `{name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote evaluation summary → %s", output_path)
    return output_path


def write_phase3_report_md(
    metrics: dict[str, float],
    config: EvaluationConfig,
    output_path: Path,
    *,
    checkpoint: str,
    dataset_version: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAYA Phase 3 — Consolidated AI Report",
        "",
        "## Scope",
        "",
        "- Sprint 3.1: EfficientNet-B0 + ModelFactory",
        "- Sprint 3.2: Training engine (AdamW, checkpoints, experiments)",
        "- Sprint 3.3: Evaluation & intelligence reporting",
        "",
        "## Evaluation setup",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Dataset version: `{dataset_version}`",
        f"- Threshold: `{config.threshold}`",
        f"- Positive class index: `{config.positive_class}` ({config.class_names[config.positive_class] if config.positive_class < len(config.class_names) else config.positive_class})",
        "",
        "## Key metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key in sorted(metrics.keys()):
        lines.append(f"| {key} | {metrics[key]:.6f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This report does not include Grad-CAM, SHA hashing, Flask APIs, or UI.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote phase3 report → %s", output_path)
    return output_path


def build_prediction_records(
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    *,
    class_names: tuple[str, ...],
    threshold: float,
    positive_class: int,
) -> list[dict[str, Any]]:
    """Serialize per-sample predictions with confidence metadata."""

    records: list[dict[str, Any]] = []
    for idx, (truth, pred, probs) in enumerate(zip(y_true, y_pred, probabilities)):
        conf = float(max(probs))
        records.append(
            {
                "index": idx,
                "true_class": int(truth),
                "true_label": class_names[truth] if 0 <= truth < len(class_names) else str(truth),
                "predicted_class": int(pred),
                "predicted_label": class_names[pred] if 0 <= pred < len(class_names) else str(pred),
                "confidence_percentage": round(conf * 100.0, 4),
                "raw_probability": [float(p) for p in probs],
                "positive_class_probability": float(probs[positive_class])
                if positive_class < len(probs)
                else None,
                "threshold_used": float(threshold),
            }
        )
    return records
