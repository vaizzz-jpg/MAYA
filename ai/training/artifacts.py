"""Write Sprint 3.2 training artefacts (CSV, metrics, markdown summaries)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ai.engine.history import TrainingHistory
from ai.engine.trainer import TrainerResult
from ai.training.training_config import TrainingConfig

logger = logging.getLogger("maya.ai.training.artifacts")


def write_sprint2_artifacts(
    result: TrainerResult,
    config: TrainingConfig,
) -> dict[str, Path]:
    """Emit standard Sprint 2 artefact files under ``config.artifact_dir``."""

    out = Path(config.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    history_csv = result.history.export_csv(out / "training_history.csv")

    metrics_path = out / "metrics.json"
    metrics_payload: dict[str, Any] = {
        "profile": config.profile,
        "final_metrics": result.metrics,
        "epochs_ran": result.epochs_ran,
        "duration_seconds": result.duration_seconds,
        "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
        "experiment_path": str(result.experiment_path) if result.experiment_path else None,
        "best_row": result.history.best(config.monitor_metric, config.monitor_mode),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary_path = out / "experiment_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# MAYA Phase 3 Sprint 2 — Experiment Summary",
                "",
                f"- Profile: `{config.profile}`",
                f"- Model: `{config.model.model_name}`",
                f"- Epochs ran: **{result.epochs_ran}** / {config.epochs}",
                f"- Duration (s): **{result.duration_seconds:.2f}**",
                f"- Final val_loss: `{result.metrics.get('val_loss')}`",
                f"- Final val_accuracy: `{result.metrics.get('val_accuracy')}`",
                f"- Experiment JSON: `{result.experiment_path}`",
                f"- Best checkpoint: `{result.best_checkpoint}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ckpt_report = out / "checkpoint_report.md"
    ckpt_report.write_text(
        "\n".join(
            [
                "# Checkpoint Report",
                "",
                f"- Checkpoint directory: `{config.checkpoint_dir}`",
                f"- Best weights: `{result.best_checkpoint}`",
                f"- Last weights: `{Path(config.checkpoint_dir) / 'last.pt'}`",
                "",
                "Checkpoints include model, optimizer, scheduler, epoch, metrics, and config.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Sprint 2 artefacts written under %s", out)
    return {
        "training_history_csv": history_csv,
        "metrics_json": metrics_path,
        "experiment_summary_md": summary_path,
        "checkpoint_report_md": ckpt_report,
    }
