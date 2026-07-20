"""Evaluation configuration for MAYA Phase 3 Sprint 3.

Purpose
-------
Centralize decision threshold, class labels, and artefact paths so evaluators
never hardcode forensic cut-offs.

Architecture
------------
Configuration Layer under ``ai/evaluation/``. Consumed by ``Evaluator`` and
reporting; independent of the training engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai.models.model_config import _project_root


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class EvaluationConfig:
    """Tunables for offline model evaluation / intelligence reporting."""

    # Decision threshold on P(positive_class)
    threshold: float = 0.5
    positive_class: int = 1  # FAKE in MAYA labeling (REAL=0, FAKE=1)
    class_names: tuple[str, ...] = ("REAL", "FAKE")

    batch_size: int = 8
    num_workers: int = 0
    max_batches: int | None = None
    device_preference: str = "auto"

    project_root: Path = field(default_factory=_project_root)
    artifact_dir: Path | None = None
    checkpoint_path: Path | None = None
    history_csv: Path | None = None

    def __post_init__(self) -> None:
        self.threshold = _env_float("MAYA_EVAL_THRESHOLD", self.threshold)
        self.project_root = Path(self.project_root)

        if self.artifact_dir is None:
            self.artifact_dir = self.project_root / "artifacts" / "phase3" / "sprint3"
        self.artifact_dir = Path(self.artifact_dir)

        if self.checkpoint_path is None:
            self.checkpoint_path = (
                self.project_root / "artifacts" / "checkpoints" / "best.pt"
            )
        self.checkpoint_path = Path(self.checkpoint_path)

        if self.history_csv is None:
            candidate = (
                self.project_root / "artifacts" / "phase3" / "sprint2" / "training_history.csv"
            )
            self.history_csv = candidate if candidate.exists() else None
        elif self.history_csv is not None:
            self.history_csv = Path(self.history_csv)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


def get_evaluation_config() -> EvaluationConfig:
    """Factory for default evaluation configuration."""

    return EvaluationConfig()
