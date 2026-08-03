"""Explainability configuration for MAYA Phase 4.

Purpose
-------
Centralize visualization tunables and explainer selection so modules never
hardcode algorithm names, opacity, colormaps, or artefact paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ai.models.model_config import ModelName, _project_root


class ColorMap(str, Enum):
    """Supported heatmap colour maps."""

    JET = "jet"
    VIRIDIS = "viridis"
    INFERNO = "inferno"
    HOT = "hot"
    MAGMA = "magma"


class InterpolationMethod(str, Enum):
    """Heatmap resize interpolation."""

    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


@dataclass
class ExplainabilityConfig:
    """Tunables for the Explainability Engine / Multi-Explainer Framework."""

    # Active single-explainer selection (Sprint 4.1 explain() path)
    explainer_name: str = "gradcam"
    # Multi-explainer comparison set (Sprint 4.2) — order preserved in artefacts
    comparison_explainers: tuple[str, ...] = (
        "gradcam",
        "gradcam_plus_plus",
        "layercam",
        "scorecam",
        "eigencam",
    )
    scorecam_batch_size: int = 8

    opacity: float = 0.45
    color_map: ColorMap = ColorMap.JET
    output_resolution: int = 224
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    heatmap_eps: float = 1e-8
    target_layer_override: str | None = None
    target_class: int | None = None  # None → use predicted class
    device_preference: str = "cpu"
    model_name: str = ModelName.EFFICIENTNET_B0.value
    model_version: str = "sprint3.2-best"
    dataset_version: str = "MAYA-Kaggle-Balanced-v1"
    image_size: int = 224

    project_root: Path = field(default_factory=_project_root)
    checkpoint_path: Path | None = None
    artifact_dir: Path | None = None
    comparison_artifact_dir: Path | None = None
    id_state_path: Path | None = None

    def __post_init__(self) -> None:
        self.opacity = _env_float("MAYA_XAI_OPACITY", self.opacity)
        self.explainer_name = _env_str("MAYA_XAI_EXPLAINER", self.explainer_name).lower()
        self.project_root = Path(self.project_root)

        if isinstance(self.color_map, str):
            self.color_map = ColorMap(self.color_map.lower())
        if isinstance(self.interpolation, str):
            self.interpolation = InterpolationMethod(self.interpolation.lower())

        if isinstance(self.comparison_explainers, list):
            self.comparison_explainers = tuple(self.comparison_explainers)

        if self.checkpoint_path is None:
            self.checkpoint_path = (
                self.project_root / "artifacts" / "checkpoints" / "best.pt"
            )
        self.checkpoint_path = Path(self.checkpoint_path)

        if self.artifact_dir is None:
            self.artifact_dir = (
                self.project_root / "artifacts" / "phase4" / "sprint1"
            )
        self.artifact_dir = Path(self.artifact_dir)

        if self.comparison_artifact_dir is None:
            self.comparison_artifact_dir = (
                self.project_root / "artifacts" / "phase4" / "sprint2"
            )
        self.comparison_artifact_dir = Path(self.comparison_artifact_dir)

        if self.id_state_path is None:
            self.id_state_path = self.artifact_dir / "investigation_id_state.json"
        self.id_state_path = Path(self.id_state_path)

        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")
        if self.output_resolution < 1:
            raise ValueError("output_resolution must be >= 1")
        if self.scorecam_batch_size < 1:
            raise ValueError("scorecam_batch_size must be >= 1")


def get_explainability_config() -> ExplainabilityConfig:
    """Return default explainability configuration."""

    return ExplainabilityConfig()
