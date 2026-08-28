"""Explainability configuration for MAYA Phase 4.

Purpose
-------
Centralize visualization tunables and explainer selection so modules never
hardcode algorithm names, opacity, colormaps, or artefact paths.

Sprint 4.5 extends this module with nested Advanced XAI settings (SHAP,
faithfulness, counterfactual, trust) — same conventions, not a parallel system.
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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
    target_class: int | None = None  # None → explain predicted class
    device_preference: str = "cpu"
    model_name: str = ModelName.EFFICIENTNET_B0.value
    model_version: str = "sprint3.2-best"
    dataset_version: str = "MAYA-Kaggle-Balanced-v1"
    image_size: int = 224

    project_root: Path = field(default_factory=_project_root)
    checkpoint_path: Path | None = None
    # Output directory is ALWAYS from this field (or explain(artifact_dir=...)),
    # never inferred from the input image path. Default keeps Sprint 4.1 layout.
    artifact_dir: Path | None = None
    comparison_artifact_dir: Path | None = None
    # Share with InferenceConfig.id_state_path to keep INV-YYYY-NNNNNN continuous
    # across predict → explain. Defaults under artifact_dir for Sprint 4.1.
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


@dataclass
class ShapSettings:
    """CPU-friendly SHAP attribution settings."""

    enabled: bool = True
    max_evaluations: int = 32
    batch_size: int = 4
    masker_blur_kernel: int = 32
    visualization: bool = True

    def __post_init__(self) -> None:
        self.enabled = _env_bool("MAYA_XAI_SHAP_ENABLED", self.enabled)
        self.max_evaluations = _env_int(
            "MAYA_XAI_SHAP_MAX_EVALS", self.max_evaluations
        )
        if self.max_evaluations < 2:
            raise ValueError("shap.max_evaluations must be >= 2")
        if self.batch_size < 1:
            raise ValueError("shap.batch_size must be >= 1")
        if self.masker_blur_kernel < 1:
            raise ValueError("shap.masker_blur_kernel must be >= 1")


@dataclass
class FaithfulnessSettings:
    """Deletion / occlusion faithfulness evaluation settings."""

    enabled: bool = True
    steps: int = 5
    perturbation_method: str = "mask"  # mask | blur | neutral
    fraction_per_step: float = 0.10
    importance_threshold: float = 0.2

    def __post_init__(self) -> None:
        self.enabled = _env_bool("MAYA_XAI_FAITHFULNESS_ENABLED", self.enabled)
        self.steps = _env_int("MAYA_XAI_FAITHFULNESS_STEPS", self.steps)
        self.perturbation_method = _env_str(
            "MAYA_XAI_FAITHFULNESS_METHOD", self.perturbation_method
        ).lower()
        if self.steps < 1:
            raise ValueError("faithfulness.steps must be >= 1")
        if self.perturbation_method not in {"mask", "blur", "neutral"}:
            raise ValueError(
                f"Unsupported faithfulness perturbation_method: "
                f"{self.perturbation_method}"
            )
        if not 0.0 < self.fraction_per_step <= 1.0:
            raise ValueError("fraction_per_step must be in (0, 1]")


@dataclass
class CounterfactualSettings:
    """Controlled model counterfactual perturbation settings."""

    enabled: bool = True
    perturbation_methods: tuple[str, ...] = ("mask", "blur", "neutral")
    importance_threshold: float = 0.35
    blur_radius: int = 8
    neutral_value: float = 0.0  # ImageNet-normalized tensor fill

    def __post_init__(self) -> None:
        self.enabled = _env_bool("MAYA_XAI_COUNTERFACTUAL_ENABLED", self.enabled)
        if isinstance(self.perturbation_methods, list):
            self.perturbation_methods = tuple(self.perturbation_methods)
        cleaned = tuple(m.strip().lower() for m in self.perturbation_methods)
        for method in cleaned:
            if method not in {"mask", "blur", "neutral"}:
                raise ValueError(f"Unsupported counterfactual method: {method}")
        self.perturbation_methods = cleaned
        if self.blur_radius < 1:
            raise ValueError("blur_radius must be >= 1")


@dataclass
class TrustScoreWeights:
    """Weights for the MAYA Explanation Trust Score (must sum > 0).

    This is an engineering composite — not a probability that an explanation
    is scientifically 'true'.
    """

    quality: float = 0.25
    faithfulness: float = 0.30
    consistency: float = 0.20
    localization: float = 0.15
    agreement: float = 0.10

    def normalized(self) -> TrustScoreWeights:
        total = (
            self.quality
            + self.faithfulness
            + self.consistency
            + self.localization
            + self.agreement
        )
        if total <= 0:
            raise ValueError("TrustScoreWeights must sum to a positive value")
        return TrustScoreWeights(
            quality=self.quality / total,
            faithfulness=self.faithfulness / total,
            consistency=self.consistency / total,
            localization=self.localization / total,
            agreement=self.agreement / total,
        )


@dataclass
class AdvancedXAIConfig:
    """Sprint 4.5 Advanced Explainability & Trust Layer configuration."""

    shap: ShapSettings = field(default_factory=ShapSettings)
    faithfulness: FaithfulnessSettings = field(default_factory=FaithfulnessSettings)
    counterfactual: CounterfactualSettings = field(
        default_factory=CounterfactualSettings
    )
    trust: TrustScoreWeights = field(default_factory=TrustScoreWeights)

    opacity: float = 0.45
    color_map: ColorMap = ColorMap.JET
    output_resolution: int = 224
    heatmap_eps: float = 1e-8
    target_class: int | None = None
    device_preference: str = "cpu"
    model_name: str = ModelName.EFFICIENTNET_B0.value
    model_version: str = "sprint3.2-best"
    dataset_version: str = "MAYA-Kaggle-Balanced-v1"
    image_size: int = 224
    primary_cam_explainer: str = "gradcam"

    project_root: Path = field(default_factory=_project_root)
    checkpoint_path: Path | None = None
    artifact_dir: Path | None = None
    id_state_path: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.trust = self.trust.normalized()
        self.primary_cam_explainer = self.primary_cam_explainer.strip().lower()

        if isinstance(self.color_map, str):
            self.color_map = ColorMap(self.color_map.lower())

        if self.checkpoint_path is None:
            self.checkpoint_path = (
                self.project_root / "artifacts" / "checkpoints" / "best.pt"
            )
        self.checkpoint_path = Path(self.checkpoint_path)

        if self.artifact_dir is None:
            self.artifact_dir = (
                self.project_root / "artifacts" / "phase4" / "sprint5"
            )
        self.artifact_dir = Path(self.artifact_dir)

        if self.id_state_path is None:
            self.id_state_path = self.artifact_dir / "investigation_id_state.json"
        self.id_state_path = Path(self.id_state_path)

        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")
        if self.output_resolution < 1:
            raise ValueError("output_resolution must be >= 1")


def get_explainability_config() -> ExplainabilityConfig:
    """Return default explainability configuration."""

    return ExplainabilityConfig()


def get_advanced_xai_config() -> AdvancedXAIConfig:
    """Return default Sprint 4.5 Advanced XAI configuration."""

    return AdvancedXAIConfig()
