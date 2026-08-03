"""Configuration for the Explainability Validation & Benchmark Suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai.explainability.registry import DEFAULT_EXPLAINER_NAMES
from ai.models.model_config import ModelName, _project_root


@dataclass
class BenchmarkWeights:
    """Weights for the overall benchmark score (must sum ≈ 1.0)."""

    quality: float = 0.35
    focus: float = 0.15
    localization: float = 0.15
    coverage: float = 0.10
    agreement: float = 0.15
    performance: float = 0.10  # higher = faster / leaner

    def normalized(self) -> BenchmarkWeights:
        total = (
            self.quality
            + self.focus
            + self.localization
            + self.coverage
            + self.agreement
            + self.performance
        )
        if total <= 0:
            raise ValueError("BenchmarkWeights must sum to a positive value")
        return BenchmarkWeights(
            quality=self.quality / total,
            focus=self.focus / total,
            localization=self.localization / total,
            coverage=self.coverage / total,
            agreement=self.agreement / total,
            performance=self.performance / total,
        )


@dataclass
class ExplainabilityBenchmarkConfig:
    """Tunables for Sprint 4.4 explainer benchmarking."""

    explainer_names: tuple[str, ...] = DEFAULT_EXPLAINER_NAMES
    weights: BenchmarkWeights = field(default_factory=BenchmarkWeights)
    coverage_threshold: float = 0.2
    warmup_runs: int = 0
    timed_runs: int = 1
    scorecam_batch_size: int = 8
    output_resolution: int = 224
    device_preference: str = "cpu"
    model_name: str = ModelName.EFFICIENTNET_B0.value
    model_version: str = "sprint3.2-best"

    project_root: Path = field(default_factory=_project_root)
    checkpoint_path: Path | None = None
    artifact_dir: Path | None = None
    sample_image: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.weights = self.weights.normalized()
        if isinstance(self.explainer_names, list):
            self.explainer_names = tuple(self.explainer_names)
        # Drop alias duplicates (gradcam++ mirrors gradcam_plus_plus)
        cleaned: list[str] = []
        for name in self.explainer_names:
            key = name.strip().lower()
            if key == "gradcam++":
                key = "gradcam_plus_plus"
            if key not in cleaned:
                cleaned.append(key)
        self.explainer_names = tuple(cleaned)

        if self.checkpoint_path is None:
            self.checkpoint_path = (
                self.project_root / "artifacts" / "checkpoints" / "best.pt"
            )
        self.checkpoint_path = Path(self.checkpoint_path)

        if self.artifact_dir is None:
            self.artifact_dir = (
                self.project_root / "artifacts" / "phase4" / "sprint4"
            )
        self.artifact_dir = Path(self.artifact_dir)

        if self.timed_runs < 1:
            raise ValueError("timed_runs must be >= 1")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")


def get_explainability_benchmark_config() -> ExplainabilityBenchmarkConfig:
    return ExplainabilityBenchmarkConfig()
