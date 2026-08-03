"""Explanation Analytics & Trust Engine coordinator (Sprint 4.3).

Consumes explanation heatmaps only — does not run Grad-CAM or other explainers.
Independent of training, evaluation, and inference packages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai.explainability.analytics.confidence import (
    ConfidenceBreakdown,
    compute_confidence_breakdown,
)
from ai.explainability.analytics.consistency import (
    ConsistencyMetrics,
    compute_consistency_metrics,
)
from ai.explainability.analytics.explanation_quality import (
    ExplanationQuality,
    compute_explanation_quality,
)
from ai.explainability.analytics.focus_metrics import FocusMetrics, compute_focus_metrics
from ai.explainability.analytics.localization import (
    LocalizationMetrics,
    compute_localization_metrics,
)
from ai.explainability.analytics.report import (
    write_analytics_report,
    write_explanation_analysis_json,
    write_focus_metrics_json,
    write_summary_json,
    write_summary_markdown,
)
from ai.explainability.analytics.statistics import (
    ActivationStatistics,
    compute_activation_statistics,
)
from ai.explainability.analytics.summary import (
    InvestigatorSummary,
    build_investigator_summary,
)
from ai.explainability.analytics.visualization import write_analytics_visualizations
from ai.models.model_config import _project_root

logger = logging.getLogger("maya.ai.explainability.analytics.analyzer")


@dataclass
class AnalyticsConfig:
    """Tunables for explanation analytics."""

    coverage_threshold: float = 0.2
    artifact_dir: Path | None = None
    project_root: Path = field(default_factory=_project_root)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.artifact_dir is None:
            self.artifact_dir = (
                self.project_root / "artifacts" / "phase4" / "sprint3"
            )
        self.artifact_dir = Path(self.artifact_dir)
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0, 1]")


@dataclass
class ExplanationAnalytics:
    """Full analytics payload for one primary explanation (+ optional peers)."""

    primary_explainer: str
    prediction: str
    image_name: str
    focus: FocusMetrics
    localization: LocalizationMetrics
    statistics: ActivationStatistics
    quality: ExplanationQuality
    confidence: ConfidenceBreakdown
    summary: InvestigatorSummary
    consistency: ConsistencyMetrics | None = None
    per_explainer_focus: dict[str, FocusMetrics] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_explainer": self.primary_explainer,
            "prediction": self.prediction,
            "image_name": self.image_name,
            "timestamp": self.timestamp,
            "focus": self.focus.to_dict(),
            "localization": self.localization.to_dict(),
            "statistics": self.statistics.to_dict(),
            "quality": self.quality.to_dict(),
            "confidence": self.confidence.to_dict(),
            "summary": self.summary.to_dict(),
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "per_explainer_focus": {
                k: v.to_dict() for k, v in self.per_explainer_focus.items()
            },
            "artifact_paths": dict(self.artifact_paths),
        }


class ExplanationAnalyticsEngine:
    """Coordinate focus, localization, quality, consistency, and reporting."""

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self.config = config or AnalyticsConfig()

    def analyze(
        self,
        heatmaps: dict[str, np.ndarray],
        *,
        primary_explainer: str | None = None,
        prediction: str = "",
        model_confidence: float = 0.0,
        image_name: str = "",
        write_artifacts: bool = True,
    ) -> ExplanationAnalytics:
        """Analyze one or more explanation heatmaps.

        Args:
            heatmaps: Mapping of explainer name → normalized 2-D map in ``[0, 1]``.
            primary_explainer: Key used for primary metrics; default = first key.
            prediction: Model label for the related investigation (metadata only).
            model_confidence: Classifier confidence 0–100 (kept separate from XAI).
            image_name: Evidence filename for reports.
            write_artifacts: Persist Sprint 4.3 outputs under ``artifact_dir``.
        """

        if not heatmaps:
            raise ValueError("heatmaps must contain at least one explainer map")

        names = list(heatmaps.keys())
        primary = primary_explainer or names[0]
        if primary not in heatmaps:
            raise KeyError(f"primary_explainer '{primary}' not in heatmaps")

        cfg = self.config
        primary_map = np.asarray(heatmaps[primary], dtype=np.float64)

        focus = compute_focus_metrics(
            primary_map, coverage_threshold=cfg.coverage_threshold
        )
        localization = compute_localization_metrics(
            primary_map, activation_threshold=cfg.coverage_threshold
        )
        stats = compute_activation_statistics(primary_map)

        consistency: ConsistencyMetrics | None = None
        per_focus: dict[str, FocusMetrics] = {}
        if len(heatmaps) >= 2:
            consistency = compute_consistency_metrics(heatmaps)
            for name, arr in heatmaps.items():
                per_focus[name] = compute_focus_metrics(
                    arr, coverage_threshold=cfg.coverage_threshold
                )
        else:
            per_focus[primary] = focus

        quality = compute_explanation_quality(
            focus, localization, consistency=consistency
        )
        confidence = compute_confidence_breakdown(
            model_confidence=model_confidence,
            quality=quality,
            focus=focus,
            consistency=consistency,
        )
        summary = build_investigator_summary(
            explainer_name=primary,
            prediction=prediction or "UNKNOWN",
            focus=focus,
            localization=localization,
            quality=quality,
            confidence=confidence,
            consistency=consistency,
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        artifact_paths: dict[str, str] = {}

        result = ExplanationAnalytics(
            primary_explainer=primary,
            prediction=prediction,
            image_name=image_name,
            focus=focus,
            localization=localization,
            statistics=stats,
            quality=quality,
            confidence=confidence,
            summary=summary,
            consistency=consistency,
            per_explainer_focus=per_focus,
            artifact_paths=artifact_paths,
            timestamp=timestamp,
        )

        if write_artifacts:
            artifact_paths = self._persist(result, primary_map)
            result.artifact_paths = artifact_paths

        logger.info(
            "Analytics complete primary=%s quality=%.1f agreement=%s",
            primary,
            quality.quality_score,
            None if consistency is None else consistency.agreement_score,
        )
        return result

    def analyze_from_heatmap_images(
        self,
        heatmap_paths: dict[str, Path | str],
        **kwargs: Any,
    ) -> ExplanationAnalytics:
        """Load RGB/L heatmap images, convert to ``[0, 1]`` intensity maps, analyze."""

        heatmaps = {
            name: load_heatmap_image(path) for name, path in heatmap_paths.items()
        }
        return self.analyze(heatmaps, **kwargs)

    def _persist(
        self,
        result: ExplanationAnalytics,
        primary_map: np.ndarray,
    ) -> dict[str, str]:
        out = Path(self.config.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)

        viz = write_analytics_visualizations(
            primary_map,
            result.focus,
            result.localization,
            result.statistics,
            out,
        )
        analysis = result.to_dict()
        paths = {
            "activation_histogram": str(viz["activation_histogram"]),
            "coverage_map": str(viz["coverage_map"]),
            "focus_distribution": str(viz["focus_distribution"]),
            "explanation_analysis_json": str(
                write_explanation_analysis_json(
                    analysis, out / "explanation_analysis.json"
                )
            ),
            "focus_metrics_json": str(
                write_focus_metrics_json(
                    {
                        "primary": result.focus.to_dict(),
                        "per_explainer": {
                            k: v.to_dict()
                            for k, v in result.per_explainer_focus.items()
                        },
                    },
                    out / "focus_metrics.json",
                )
            ),
            "summary_json": str(
                write_summary_json(result.summary, out / "summary.json")
            ),
            "summary_md": str(
                write_summary_markdown(result.summary, out / "summary.md")
            ),
            "analytics_report_md": str(
                write_analytics_report(
                    analysis=analysis,
                    summary=result.summary,
                    path=out / "analytics_report.md",
                )
            ),
        }
        return paths


def load_heatmap_image(path: Path | str) -> np.ndarray:
    """Load a saved heatmap image as a normalized float map in ``[0, 1]``."""

    path = Path(path)
    with Image.open(path) as img:
        if img.mode in {"L", "F", "I"}:
            arr = np.asarray(img.convert("F"), dtype=np.float64)
        else:
            # Coloured CAM PNG → luminance proxy for analytics consumers
            rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
            arr = (
                0.2989 * rgb[..., 0]
                + 0.5870 * rgb[..., 1]
                + 0.1140 * rgb[..., 2]
            )
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return ((arr - lo) / (hi - lo)).astype(np.float64)
