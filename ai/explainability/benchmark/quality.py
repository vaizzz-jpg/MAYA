"""Quality-side metrics for the benchmark — delegates to Sprint 4.3 analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ai.explainability.analytics.explanation_quality import (
    ExplanationQuality,
    compute_explanation_quality,
)
from ai.explainability.analytics.focus_metrics import FocusMetrics, compute_focus_metrics
from ai.explainability.analytics.localization import (
    LocalizationMetrics,
    compute_localization_metrics,
)


@dataclass(frozen=True)
class QualityBundle:
    """Focus / localization / coverage / quality for one heatmap."""

    focus: FocusMetrics
    localization: LocalizationMetrics
    quality: ExplanationQuality
    coverage: float
    focus_score: float  # 0–100 mirror of quality.focus_component
    localization_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus.to_dict(),
            "localization": self.localization.to_dict(),
            "quality": self.quality.to_dict(),
            "coverage": self.coverage,
            "focus_score": self.focus_score,
            "localization_score": self.localization_score,
        }


def evaluate_heatmap_quality(
    heatmap: np.ndarray,
    *,
    coverage_threshold: float = 0.2,
) -> QualityBundle:
    """Compute quality metrics via Sprint 4.3 modules (no duplicated formulas)."""

    focus = compute_focus_metrics(heatmap, coverage_threshold=coverage_threshold)
    localization = compute_localization_metrics(
        heatmap, activation_threshold=coverage_threshold
    )
    # Per-explainer quality without consistency (agreement applied at suite level)
    quality = compute_explanation_quality(focus, localization, consistency=None)
    return QualityBundle(
        focus=focus,
        localization=localization,
        quality=quality,
        coverage=focus.coverage,
        focus_score=quality.focus_component,
        localization_score=quality.localization_component,
    )
