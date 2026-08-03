"""Separate model prediction confidence from explanation confidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai.explainability.analytics.consistency import ConsistencyMetrics
from ai.explainability.analytics.explanation_quality import ExplanationQuality
from ai.explainability.analytics.focus_metrics import FocusMetrics


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Two distinct confidence notions investigators must not conflate."""

    model_prediction_confidence: float  # 0–100, from classifier
    explanation_confidence: float  # 0–100, from map quality / agreement
    model_confidence_level: str
    explanation_confidence_level: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_confidence_breakdown(
    *,
    model_confidence: float,
    quality: ExplanationQuality,
    focus: FocusMetrics,
    consistency: ConsistencyMetrics | None = None,
) -> ConfidenceBreakdown:
    """Derive explanation confidence independently of model confidence.

    Explanation confidence blends quality score with focus peak and optional
    cross-explainer agreement — never copied from the model softmax.
    """

    model_c = _clamp_pct(model_confidence)
    # Explanation confidence: quality dominates; agreement boosts when present
    expl = 0.70 * quality.quality_score + 0.20 * (100.0 * _clamp01(focus.peak_activation))
    if consistency is not None and len(consistency.explainer_names) >= 2:
        expl = 0.55 * quality.quality_score + 0.25 * (
            100.0 * consistency.agreement_score
        ) + 0.20 * (100.0 * _clamp01(focus.peak_activation))
    expl = round(_clamp_pct(expl), 2)

    return ConfidenceBreakdown(
        model_prediction_confidence=round(model_c, 2),
        explanation_confidence=expl,
        model_confidence_level=_level(model_c),
        explanation_confidence_level=_level(expl),
        notes=(
            "Model confidence reflects class probability. "
            "Explanation confidence reflects map focus/quality"
            + (" and cross-explainer agreement." if consistency and len(consistency.explainer_names) >= 2 else ".")
        ),
    )


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _level(pct: float) -> str:
    if pct >= 85.0:
        return "Very High"
    if pct >= 70.0:
        return "High"
    if pct >= 50.0:
        return "Moderate"
    return "Low"
