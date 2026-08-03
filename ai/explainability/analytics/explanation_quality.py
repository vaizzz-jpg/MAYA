"""Overall explanation quality score (interpretable composite)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai.explainability.analytics.consistency import ConsistencyMetrics
from ai.explainability.analytics.focus_metrics import FocusMetrics
from ai.explainability.analytics.localization import LocalizationMetrics


@dataclass(frozen=True)
class ExplanationQuality:
    """Composite quality assessment for investigator review."""

    quality_score: float  # 0–100
    grade: str  # High / Moderate / Low / Insufficient
    focus_component: float
    localization_component: float
    consistency_component: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def compute_explanation_quality(
    focus: FocusMetrics,
    localization: LocalizationMetrics,
    *,
    consistency: ConsistencyMetrics | None = None,
) -> ExplanationQuality:
    """Combine focus, localization, and optional consistency into one score.

    Design (transparent, not a black box):
    - Focus (40%): peak presence + moderate coverage + lower entropy (sharper)
    - Localization (35%): non-empty largest region + sensible coverage band
    - Consistency (25%): cross-explainer agreement when available; else redistribute
    """

    focus_c = _focus_component(focus)
    loc_c = _localization_component(localization)

    if consistency is not None and len(consistency.explainer_names) >= 2:
        cons_c = float(np_clip(consistency.agreement_score))
        w_f, w_l, w_c = 0.40, 0.35, 0.25
    else:
        cons_c = focus_c  # unused weight path
        w_f, w_l, w_c = 0.55, 0.45, 0.0

    score01 = w_f * focus_c + w_l * loc_c + w_c * cons_c
    score = round(100.0 * float(np_clip(score01)), 2)
    return ExplanationQuality(
        quality_score=score,
        grade=_grade(score),
        focus_component=round(100.0 * focus_c, 2),
        localization_component=round(100.0 * loc_c, 2),
        consistency_component=round(100.0 * cons_c, 2) if w_c > 0 else 0.0,
        components={
            "focus_weight": w_f,
            "localization_weight": w_l,
            "consistency_weight": w_c,
        },
    )


def np_clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _focus_component(focus: FocusMetrics) -> float:
    # Prefer maps that light up somewhere (peak) without painting everything.
    peak = np_clip(focus.peak_activation)
    # Ideal coverage around 10–40%
    cov = focus.coverage
    if cov <= 0.0:
        cov_score = 0.0
    elif cov < 0.10:
        cov_score = cov / 0.10
    elif cov <= 0.40:
        cov_score = 1.0
    elif cov <= 0.70:
        cov_score = 1.0 - (cov - 0.40) / 0.30
    else:
        cov_score = max(0.0, 1.0 - (cov - 0.70) / 0.30)
    # Lower entropy → more focused (invert)
    sharpness = 1.0 - np_clip(focus.activation_entropy)
    return np_clip(0.35 * peak + 0.35 * cov_score + 0.30 * sharpness)


def _localization_component(localization: LocalizationMetrics) -> float:
    region = np_clip(localization.largest_region_fraction * 8.0)  # ~12.5% → 1.0
    cov = localization.coverage_percentage / 100.0
    if 5.0 <= localization.coverage_percentage <= 45.0:
        cov_score = 1.0
    elif localization.coverage_percentage < 5.0:
        cov_score = localization.coverage_percentage / 5.0
    else:
        cov_score = max(0.0, 1.0 - (cov - 0.45) / 0.55)
    has_structure = 1.0 if localization.largest_region_area > 0 else 0.0
    return np_clip(0.45 * region + 0.35 * cov_score + 0.20 * has_structure)


def _grade(score: float) -> str:
    if score >= 75.0:
        return "High"
    if score >= 50.0:
        return "Moderate"
    if score >= 25.0:
        return "Low"
    return "Insufficient"
