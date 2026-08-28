"""Configurable MAYA Explanation Trust Score."""

from __future__ import annotations

import logging
from typing import Any

from ai.explainability.config import TrustScoreWeights
from ai.explainability.fusion.result import ExplanationTrustResult

logger = logging.getLogger("maya.ai.explainability.fusion.trust")

# Alias for public API clarity
TrustWeights = TrustScoreWeights


def _grade(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Moderate-to-High"
    if score >= 40:
        return "Moderate"
    if score >= 25:
        return "Low"
    return "Insufficient"


def compute_trust_score(
    *,
    quality_score: float | None = None,
    faithfulness_score: float | None = None,
    consistency_score: float | None = None,
    localization_score: float | None = None,
    agreement_score: float | None = None,
    weights: TrustScoreWeights | None = None,
) -> ExplanationTrustResult:
    """Compute a weighted engineering trust score from available components.

    Component scales:
    - quality_score: 0–100 (Sprint 4.3)
    - faithfulness_score: 0–1 → scaled to 0–100
    - consistency_score / agreement_score: 0–1 → scaled to 0–100
    - localization_score: 0–100 (from analytics localization component)

    Missing components are omitted and remaining weights are renormalized.
    Values are never invented.
    """

    w = (weights or TrustScoreWeights()).normalized()
    available: dict[str, tuple[float, float]] = {}
    missing: list[str] = []

    def _add(name: str, value: float | None, weight: float, *, scale_100: bool) -> None:
        if value is None:
            missing.append(name)
            return
        score = float(value)
        if not scale_100:
            score = 100.0 * score
        score = max(0.0, min(100.0, score))
        available[name] = (score, weight)

    _add("quality", quality_score, w.quality, scale_100=True)
    _add("faithfulness", faithfulness_score, w.faithfulness, scale_100=False)
    _add("consistency", consistency_score, w.consistency, scale_100=False)
    _add("localization", localization_score, w.localization, scale_100=True)
    _add("agreement", agreement_score, w.agreement, scale_100=False)

    if not available:
        raise ValueError(
            "Cannot compute trust score: no explanation components available"
        )

    weight_sum = sum(wt for _, wt in available.values())
    if weight_sum <= 0:
        raise ValueError("Effective trust weights sum to zero")

    components: dict[str, float] = {}
    weights_used: dict[str, float] = {}
    total = 0.0
    for name, (score, wt) in available.items():
        norm_w = wt / weight_sum
        components[name] = round(score, 4)
        weights_used[name] = round(norm_w, 6)
        total += score * norm_w

    trust = round(float(total), 2)
    result = ExplanationTrustResult(
        trust_score=trust,
        grade=_grade(trust),
        components=components,
        weights_used=weights_used,
        missing_components=missing,
    )
    logger.info(
        "Trust score=%.2f grade=%s missing=%s",
        trust,
        result.grade,
        missing or "none",
    )
    return result


def extract_analytics_components(
    analytics: Any | None,
) -> dict[str, float | None]:
    """Pull quality / localization / consistency / agreement from analytics."""

    if analytics is None:
        return {
            "quality_score": None,
            "localization_score": None,
            "consistency_score": None,
            "agreement_score": None,
        }
    quality = getattr(analytics, "quality", None)
    loc = getattr(analytics, "localization", None)
    consistency = getattr(analytics, "consistency", None)

    quality_score = float(quality.quality_score) if quality is not None else None
    localization_score = (
        float(quality.localization_component) if quality is not None else None
    )
    if localization_score is None and loc is not None:
        # Fallback heuristic from coverage band
        cov = float(loc.coverage_percentage)
        if 5.0 <= cov <= 45.0:
            localization_score = 80.0
        elif cov <= 0.0:
            localization_score = 0.0
        else:
            localization_score = 45.0

    consistency_score = None
    agreement_score = None
    if consistency is not None:
        agreement_score = float(consistency.agreement_score)
        consistency_score = agreement_score

    return {
        "quality_score": quality_score,
        "localization_score": localization_score,
        "consistency_score": consistency_score,
        "agreement_score": agreement_score,
    }
