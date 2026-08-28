"""Explanation fusion — preserve meaning of each measurement."""

from __future__ import annotations

import logging
from typing import Any

from ai.explainability.config import TrustScoreWeights
from ai.explainability.counterfactual.result import CounterfactualResult
from ai.explainability.faithfulness.result import FaithfulnessResult
from ai.explainability.fusion.result import (
    AdvancedInvestigatorSummary,
    ExplanationTrustResult,
    FusedExplanationResult,
)
from ai.explainability.fusion.trust import (
    compute_trust_score,
    extract_analytics_components,
)
from ai.explainability.shap.result import ShapResult

logger = logging.getLogger("maya.ai.explainability.fusion.fusion")


def _visual_evidence_text(analytics: Any | None, cam_name: str) -> str:
    if analytics is None:
        return (
            f"Primary visual evidence from `{cam_name}` was not supplied for fusion. "
            "CAM activation maps indicate localization of model activation, "
            "not proven causation."
        )
    focus = getattr(analytics, "focus", None)
    quality = getattr(analytics, "quality", None)
    loc = getattr(analytics, "localization", None)
    parts = [
        f"Primary visual evidence uses `{cam_name}` activation localization."
    ]
    if quality is not None:
        parts.append(
            f"Explanation quality is {quality.quality_score:.1f}/100 ({quality.grade})."
        )
    if focus is not None:
        parts.append(
            f"About {100.0 * focus.coverage:.1f}% of the map exceeds the "
            f"activation threshold; peak activation is {focus.peak_activation:.2f}."
        )
    if loc is not None:
        cy, cx = loc.center_of_attention
        parts.append(
            f"Center of activation is near (row {cy:.1f}, col {cx:.1f}) with "
            f"coverage {loc.coverage_percentage:.1f}%."
        )
    parts.append(
        "These describe model activation/localization, not legal certainty."
    )
    return " ".join(parts)


def _shap_text(shap: ShapResult | None) -> str:
    if shap is None:
        return "SHAP attribution was not available for this run."
    s = shap.attribution_stats
    tone = (
        "Strong positive contribution concentrated in highlighted regions."
        if s.mean_positive > abs(s.mean_negative) and s.positive_fraction > 0.15
        else "Mixed positive/negative contribution across the image."
    )
    return (
        f"{tone} Mean |SHAP|={s.mean_abs:.4f}; "
        f"positive fraction={100.0 * s.positive_fraction:.1f}%, "
        f"negative fraction={100.0 * s.negative_fraction:.1f}%. "
        "SHAP provides attribution estimates, not Grad-CAM attention."
    )


def _faithfulness_text(faith: FaithfulnessResult | None) -> str:
    if faith is None:
        return "Faithfulness testing was not available for this run."
    return faith.interpretation


def _counterfactual_text(cf: CounterfactualResult | None) -> str:
    if cf is None:
        return "Counterfactual perturbation experiments were not available."
    return cf.interpretation


def _assessment_text(trust: ExplanationTrustResult) -> str:
    return (
        f"MAYA Explanation Trust Score is {trust.trust_score:.1f}/100 "
        f"({trust.grade}). Components used: "
        + ", ".join(
            f"{k}={v:.1f} (w={trust.weights_used[k]:.2f})"
            for k, v in trust.components.items()
        )
        + (
            f". Missing: {', '.join(trust.missing_components)}."
            if trust.missing_components
            else "."
        )
        + " This is an engineering summary of measured characteristics, "
        "not a probability that the explanation is true."
    )


def build_advanced_summary(
    *,
    prediction: str,
    confidence: float,
    cam_name: str,
    analytics: Any | None,
    shap: ShapResult | None,
    faithfulness: FaithfulnessResult | None,
    counterfactual: CounterfactualResult | None,
    trust: ExplanationTrustResult,
) -> AdvancedInvestigatorSummary:
    return AdvancedInvestigatorSummary(
        prediction=prediction,
        model_confidence=confidence,
        primary_visual_evidence=_visual_evidence_text(analytics, cam_name),
        shap_contribution=_shap_text(shap),
        faithfulness=_faithfulness_text(faithfulness),
        counterfactual_sensitivity=_counterfactual_text(counterfactual),
        explanation_assessment=_assessment_text(trust),
        caveats=(
            "CAM heatmaps indicate activation/localization. "
            "SHAP provides attribution estimates. "
            "Faithfulness measures sensitivity to perturbations. "
            "Counterfactual experiments measure prediction changes under "
            "controlled edits. None of these independently prove real-world "
            "causation. Prefer: the model's prediction was strongly associated "
            "with highlighted regions — avoid claims of definite manipulation."
        ),
    )


def fuse_explanations(
    *,
    investigation_id: str,
    image_name: str,
    prediction: str,
    confidence: float,
    cam_name: str = "gradcam",
    analytics: Any | None = None,
    shap: ShapResult | None = None,
    faithfulness: FaithfulnessResult | None = None,
    counterfactual: CounterfactualResult | None = None,
    trust_weights: TrustScoreWeights | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> FusedExplanationResult:
    """Combine available evidence without averaging unrelated scores blindly.

    Each modality remains a distinct field. The trust score uses only
    available, documented components with renormalized weights.
    """

    analytics_bits = extract_analytics_components(analytics)
    faith_score = (
        float(faithfulness.faithfulness_score) if faithfulness is not None else None
    )

    trust = compute_trust_score(
        quality_score=analytics_bits["quality_score"],
        faithfulness_score=faith_score,
        consistency_score=analytics_bits["consistency_score"],
        localization_score=analytics_bits["localization_score"],
        agreement_score=analytics_bits["agreement_score"],
        weights=trust_weights,
    )

    summary = build_advanced_summary(
        prediction=prediction,
        confidence=confidence,
        cam_name=cam_name,
        analytics=analytics,
        shap=shap,
        faithfulness=faithfulness,
        counterfactual=counterfactual,
        trust=trust,
    )

    visual: dict[str, Any] = {"primary_cam": cam_name}
    if analytics is not None and hasattr(analytics, "to_dict"):
        visual["analytics"] = {
            "quality": analytics.quality.to_dict()
            if getattr(analytics, "quality", None)
            else None,
            "focus": analytics.focus.to_dict()
            if getattr(analytics, "focus", None)
            else None,
            "localization": analytics.localization.to_dict()
            if getattr(analytics, "localization", None)
            else None,
        }

    fused = FusedExplanationResult(
        investigation_id=investigation_id,
        image_name=image_name,
        prediction=prediction,
        confidence=confidence,
        trust=trust,
        summary=summary,
        visual_evidence=visual,
        shap=shap.to_dict() if shap is not None else None,
        faithfulness=faithfulness.to_dict() if faithfulness is not None else None,
        counterfactual=counterfactual.to_dict()
        if counterfactual is not None
        else None,
        artifact_paths=dict(artifact_paths or {}),
        timestamp=FusedExplanationResult.utc_now(),
    )
    logger.info(
        "Fused explanation id=%s trust=%.1f grade=%s",
        investigation_id,
        trust.trust_score,
        trust.grade,
    )
    return fused
