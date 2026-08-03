"""Investigator-friendly explanation summaries (plain language)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai.explainability.analytics.confidence import ConfidenceBreakdown
from ai.explainability.analytics.consistency import ConsistencyMetrics
from ai.explainability.analytics.explanation_quality import ExplanationQuality
from ai.explainability.analytics.focus_metrics import FocusMetrics
from ai.explainability.analytics.localization import LocalizationMetrics


@dataclass(frozen=True)
class InvestigatorSummary:
    """Concise narrative for case review — avoids CAM jargon where possible."""

    headline: str
    focus_summary: str
    localization_summary: str
    consistency_summary: str
    confidence_summary: str
    quality_summary: str
    investigator_guidance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Explanation Summary",
                "",
                f"**{self.headline}**",
                "",
                "## What stood out",
                self.focus_summary,
                "",
                "## Where attention landed",
                self.localization_summary,
                "",
                "## Agreement across methods",
                self.consistency_summary,
                "",
                "## Confidence (model vs explanation)",
                self.confidence_summary,
                "",
                "## Overall quality",
                self.quality_summary,
                "",
                "## Guidance",
                self.investigator_guidance,
                "",
            ]
        )


def build_investigator_summary(
    *,
    explainer_name: str,
    prediction: str,
    focus: FocusMetrics,
    localization: LocalizationMetrics,
    quality: ExplanationQuality,
    confidence: ConfidenceBreakdown,
    consistency: ConsistencyMetrics | None = None,
) -> InvestigatorSummary:
    """Compose plain-language summaries from analytics metrics."""

    focus_txt = _focus_text(focus)
    loc_txt = _localization_text(localization)
    cons_txt = _consistency_text(consistency)
    conf_txt = (
        f"The model’s prediction confidence is "
        f"{confidence.model_prediction_confidence:.1f}% "
        f"({confidence.model_confidence_level}). "
        f"Separately, explanation confidence is "
        f"{confidence.explanation_confidence:.1f}% "
        f"({confidence.explanation_confidence_level}). "
        f"These measure different things and should not be treated as the same."
    )
    quality_txt = (
        f"Overall explanation quality is {quality.quality_score:.1f}/100 "
        f"({quality.grade})."
    )
    headline = (
        f"Explanation for prediction **{prediction}** using `{explainer_name}` "
        f"— quality {quality.grade.lower()} ({quality.quality_score:.0f}/100)."
    )
    guidance = _guidance(quality, confidence, consistency)

    return InvestigatorSummary(
        headline=headline,
        focus_summary=focus_txt,
        localization_summary=loc_txt,
        consistency_summary=cons_txt,
        confidence_summary=conf_txt,
        quality_summary=quality_txt,
        investigator_guidance=guidance,
    )


def _focus_text(focus: FocusMetrics) -> str:
    cov_pct = 100.0 * focus.coverage
    if focus.activation_entropy < 0.45 and focus.peak_activation > 0.5:
        pattern = "The map is relatively focused on a smaller set of regions."
    elif focus.activation_entropy > 0.75:
        pattern = "The map is relatively diffuse across many regions."
    else:
        pattern = "The map shows a moderate spread of highlighted areas."
    return (
        f"{pattern} About {cov_pct:.1f}% of the image exceeds the activation "
        f"threshold ({focus.threshold_used:.2f}). Peak activation is "
        f"{focus.peak_activation:.2f}; mean activation is {focus.mean_activation:.2f}."
    )


def _localization_text(localization: LocalizationMetrics) -> str:
    r0, c0, r1, c1 = localization.bbox
    cy, cx = localization.center_of_attention
    return (
        f"Highlighted coverage is about {localization.coverage_percentage:.1f}% "
        f"of the image. The main attention box spans rows {r0}–{r1} and columns "
        f"{c0}–{c1}. The center of attention is near pixel (row {cy:.1f}, "
        f"col {cx:.1f}). The largest connected highlight covers "
        f"{localization.largest_region_area} pixels "
        f"({100.0 * localization.largest_region_fraction:.2f}% of the map)."
    )


def _consistency_text(consistency: ConsistencyMetrics | None) -> str:
    if consistency is None or len(consistency.explainer_names) < 2:
        return (
            "Only one explanation method was analyzed, so cross-method agreement "
            "was not computed."
        )
    score = 100.0 * consistency.agreement_score
    if score >= 75:
        tone = "The methods largely agree on where to look."
    elif score >= 50:
        tone = "The methods partially agree; review differences carefully."
    else:
        tone = "The methods disagree substantially; treat highlights cautiously."
    return (
        f"{tone} Agreement score is {score:.1f}/100 "
        f"(mean correlation {consistency.mean_pearson:.3f}, "
        f"mean cosine {consistency.mean_cosine:.3f})."
    )


def _guidance(
    quality: ExplanationQuality,
    confidence: ConfidenceBreakdown,
    consistency: ConsistencyMetrics | None,
) -> str:
    tips: list[str] = [
        "Use the overlay next to the original evidence; do not rely on numbers alone.",
        "Explanation confidence is not a legal certainty score.",
    ]
    if quality.grade in {"Low", "Insufficient"}:
        tips.append(
            "Quality is limited — consider another explainer or a closer manual review."
        )
    if (
        abs(
            confidence.model_prediction_confidence
            - confidence.explanation_confidence
        )
        >= 25.0
    ):
        tips.append(
            "Model confidence and explanation confidence diverge — investigate why."
        )
    if consistency is not None and consistency.agreement_score < 0.5:
        tips.append(
            "Low agreement across explainers — avoid over-interpreting any single map."
        )
    return " ".join(tips)
