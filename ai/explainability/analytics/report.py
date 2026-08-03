"""Persist explanation analytics artefacts (JSON + Markdown)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ai.explainability.analytics.summary import InvestigatorSummary

logger = logging.getLogger("maya.ai.explainability.analytics.report")


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_explanation_analysis_json(
    analysis: dict[str, Any],
    path: Path,
) -> Path:
    return write_json(analysis, path)


def write_focus_metrics_json(focus_payload: dict[str, Any], path: Path) -> Path:
    return write_json(focus_payload, path)


def write_summary_json(summary: InvestigatorSummary, path: Path) -> Path:
    return write_json(summary.to_dict(), path)


def write_summary_markdown(summary: InvestigatorSummary, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.to_markdown(), encoding="utf-8")
    return path


def write_analytics_report(
    *,
    analysis: dict[str, Any],
    summary: InvestigatorSummary,
    path: Path,
) -> Path:
    """Full investigator-facing analytics Markdown report."""

    focus = analysis.get("focus", {})
    localization = analysis.get("localization", {})
    quality = analysis.get("quality", {})
    confidence = analysis.get("confidence", {})
    consistency = analysis.get("consistency")

    lines = [
        "# MAYA Explanation Analytics Report",
        "",
        f"- Primary explainer: `{analysis.get('primary_explainer', '')}`",
        f"- Prediction: **{analysis.get('prediction', '')}**",
        f"- Image: `{analysis.get('image_name', '')}`",
        f"- Timestamp (UTC): `{analysis.get('timestamp', '')}`",
        "",
        "## Quality",
        "",
        f"- Score: **{quality.get('quality_score', 0)} / 100** ({quality.get('grade', '')})",
        f"- Focus component: `{quality.get('focus_component', 0)}`",
        f"- Localization component: `{quality.get('localization_component', 0)}`",
        f"- Consistency component: `{quality.get('consistency_component', 0)}`",
        "",
        "## Focus metrics",
        "",
        f"- Coverage: `{focus.get('coverage', 0)}`",
        f"- Peak activation: `{focus.get('peak_activation', 0)}`",
        f"- Mean activation: `{focus.get('mean_activation', 0)}`",
        f"- Variance: `{focus.get('activation_variance', 0)}`",
        f"- Entropy: `{focus.get('activation_entropy', 0)}`",
        "",
        "## Localization",
        "",
        f"- Coverage %: `{localization.get('coverage_percentage', 0)}`",
        f"- Bounding box: `{localization.get('bbox', [])}`",
        f"- Center of attention: `{localization.get('center_of_attention', [])}`",
        f"- Largest region area: `{localization.get('largest_region_area', 0)}`",
        "",
        "## Confidence",
        "",
        f"- Model prediction confidence: "
        f"`{confidence.get('model_prediction_confidence', 0)}%` "
        f"({confidence.get('model_confidence_level', '')})",
        f"- Explanation confidence: "
        f"`{confidence.get('explanation_confidence', 0)}%` "
        f"({confidence.get('explanation_confidence_level', '')})",
        f"- Notes: {confidence.get('notes', '')}",
        "",
    ]

    if consistency:
        lines.extend(
            [
                "## Consistency",
                "",
                f"- Agreement score: `{consistency.get('agreement_score', 0)}`",
                f"- Mean Pearson: `{consistency.get('mean_pearson', 0)}`",
                f"- Mean cosine: `{consistency.get('mean_cosine', 0)}`",
                f"- Explainers: `{', '.join(consistency.get('explainer_names', []))}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Investigator summary",
            "",
            summary.headline,
            "",
            summary.focus_summary,
            "",
            summary.localization_summary,
            "",
            summary.consistency_summary,
            "",
            summary.confidence_summary,
            "",
            summary.quality_summary,
            "",
            "### Guidance",
            "",
            summary.investigator_guidance,
            "",
            "## Generated files",
            "",
            "- `activation_histogram.png`",
            "- `coverage_map.png`",
            "- `focus_distribution.png`",
            "- `explanation_analysis.json`",
            "- `focus_metrics.json`",
            "- `summary.json`",
            "- `summary.md`",
            "",
        ]
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved analytics report → %s", path)
    return path
