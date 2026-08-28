"""Markdown / JSON report writers for Sprint 4.5 artefacts."""

from __future__ import annotations

import logging
from pathlib import Path

from ai.explainability.counterfactual.result import CounterfactualResult
from ai.explainability.faithfulness.result import FaithfulnessResult
from ai.explainability.fusion.audit import XaiAuditRecord
from ai.explainability.fusion.result import FusedExplanationResult
from ai.explainability.shap.result import ShapResult

logger = logging.getLogger("maya.ai.explainability.fusion.report")


def _write(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_shap_report(result: ShapResult, path: Path) -> Path:
    s = result.attribution_stats
    lines = [
        "# SHAP Attribution Report",
        "",
        f"- Investigation ID: `{result.investigation_id}`",
        f"- Image: `{result.image_name}`",
        f"- Prediction: **{result.prediction}** ({result.confidence:.2f}%)",
        f"- Target class: `{result.target_class}`",
        f"- Model: `{result.model_name}` / `{result.model_version}`",
        f"- Dataset: `{result.dataset_version}`",
        f"- Device: `{result.device}`",
        f"- Max evaluations: `{result.max_evaluations}`",
        f"- Generation time: `{result.generation_time_ms:.2f} ms`",
        f"- Timestamp: `{result.timestamp}`",
        "",
        "## Attribution statistics (measured)",
        "",
        f"- Mean |SHAP|: `{s.mean_abs}`",
        f"- Max |SHAP|: `{s.max_abs}`",
        f"- Mean positive: `{s.mean_positive}`",
        f"- Mean negative: `{s.mean_negative}`",
        f"- Positive fraction: `{100.0 * s.positive_fraction:.2f}%`",
        f"- Negative fraction: `{100.0 * s.negative_fraction:.2f}%`",
        "",
        "## Interpretation",
        "",
        "SHAP estimates **feature contribution** to the predicted class score.",
        "This is **not** a Grad-CAM attention / activation map.",
        "Attribution estimates do not independently prove real-world causation.",
        "",
        "## Artefacts",
        "",
        f"- Heatmap: `{result.heatmap_path}`",
        f"- Overlay: `{result.overlay_path}`",
        f"- Comparison: `{result.comparison_path}`",
        "",
    ]
    path = _write(path, "\n".join(lines))
    logger.info("Wrote SHAP report → %s", path)
    return path


def write_faithfulness_report(result: FaithfulnessResult, path: Path) -> Path:
    lines = [
        "# Faithfulness Evaluation Report",
        "",
        f"- Investigation ID: `{result.investigation_id}`",
        f"- Image: `{result.image_name}`",
        f"- Prediction: **{result.prediction}**",
        f"- Original confidence: `{result.original_confidence:.2f}%`",
        f"- Original target-class score: `{result.original_score}`",
        f"- Perturbation method: `{result.perturbation_method}`",
        f"- Steps: `{result.steps}`",
        f"- Deletion score: `{result.deletion_score}`",
        f"- Insertion proxy score: `{result.insertion_proxy_score}`",
        f"- Faithfulness score: `{result.faithfulness_score}`",
        f"- Mean confidence drop (important): `{result.confidence_drop_important}` pp",
        f"- Mean confidence drop (less-important): "
        f"`{result.confidence_drop_unimportant}` pp",
        f"- Generation time: `{result.generation_time_ms:.2f} ms`",
        "",
        "## Step measurements",
        "",
        "| Step | Fraction | Important score | Unimportant score | Imp drop | Unimp drop |",
        "|------|----------|-----------------|-------------------|----------|------------|",
    ]
    for step in result.step_results:
        lines.append(
            f"| {step.step} | {step.fraction_masked:.3f} | "
            f"{step.important_score:.4f} | {step.unimportant_score:.4f} | "
            f"{step.important_drop:.4f} | {step.unimportant_drop:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result.interpretation,
            "",
            f"Visualization: `{result.visualization_path}`",
            "",
        ]
    )
    path = _write(path, "\n".join(lines))
    logger.info("Wrote faithfulness report → %s", path)
    return path


def write_counterfactual_report(result: CounterfactualResult, path: Path) -> Path:
    lines = [
        "# Counterfactual Perturbation Report",
        "",
        f"- Investigation ID: `{result.investigation_id}`",
        f"- Image: `{result.image_name}`",
        f"- Original: **{result.original_prediction}** "
        f"({result.original_confidence:.2f}%)",
        f"- Target class: `{result.target_class}`",
        f"- Generation time: `{result.generation_time_ms:.2f} ms`",
        "",
        "## Experiments (measured)",
        "",
    ]
    for exp in result.experiments:
        lines.extend(
            [
                f"### Strategy: `{exp.perturbation_strategy}`",
                "",
                f"- Affected region: `{exp.affected_region}`",
                f"- Original: {exp.original_prediction} {exp.original_confidence:.2f}%",
                f"- Perturbed: {exp.perturbed_prediction} "
                f"{exp.perturbed_confidence:.2f}%",
                f"- Confidence change: `{exp.confidence_delta:+.2f}` percentage points",
                f"- Prediction changed: `{exp.prediction_changed}`",
                f"- Target-class score: `{exp.original_score}` → "
                f"`{exp.perturbed_score}` (Δ `{exp.score_delta:+.4f}`)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            result.interpretation,
            "",
            "These are **model counterfactual perturbation experiments**, not "
            "generative real-world counterfactuals.",
            "",
            f"Visualization: `{result.visualization_path}`",
            "",
        ]
    )
    path = _write(path, "\n".join(lines))
    logger.info("Wrote counterfactual report → %s", path)
    return path


def write_trust_report(fused: FusedExplanationResult, path: Path) -> Path:
    t = fused.trust
    lines = [
        "# Explanation Trust Report",
        "",
        f"- Investigation ID: `{fused.investigation_id}`",
        f"- Image: `{fused.image_name}`",
        f"- Prediction: **{fused.prediction}** ({fused.confidence:.2f}%)",
        f"- MAYA Explanation Trust Score: **{t.trust_score:.2f}/100** ({t.grade})",
        "",
        "## Components (measured)",
        "",
        "| Component | Score (0–100) | Weight used |",
        "|-----------|---------------|-------------|",
    ]
    for name, score in t.components.items():
        lines.append(
            f"| {name} | {score:.2f} | {t.weights_used.get(name, 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            f"Missing components: "
            f"{', '.join(t.missing_components) if t.missing_components else 'none'}",
            "",
            "## Methodology",
            "",
            t.methodology_note,
            "",
            "## Investigator summary",
            "",
            fused.summary.to_markdown(),
        ]
    )
    path = _write(path, "\n".join(lines))
    logger.info("Wrote trust report → %s", path)
    return path


def write_advanced_summary(
    fused: FusedExplanationResult,
    path: Path,
) -> Path:
    path = _write(path, fused.summary.to_markdown())
    logger.info("Wrote advanced XAI summary → %s", path)
    return path


def write_advanced_result_json(
    fused: FusedExplanationResult,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fused.to_json(), encoding="utf-8")
    logger.info("Wrote advanced_xai_result.json → %s", path)
    return path


def write_sprint5_reports(
    *,
    reports_dir: Path,
    shap: ShapResult | None,
    faithfulness: FaithfulnessResult | None,
    counterfactual: CounterfactualResult | None,
    fused: FusedExplanationResult,
    audit: XaiAuditRecord,
) -> dict[str, str]:
    """Write all Sprint 4.5 report artefacts that have measured data."""

    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if shap is not None:
        paths["shap_report"] = str(write_shap_report(shap, out / "shap_report.md"))
        shap.save_json(out / "shap_result.json")
        paths["shap_result"] = str(out / "shap_result.json")
    if faithfulness is not None:
        paths["faithfulness_report"] = str(
            write_faithfulness_report(faithfulness, out / "faithfulness_report.md")
        )
        faithfulness.save_json(out / "faithfulness_result.json")
        paths["faithfulness_result"] = str(out / "faithfulness_result.json")
    if counterfactual is not None:
        paths["counterfactual_report"] = str(
            write_counterfactual_report(
                counterfactual, out / "counterfactual_report.md"
            )
        )
        counterfactual.save_json(out / "counterfactual_result.json")
        paths["counterfactual_result"] = str(out / "counterfactual_result.json")

    paths["explanation_trust_report"] = str(
        write_trust_report(fused, out / "explanation_trust_report.md")
    )
    paths["advanced_xai_summary"] = str(
        write_advanced_summary(fused, out / "advanced_xai_summary.md")
    )
    paths["advanced_xai_result"] = str(
        write_advanced_result_json(fused, out / "advanced_xai_result.json")
    )
    paths["xai_audit"] = str(audit.save_json(out / "xai_audit.json"))
    return paths
