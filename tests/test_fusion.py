"""Tests for MAYA Phase 4 Sprint 4.5 explanation fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.faithfulness.result import FaithfulnessResult
from ai.explainability.fusion.fusion import fuse_explanations
from ai.explainability.shap.result import ShapAttributionStats, ShapResult


def _shap() -> ShapResult:
    return ShapResult(
        investigation_id="INV-1",
        explainer_name="shap",
        model_name="m",
        model_version="v",
        dataset_version="d",
        prediction="FAKE",
        confidence=90.0,
        target_class=1,
        attribution_stats=ShapAttributionStats(
            mean_abs=0.1,
            max_abs=0.5,
            mean_positive=0.2,
            mean_negative=-0.05,
            positive_fraction=0.4,
            negative_fraction=0.2,
        ),
        generation_time_ms=10.0,
        device="CPU",
        visualization_path="",
        timestamp="t",
    )


def _faith(score: float = 0.7) -> FaithfulnessResult:
    return FaithfulnessResult(
        investigation_id="INV-1",
        image_name="x.png",
        prediction="FAKE",
        original_confidence=90.0,
        original_score=0.9,
        target_class=1,
        perturbation_method="mask",
        steps=5,
        deletion_score=0.2,
        insertion_proxy_score=0.3,
        faithfulness_score=score,
        confidence_drop_important=20.0,
        confidence_drop_unimportant=5.0,
        interpretation="Moderate measured influence.",
        timestamp="t",
    )


def test_fusion_combines_available_evidence() -> None:
    fused = fuse_explanations(
        investigation_id="INV-1",
        image_name="x.png",
        prediction="FAKE",
        confidence=92.4,
        shap=_shap(),
        faithfulness=_faith(0.72),
    )
    assert fused.shap is not None
    assert fused.faithfulness is not None
    assert fused.counterfactual is None
    assert fused.trust.trust_score >= 0.0
    assert "SHAP" in fused.summary.shap_contribution or "contribution" in (
        fused.summary.shap_contribution.lower()
    )


def test_fusion_missing_component_handling() -> None:
    fused = fuse_explanations(
        investigation_id="INV-2",
        image_name="y.png",
        prediction="REAL",
        confidence=61.0,
        faithfulness=_faith(0.5),
    )
    assert fused.shap is None
    assert "faithfulness" in fused.trust.components
    assert "quality" in fused.trust.missing_components


def test_fusion_no_incorrect_averaging() -> None:
    """Trust must not invent CAM quality when analytics are absent."""

    fused = fuse_explanations(
        investigation_id="INV-3",
        image_name="z.png",
        prediction="FAKE",
        confidence=80.0,
        faithfulness=_faith(0.9),
    )
    assert "quality" not in fused.trust.components
    assert fused.trust.components["faithfulness"] == pytest.approx(90.0, abs=0.1)
    # Weight for faithfulness should be renormalized to 1.0 when alone
    assert fused.trust.weights_used["faithfulness"] == pytest.approx(1.0, abs=1e-6)
