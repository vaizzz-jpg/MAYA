"""Tests for MAYA Phase 4 Sprint 4.5 trust scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.config import TrustScoreWeights
from ai.explainability.fusion.trust import compute_trust_score


def test_configurable_weights() -> None:
    a = compute_trust_score(
        quality_score=100.0,
        faithfulness_score=0.0,
        weights=TrustScoreWeights(
            quality=1.0,
            faithfulness=0.0,
            consistency=0.0,
            localization=0.0,
            agreement=0.0,
        ),
    )
    b = compute_trust_score(
        quality_score=100.0,
        faithfulness_score=0.0,
        weights=TrustScoreWeights(
            quality=0.0,
            faithfulness=1.0,
            consistency=0.0,
            localization=0.0,
            agreement=0.0,
        ),
    )
    assert a.trust_score == pytest.approx(100.0)
    assert b.trust_score == pytest.approx(0.0)


def test_score_calculation_and_bounds() -> None:
    result = compute_trust_score(
        quality_score=80.0,
        faithfulness_score=0.6,
        consistency_score=0.5,
        localization_score=70.0,
        agreement_score=0.4,
    )
    assert 0.0 <= result.trust_score <= 100.0
    assert set(result.components) == {
        "quality",
        "faithfulness",
        "consistency",
        "localization",
        "agreement",
    }
    assert abs(sum(result.weights_used.values()) - 1.0) < 1e-6


def test_grade_generation() -> None:
    high = compute_trust_score(quality_score=95.0, faithfulness_score=0.95)
    low = compute_trust_score(quality_score=10.0, faithfulness_score=0.1)
    assert high.grade in {"High", "Moderate-to-High"}
    assert low.grade in {"Low", "Insufficient"}


def test_trust_requires_at_least_one_component() -> None:
    with pytest.raises(ValueError, match="no explanation components"):
        compute_trust_score()
