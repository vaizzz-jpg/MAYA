"""Pytest suite for MAYA Phase 4 Sprint 4.3 Explanation Analytics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.analytics import (
    AnalyticsConfig,
    ExplanationAnalyticsEngine,
)
from ai.explainability.analytics.confidence import compute_confidence_breakdown
from ai.explainability.analytics.consistency import compute_consistency_metrics
from ai.explainability.analytics.explanation_quality import compute_explanation_quality
from ai.explainability.analytics.focus_metrics import compute_focus_metrics
from ai.explainability.analytics.localization import compute_localization_metrics
from ai.explainability.analytics.statistics import compute_activation_statistics
from ai.explainability.analytics.summary import build_investigator_summary


def _blob_heatmap(size: int = 64, cy: float = 20, cx: float = 40, sigma: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return (heat / heat.max()).astype(np.float64)


@pytest.fixture()
def focused_map() -> np.ndarray:
    return _blob_heatmap()


@pytest.fixture()
def analytics_config(tmp_path: Path) -> AnalyticsConfig:
    return AnalyticsConfig(
        project_root=ROOT,
        artifact_dir=tmp_path / "artifacts" / "phase4" / "sprint3",
        coverage_threshold=0.2,
    )


def test_coverage_calculations(focused_map: np.ndarray) -> None:
    focus = compute_focus_metrics(focused_map, coverage_threshold=0.2)
    assert 0.0 < focus.coverage < 1.0
    assert focus.peak_activation == pytest.approx(1.0, abs=1e-5)
    assert focus.mean_activation > 0.0
    assert focus.activation_variance >= 0.0
    assert 0.0 <= focus.activation_entropy <= 1.0


def test_localization(focused_map: np.ndarray) -> None:
    loc = compute_localization_metrics(focused_map, activation_threshold=0.2)
    assert loc.coverage_percentage > 0.0
    r0, c0, r1, c1 = loc.bbox
    assert r0 <= r1 and c0 <= c1
    assert loc.largest_region_area > 0
    cy, cx = loc.center_of_attention
    assert 15 < cy < 25
    assert 35 < cx < 45


def test_quality_score_generation(focused_map: np.ndarray) -> None:
    focus = compute_focus_metrics(focused_map)
    loc = compute_localization_metrics(focused_map)
    quality = compute_explanation_quality(focus, loc)
    assert 0.0 <= quality.quality_score <= 100.0
    assert quality.grade in {"High", "Moderate", "Low", "Insufficient"}


def test_consistency_calculations(focused_map: np.ndarray) -> None:
    other = np.roll(focused_map, shift=3, axis=0)
    metrics = compute_consistency_metrics(
        {"gradcam": focused_map, "layercam": other, "eigencam": focused_map}
    )
    assert 0.0 <= metrics.agreement_score <= 1.0
    assert len(metrics.pairwise) == 3
    assert metrics.mean_cosine > 0.5


def test_summary_generation(focused_map: np.ndarray) -> None:
    focus = compute_focus_metrics(focused_map)
    loc = compute_localization_metrics(focused_map)
    quality = compute_explanation_quality(focus, loc)
    conf = compute_confidence_breakdown(
        model_confidence=88.0, quality=quality, focus=focus
    )
    summary = build_investigator_summary(
        explainer_name="gradcam",
        prediction="FAKE",
        focus=focus,
        localization=loc,
        quality=quality,
        confidence=conf,
    )
    assert "FAKE" in summary.headline
    assert "confidence" in summary.confidence_summary.lower()
    md = summary.to_markdown()
    assert "Guidance" in md


def test_model_vs_explanation_confidence_are_separate(focused_map: np.ndarray) -> None:
    focus = compute_focus_metrics(focused_map)
    loc = compute_localization_metrics(focused_map)
    quality = compute_explanation_quality(focus, loc)
    conf = compute_confidence_breakdown(
        model_confidence=95.0, quality=quality, focus=focus
    )
    assert conf.model_prediction_confidence == pytest.approx(95.0)
    assert conf.explanation_confidence != conf.model_prediction_confidence
    assert "Model confidence" in conf.notes or "model" in conf.notes.lower()


def test_report_and_json_and_visualizations(
    focused_map: np.ndarray, analytics_config: AnalyticsConfig
) -> None:
    engine = ExplanationAnalyticsEngine(analytics_config)
    result = engine.analyze(
        {
            "gradcam": focused_map,
            "gradcam_plus_plus": np.clip(focused_map * 0.9 + 0.05, 0, 1),
            "layercam": np.roll(focused_map, 2, axis=1),
        },
        primary_explainer="gradcam",
        prediction="FAKE",
        model_confidence=82.5,
        image_name="evidence.jpg",
        write_artifacts=True,
    )
    out = Path(analytics_config.artifact_dir)
    assert (out / "activation_histogram.png").exists()
    assert (out / "coverage_map.png").exists()
    assert (out / "focus_distribution.png").exists()
    assert (out / "explanation_analysis.json").exists()
    assert (out / "analytics_report.md").exists()
    assert (out / "focus_metrics.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "summary.md").exists()

    payload = json.loads((out / "explanation_analysis.json").read_text(encoding="utf-8"))
    assert payload["primary_explainer"] == "gradcam"
    assert "quality" in payload
    assert result.consistency is not None
    assert result.quality.quality_score >= 0.0


def test_json_serialization(focused_map: np.ndarray) -> None:
    stats = compute_activation_statistics(focused_map)
    data = stats.to_dict()
    assert data["count"] == focused_map.size
    dumped = json.dumps(data)
    assert "median" in dumped


def test_load_heatmap_image_and_analyze(tmp_path: Path) -> None:
    heat = _blob_heatmap(32)
    path = tmp_path / "cam.png"
    Image.fromarray((heat * 255).astype(np.uint8), mode="L").save(path)
    cfg = AnalyticsConfig(artifact_dir=tmp_path / "out", project_root=ROOT)
    engine = ExplanationAnalyticsEngine(cfg)
    result = engine.analyze_from_heatmap_images(
        {"gradcam": path},
        prediction="REAL",
        model_confidence=60.0,
        image_name="cam.png",
    )
    assert result.primary_explainer == "gradcam"
    assert (Path(cfg.artifact_dir) / "summary.md").exists()
