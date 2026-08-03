"""Pytest suite for MAYA Phase 4 Sprint 4.4 Explainability Benchmark.

Note: Phase 3.5 inference benchmarks remain in ``tests/test_benchmark.py``.
This file covers the explainability validation suite only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.benchmark import (
    ExplainabilityBenchmarkConfig,
    ExplainabilityBenchmarkSuite,
)
from ai.explainability.benchmark.agreement import compute_agreement
from ai.explainability.benchmark.config import BenchmarkWeights
from ai.explainability.benchmark.performance import measure_explainer_performance
from ai.explainability.benchmark.ranking import RankableExplainer, rank_explainers
from ai.explainability.benchmark.recommendation import build_recommendation
from ai.explainability.benchmark.report import write_benchmark_reports
from ai.explainability.benchmark.visualization import write_benchmark_visualizations
import numpy as np


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (128, 128), color=(80, 120, 160)).save(path)
    return path


@pytest.fixture()
def bench_config(tmp_path: Path, sample_image: Path) -> ExplainabilityBenchmarkConfig:
    ckpt = ROOT / "artifacts" / "checkpoints" / "best.pt"
    return ExplainabilityBenchmarkConfig(
        project_root=ROOT,
        checkpoint_path=ckpt if ckpt.exists() else ROOT / "missing.pt",
        artifact_dir=tmp_path / "artifacts" / "phase4" / "sprint4",
        sample_image=sample_image,
        device_preference="cpu",
        # Exclude ScoreCAM for test runtime; covered in full artifact runs.
        explainer_names=("gradcam", "gradcam_plus_plus", "layercam", "eigencam"),
        timed_runs=1,
        warmup_runs=0,
        output_resolution=112,
        scorecam_batch_size=4,
    )


def test_performance_metrics() -> None:
    def work():
        return sum(range(1000))

    result, metrics = measure_explainer_performance(work, timed_runs=2)
    assert result == sum(range(1000))
    assert metrics.generation_time_ms >= 0.0
    assert metrics.peak_ram_mb > 0.0
    assert metrics.runs == 2


def test_agreement_calculations() -> None:
    a = np.zeros((16, 16), dtype=np.float64)
    a[4:8, 4:8] = 1.0
    b = np.roll(a, 1, axis=0)
    agr = compute_agreement({"gradcam": a, "layercam": b})
    assert "gradcam" in agr.matrix
    assert agr.matrix["gradcam"]["gradcam"] == 1.0
    assert 0.0 <= agr.per_explainer_mean_agreement["gradcam"] <= 1.0


def test_ranking_and_recommendation() -> None:
    items = [
        RankableExplainer("slow_cam", 90, 80, 80, 0.2, 0.9, 5000.0, 800.0),
        RankableExplainer("fast_cam", 85, 78, 75, 0.22, 0.85, 100.0, 500.0),
        RankableExplainer("weak_cam", 40, 30, 30, 0.8, 0.2, 200.0, 500.0),
    ]
    ranking = rank_explainers(items, BenchmarkWeights().normalized())
    assert ranking.entries[0].rank == 1
    assert len(ranking.entries) == 3
    # Ranking must be data-driven (not a hardcoded name)
    names = {e.explainer_name for e in ranking.entries}
    assert names == {"slow_cam", "fast_cam", "weak_cam"}
    assert ranking.entries[-1].explainer_name == "weak_cam"

    rec = build_recommendation(ranking)
    assert rec.recommended_explainer == ranking.entries[0].explainer_name
    assert rec.rationale
    assert rec.recommended_explainer in rec.rationale[0]


def test_json_and_markdown_and_visualizations(tmp_path: Path) -> None:
    items = [
        RankableExplainer("gradcam", 70, 65, 60, 0.25, 0.8, 120.0, 400.0),
        RankableExplainer("layercam", 72, 68, 62, 0.28, 0.82, 140.0, 410.0),
    ]
    ranking = rank_explainers(items, BenchmarkWeights().normalized())
    rec = build_recommendation(ranking)
    heat = {"gradcam": np.random.rand(8, 8), "layercam": np.random.rand(8, 8)}
    agr = compute_agreement(heat)
    out = tmp_path / "viz"
    viz = write_benchmark_visualizations(ranking, agr, out)
    for key in (
        "leaderboard",
        "performance_chart",
        "quality_chart",
        "agreement_matrix",
        "memory_usage",
        "comparison_dashboard",
    ):
        assert viz[key].exists()

    payload = {
        "image_name": "x.jpg",
        "prediction": "FAKE",
        "device": "CPU",
        "explainer_names": ["gradcam", "layercam"],
        "statistics": {"mean_quality": 71.0},
        "timestamp": "t",
    }
    reports = write_benchmark_reports(
        benchmark_payload=payload,
        ranking=ranking,
        recommendation=rec,
        artifact_dir=out,
    )
    assert reports["benchmark_json"].exists()
    assert reports["leaderboard_json"].exists()
    assert reports["recommendation_json"].exists()
    assert reports["summary_md"].exists()
    assert reports["benchmark_report_md"].exists()
    data = json.loads(reports["leaderboard_json"].read_text(encoding="utf-8"))
    assert "entries" in data


def test_benchmark_execution(bench_config: ExplainabilityBenchmarkConfig) -> None:
    suite = ExplainabilityBenchmarkSuite(bench_config)
    result = suite.run()
    assert set(result.explainer_names) == set(bench_config.explainer_names)
    assert result.recommendation["recommended_explainer"]
    assert Path(bench_config.artifact_dir, "benchmark.json").exists()
    assert Path(bench_config.artifact_dir, "leaderboard.png").exists()
    assert Path(bench_config.artifact_dir, "recommendation.json").exists()
    assert Path(bench_config.artifact_dir, "summary.md").exists()
    leaderboard = json.loads(
        (Path(bench_config.artifact_dir) / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert len(leaderboard["entries"]) == len(bench_config.explainer_names)
