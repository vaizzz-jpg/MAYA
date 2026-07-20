"""Pytest suite for MAYA Phase 3 Sprint 5 benchmark modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.benchmark.benchmark import run_benchmark_suite
from ai.benchmark.consistency import measure_consistency
from ai.benchmark.core import CoreInferencer
from ai.benchmark.latency import measure_latency
from ai.benchmark.memory import estimate_model_memory_mb, measure_memory
from ai.benchmark.report import write_benchmark_reports
from ai.benchmark.system_info import collect_system_info
from ai.benchmark.throughput import measure_throughput
from ai.inference.inference_config import InferenceConfig


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "bench.jpg"
    Image.new("RGB", (96, 96), color=(30, 80, 120)).save(path)
    return path


@pytest.fixture()
def infer_config(tmp_path: Path) -> InferenceConfig:
    ckpt = ROOT / "artifacts" / "checkpoints" / "best.pt"
    return InferenceConfig(
        project_root=ROOT,
        checkpoint_path=ckpt if ckpt.exists() else ROOT / "missing.pt",
        artifact_dir=tmp_path / "scratch",
        id_state_path=tmp_path / "scratch" / "ids.json",
        device_preference="cpu",
        image_size=224,
    )


@pytest.fixture()
def inferencer(infer_config: InferenceConfig) -> CoreInferencer:
    return CoreInferencer(infer_config)


def test_system_information_collection() -> None:
    info = collect_system_info("cpu")
    assert info.operating_system
    assert info.total_ram_gb > 0
    assert info.python_version
    assert info.pytorch_version
    assert "cpu" in info.active_device.lower() or info.active_device


def test_latency_calculations(inferencer: CoreInferencer, sample_image: Path) -> None:
    report = measure_latency(sample_image, inferencer, runs=3, warmup=1)
    assert report.runs == 3
    assert len(report.latencies_ms) == 3
    assert report.minimum_ms <= report.average_ms <= report.maximum_ms
    assert report.median_ms > 0


def test_memory_measurements(inferencer: CoreInferencer, sample_image: Path) -> None:
    report = measure_memory(sample_image, inferencer, runs=2)
    assert report.baseline_rss_mb > 0
    assert report.peak_rss_mb >= report.baseline_rss_mb
    loaded = inferencer.ensure_loaded()
    assert estimate_model_memory_mb(loaded.model) > 0


def test_throughput_calculations(inferencer: CoreInferencer, sample_image: Path) -> None:
    report = measure_throughput(sample_image, inferencer, batch_sizes=(1, 2))
    assert len(report.points) == 2
    assert report.points[0].images_per_second > 0


def test_consistency_calculations(inferencer: CoreInferencer, sample_image: Path) -> None:
    report = measure_consistency(sample_image, inferencer, runs=3)
    assert report.runs == 3
    assert 0 <= report.consistency_percentage <= 100
    assert report.dominant_prediction in {"REAL", "FAKE"}
    assert 0 <= report.reliability_score <= 100


def test_report_generation_and_json(tmp_path: Path) -> None:
    package = {
        "system": {"active_device": "cpu"},
        "model": {"model_name": "efficientnet_b0", "model_version": "t"},
        "latency": {"runs": 2, "average_ms": 10.0},
        "memory": {"peak_rss_mb": 100.0},
        "throughput": {"points": [{"image_count": 1, "images_per_second": 2.0, "elapsed_seconds": 0.5}]},
        "consistency": {"consistency_percentage": 100.0, "reliability_score": 95.0},
    }
    paths = write_benchmark_reports(
        tmp_path,
        package=package,
        latency=package["latency"],
        memory=package["memory"],
        system={"active_device": "cpu", "operating_system": "Windows", "total_ram_gb": 8},
    )
    assert paths["benchmark_json"].exists()
    assert paths["benchmark_report_md"].exists()
    data = json.loads(paths["benchmark_json"].read_text(encoding="utf-8"))
    assert "latency" in data


def test_full_suite_smoke(tmp_path: Path, sample_image: Path, infer_config: InferenceConfig) -> None:
    out = tmp_path / "artifacts" / "phase3" / "benchmark"
    package = run_benchmark_suite(
        image_path=sample_image,
        runs=2,
        config=infer_config,
        artifact_dir=out,
    )
    assert package.latency.runs == 2
    assert (out / "benchmark.json").exists()
    assert (out / "latency_distribution.png").exists()
    assert (out / "memory_usage.png").exists()
    assert (out / "throughput.png").exists()
    assert (out / "consistency_chart.png").exists()
    assert (out / "system_summary.png").exists()
    assert (out / "benchmark_report.md").exists()
    assert (out / "performance_summary.md").exists()
