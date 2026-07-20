"""Benchmark coordinator — runs all modules and packages artefacts.

Purpose
-------
Orchestrate system / model / latency / memory / throughput / consistency
benchmarks, then delegate plotting + reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.benchmark.consistency import ConsistencyReport, measure_consistency
from ai.benchmark.core import CoreInferencer
from ai.benchmark.latency import LatencyReport, measure_latency
from ai.benchmark.memory import MemoryReport, measure_memory
from ai.benchmark.model_stats import ModelStats, collect_model_stats
from ai.benchmark.plots import (
    plot_consistency,
    plot_latency_distribution,
    plot_memory_usage,
    plot_system_summary,
    plot_throughput,
)
from ai.benchmark.report import write_benchmark_reports
from ai.benchmark.system_info import SystemInfo, collect_system_info
from ai.benchmark.throughput import ThroughputReport, measure_throughput
from ai.inference.inference_config import InferenceConfig, get_inference_config

logger = logging.getLogger("maya.ai.benchmark.coordinator")


@dataclass
class BenchmarkPackage:
    """Aggregate result of a full benchmark suite run."""

    artifact_dir: Path
    system: SystemInfo
    model: ModelStats
    latency: LatencyReport
    memory: MemoryReport
    throughput: ThroughputReport
    consistency: ConsistencyReport
    report_paths: dict[str, Path] = field(default_factory=dict)
    plot_paths: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_dir": str(self.artifact_dir),
            "system": self.system.to_dict(),
            "model": self.model.to_dict(),
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "throughput": self.throughput.to_dict(),
            "consistency": self.consistency.to_dict(),
        }


def resolve_sample_image(project_root: Path, explicit: Path | None = None) -> Path:
    """Pick an explicit image or the first processed test REAL jpg."""

    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Sample image not found: {path}")
        return path

    candidates = [
        project_root / "dataset" / "processed" / "test" / "REAL",
        project_root / "dataset" / "processed" / "test" / "FAKE",
        project_root / "dataset" / "raw" / "REAL",
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jpg")):
            return path
    raise FileNotFoundError(
        "No sample image found. Pass --image or build the processed dataset first."
    )


def run_benchmark_suite(
    *,
    image_path: Path | None = None,
    runs: int = 10,
    config: InferenceConfig | None = None,
    artifact_dir: Path | None = None,
) -> BenchmarkPackage:
    """Execute the full Sprint 5 benchmark suite."""

    cfg = config or get_inference_config()
    root = Path(cfg.project_root)
    out = Path(artifact_dir) if artifact_dir else (root / "artifacts" / "phase3" / "benchmark")
    out.mkdir(parents=True, exist_ok=True)

    sample = resolve_sample_image(root, image_path)
    logger.info("Benchmark suite start image=%s runs=%s", sample, runs)

    # Point inference artefacts away from sprint4 pollution during bench
    cfg.artifact_dir = out / "_infer_scratch"
    cfg.id_state_path = out / "_infer_scratch" / "investigation_id_state.json"

    inferencer = CoreInferencer(cfg)
    loaded = inferencer.ensure_loaded()

    system = collect_system_info(cfg.device_preference)
    model = collect_model_stats(loaded, cfg)
    latency = measure_latency(sample, inferencer, runs=runs, warmup=1)
    memory = measure_memory(sample, inferencer, runs=min(5, max(2, runs // 2)))
    throughput = measure_throughput(sample, inferencer)
    consistency = measure_consistency(sample, inferencer, runs=runs)

    plot_paths = {
        "latency_distribution": plot_latency_distribution(
            latency, out / "latency_distribution.png"
        ),
        "memory_usage": plot_memory_usage(memory, out / "memory_usage.png"),
        "throughput": plot_throughput(throughput, out / "throughput.png"),
        "consistency_chart": plot_consistency(consistency, out / "consistency_chart.png"),
        "system_summary": plot_system_summary(system, out / "system_summary.png"),
    }

    package_dict: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_image": str(sample),
        "runs": runs,
        "system": system.to_dict(),
        "model": model.to_dict(),
        "latency": latency.to_dict(),
        "memory": memory.to_dict(),
        "throughput": throughput.to_dict(),
        "consistency": consistency.to_dict(),
    }

    report_paths = write_benchmark_reports(
        out,
        package=package_dict,
        latency=latency.to_dict(),
        memory=memory.to_dict(),
        system=system.to_dict(),
    )

    result = BenchmarkPackage(
        artifact_dir=out,
        system=system,
        model=model,
        latency=latency,
        memory=memory,
        throughput=throughput,
        consistency=consistency,
        report_paths=report_paths,
        plot_paths=plot_paths,
    )
    logger.info("Benchmark suite complete → %s", out)
    return result
