"""Memory usage benchmarks for MAYA inference.

Purpose
-------
Track process RSS before/during/after inference and estimate model footprint.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn

from ai.benchmark.core import CoreInferencer

logger = logging.getLogger("maya.ai.benchmark.memory")


def _rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def estimate_model_memory_mb(model: nn.Module) -> float:
    """Estimate parameter (+ buffer) memory in MiB without allocating extras."""

    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.numel() * buf.element_size()
    return total_bytes / (1024**2)


@dataclass
class MemoryReport:
    """Memory measurements in MiB."""

    baseline_rss_mb: float
    peak_rss_mb: float
    after_rss_mb: float
    delta_rss_mb: float
    estimated_model_mb: float
    samples_mb: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_memory(
    image_path: Path,
    inferencer: CoreInferencer,
    *,
    runs: int = 5,
) -> MemoryReport:
    """Sample RSS around repeated inferences; record peak."""

    loaded = inferencer.ensure_loaded()
    baseline = _rss_mb()
    samples: list[float] = [baseline]
    peak = baseline

    for _ in range(max(1, runs)):
        inferencer.infer_once(image_path)
        current = _rss_mb()
        samples.append(current)
        peak = max(peak, current)

    after = _rss_mb()
    model_mb = estimate_model_memory_mb(loaded.model)
    # Touch torch briefly so unused import stays intentional for CUDA hosts
    _ = torch.cuda.is_available()

    report = MemoryReport(
        baseline_rss_mb=round(baseline, 2),
        peak_rss_mb=round(peak, 2),
        after_rss_mb=round(after, 2),
        delta_rss_mb=round(peak - baseline, 2),
        estimated_model_mb=round(model_mb, 2),
        samples_mb=[round(v, 2) for v in samples],
    )
    logger.info(
        "Memory baseline=%.1f peak=%.1f model≈%.1f MiB",
        report.baseline_rss_mb,
        report.peak_rss_mb,
        report.estimated_model_mb,
    )
    return report
