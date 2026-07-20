"""Latency benchmarks for MAYA inference.

Purpose
-------
Measure average / min / max / median / std of core inference latency over N runs.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai.benchmark.core import CoreInferencer

logger = logging.getLogger("maya.ai.benchmark.latency")


@dataclass
class LatencyReport:
    """Latency statistics in milliseconds."""

    runs: int
    latencies_ms: list[float] = field(default_factory=list)
    average_ms: float = 0.0
    minimum_ms: float = 0.0
    maximum_ms: float = 0.0
    median_ms: float = 0.0
    std_dev_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_latency(
    image_path: Path,
    inferencer: CoreInferencer,
    *,
    runs: int = 10,
    warmup: int = 1,
) -> LatencyReport:
    """Time ``runs`` inferences after optional warmup passes."""

    if runs < 1:
        raise ValueError("runs must be >= 1")

    for _ in range(max(0, warmup)):
        inferencer.infer_once(image_path)

    samples: list[float] = []
    for index in range(runs):
        outcome = inferencer.infer_once(image_path)
        samples.append(outcome.latency_ms)
        logger.debug("Latency run %s/%s → %.2f ms", index + 1, runs, outcome.latency_ms)

    report = LatencyReport(
        runs=runs,
        latencies_ms=samples,
        average_ms=float(statistics.mean(samples)),
        minimum_ms=float(min(samples)),
        maximum_ms=float(max(samples)),
        median_ms=float(statistics.median(samples)),
        std_dev_ms=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    )
    logger.info(
        "Latency avg=%.2f ms med=%.2f ms min=%.2f max=%.2f (n=%s)",
        report.average_ms,
        report.median_ms,
        report.minimum_ms,
        report.maximum_ms,
        runs,
    )
    return report
