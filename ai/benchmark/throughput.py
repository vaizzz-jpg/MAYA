"""Throughput benchmarks (images per second).

Purpose
-------
Measure end-to-end core inference throughput for 1 / 5 / 10 / 20 image batches.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai.benchmark.core import CoreInferencer

logger = logging.getLogger("maya.ai.benchmark.throughput")

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 5, 10, 20)


@dataclass
class ThroughputPoint:
    """One batch-size measurement."""

    image_count: int
    elapsed_seconds: float
    images_per_second: float


@dataclass
class ThroughputReport:
    """Throughput across configured batch sizes."""

    points: list[ThroughputPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"points": [asdict(p) for p in self.points]}


def measure_throughput(
    image_path: Path,
    inferencer: CoreInferencer,
    *,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
) -> ThroughputReport:
    """Repeatedly infer the same image ``n`` times for each batch size.

    Uses one physical file to stay memory-light on 8 GB hosts while still
    exercising the full preprocess → predict path ``n`` times.
    """

    inferencer.ensure_loaded()
    points: list[ThroughputPoint] = []

    for count in batch_sizes:
        started = time.perf_counter()
        for _ in range(count):
            inferencer.infer_once(image_path)
        elapsed = time.perf_counter() - started
        ips = float(count / elapsed) if elapsed > 0 else 0.0
        point = ThroughputPoint(
            image_count=count,
            elapsed_seconds=round(elapsed, 4),
            images_per_second=round(ips, 4),
        )
        points.append(point)
        logger.info(
            "Throughput n=%s → %.3f img/s (%.3fs)",
            count,
            point.images_per_second,
            point.elapsed_seconds,
        )

    return ThroughputReport(points=points)
