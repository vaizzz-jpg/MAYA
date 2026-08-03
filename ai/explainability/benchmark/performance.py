"""Performance timing and memory sampling for explainer generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from ai.explainability.benchmark.utils import cpu_percent, rss_mb

logger = logging.getLogger("maya.ai.explainability.benchmark.performance")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Runtime / resource cost of generating one explainer's map."""

    generation_time_ms: float
    peak_ram_mb: float
    average_ram_mb: float
    baseline_ram_mb: float
    cpu_percent: float
    runs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_explainer_performance(
    generate_fn: Callable[[], Any],
    *,
    warmup_runs: int = 0,
    timed_runs: int = 1,
) -> tuple[Any, PerformanceMetrics]:
    """Time ``generate_fn`` and sample RSS around execution.

    Returns the last ``generate_fn`` result plus aggregated metrics.
    """

    for _ in range(max(0, warmup_runs)):
        generate_fn()

    # Prime CPU percent counter
    cpu_percent(0.0)
    baseline = rss_mb()
    samples: list[float] = [baseline]
    times_ms: list[float] = []
    result: Any = None

    for _ in range(max(1, timed_runs)):
        started = time.perf_counter()
        result = generate_fn()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times_ms.append(elapsed_ms)
        current = rss_mb()
        samples.append(current)

    cpu = cpu_percent(0.0)
    peak = max(samples)
    avg = sum(samples) / len(samples)
    metrics = PerformanceMetrics(
        generation_time_ms=round(sum(times_ms) / len(times_ms), 2),
        peak_ram_mb=round(peak, 2),
        average_ram_mb=round(avg, 2),
        baseline_ram_mb=round(baseline, 2),
        cpu_percent=round(cpu, 2),
        runs=len(times_ms),
    )
    logger.info(
        "Performance: %.1f ms | peak RSS %.1f MiB | CPU %.1f%%",
        metrics.generation_time_ms,
        metrics.peak_ram_mb,
        metrics.cpu_percent,
    )
    return result, metrics
