"""Aggregate statistics across benchmarked explainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ai.explainability.benchmark.ranking import RankingEntry


@dataclass(frozen=True)
class BenchmarkStatistics:
    """Suite-level descriptive stats."""

    explainer_count: int
    mean_quality: float
    mean_generation_time_ms: float
    mean_peak_ram_mb: float
    mean_agreement: float
    mean_overall_score: float
    fastest_explainer: str
    leanest_explainer: str
    highest_quality_explainer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_ranking(entries: tuple[RankingEntry, ...] | list[RankingEntry]) -> BenchmarkStatistics:
    if not entries:
        return BenchmarkStatistics(
            explainer_count=0,
            mean_quality=0.0,
            mean_generation_time_ms=0.0,
            mean_peak_ram_mb=0.0,
            mean_agreement=0.0,
            mean_overall_score=0.0,
            fastest_explainer="",
            leanest_explainer="",
            highest_quality_explainer="",
        )

    fastest = min(entries, key=lambda e: e.generation_time_ms)
    leanest = min(entries, key=lambda e: e.peak_ram_mb)
    best_q = max(entries, key=lambda e: e.quality_score)
    return BenchmarkStatistics(
        explainer_count=len(entries),
        mean_quality=round(float(np.mean([e.quality_score for e in entries])), 2),
        mean_generation_time_ms=round(
            float(np.mean([e.generation_time_ms for e in entries])), 2
        ),
        mean_peak_ram_mb=round(float(np.mean([e.peak_ram_mb for e in entries])), 2),
        mean_agreement=round(float(np.mean([e.agreement_score for e in entries])), 6),
        mean_overall_score=round(float(np.mean([e.overall_score for e in entries])), 2),
        fastest_explainer=fastest.explainer_name,
        leanest_explainer=leanest.explainer_name,
        highest_quality_explainer=best_q.explainer_name,
    )
