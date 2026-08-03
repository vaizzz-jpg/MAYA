"""Automatic explainer ranking from configurable weighted scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai.explainability.benchmark.config import BenchmarkWeights
from ai.explainability.benchmark.utils import clamp01


@dataclass(frozen=True)
class RankingEntry:
    """One explainer's position on the leaderboard."""

    rank: int
    explainer_name: str
    overall_score: float  # 0–100
    quality_score: float
    focus_score: float
    localization_score: float
    coverage: float
    agreement_score: float
    performance_score: float
    generation_time_ms: float
    peak_ram_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingResult:
    """Sorted leaderboard."""

    entries: tuple[RankingEntry, ...]
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(frozen=True)
class RankableExplainer:
    """Minimal inputs required to score an explainer."""

    explainer_name: str
    quality_score: float  # 0–100
    focus_score: float  # 0–100
    localization_score: float  # 0–100
    coverage: float  # 0–1
    agreement_score: float  # 0–1
    generation_time_ms: float
    peak_ram_mb: float


def rank_explainers(
    items: list[RankableExplainer],
    weights: BenchmarkWeights,
    *,
    time_ref_ms: float | None = None,
    ram_ref_mb: float | None = None,
) -> RankingResult:
    """Rank explainers by weighted overall score (no hardcoded favorites)."""

    if not items:
        return RankingResult(entries=(), weights=_weight_dict(weights))

    times = [max(1e-6, i.generation_time_ms) for i in items]
    rams = [max(1e-6, i.peak_ram_mb) for i in items]
    t_ref = float(time_ref_ms) if time_ref_ms is not None else max(times)
    r_ref = float(ram_ref_mb) if ram_ref_mb is not None else max(rams)

    scored: list[RankingEntry] = []
    for item in items:
        perf = _performance_score(
            item.generation_time_ms, item.peak_ram_mb, t_ref, r_ref
        )
        overall = 100.0 * (
            weights.quality * clamp01(item.quality_score / 100.0)
            + weights.focus * clamp01(item.focus_score / 100.0)
            + weights.localization * clamp01(item.localization_score / 100.0)
            + weights.coverage * _coverage_score(item.coverage)
            + weights.agreement * clamp01(item.agreement_score)
            + weights.performance * perf
        )
        scored.append(
            RankingEntry(
                rank=0,
                explainer_name=item.explainer_name,
                overall_score=round(overall, 2),
                quality_score=round(item.quality_score, 2),
                focus_score=round(item.focus_score, 2),
                localization_score=round(item.localization_score, 2),
                coverage=round(item.coverage, 6),
                agreement_score=round(item.agreement_score, 6),
                performance_score=round(100.0 * perf, 2),
                generation_time_ms=round(item.generation_time_ms, 2),
                peak_ram_mb=round(item.peak_ram_mb, 2),
            )
        )

    scored.sort(key=lambda e: (-e.overall_score, e.generation_time_ms, e.explainer_name))
    ranked = tuple(
        RankingEntry(**{**asdict(e), "rank": i + 1}) for i, e in enumerate(scored)
    )
    return RankingResult(entries=ranked, weights=_weight_dict(weights))


def _performance_score(
    time_ms: float,
    ram_mb: float,
    time_ref: float,
    ram_ref: float,
) -> float:
    """Higher is better: relative speed + relative memory frugality."""

    speed = 1.0 - clamp01(time_ms / max(time_ref, 1e-6))
    lean = 1.0 - clamp01(ram_mb / max(ram_ref, 1e-6))
    return clamp01(0.7 * (0.15 + 0.85 * speed) + 0.3 * (0.15 + 0.85 * lean))


def _coverage_score(coverage: float) -> float:
    """Prefer moderate coverage (aligned with Sprint 4.3 focus ideal)."""

    cov = float(coverage)
    if cov <= 0.0:
        return 0.0
    if cov < 0.10:
        return clamp01(cov / 0.10)
    if cov <= 0.40:
        return 1.0
    if cov <= 0.70:
        return clamp01(1.0 - (cov - 0.40) / 0.30)
    return clamp01(1.0 - (cov - 0.70) / 0.30)


def _weight_dict(weights: BenchmarkWeights) -> dict[str, float]:
    return {
        "quality": weights.quality,
        "focus": weights.focus,
        "localization": weights.localization,
        "coverage": weights.coverage,
        "agreement": weights.agreement,
        "performance": weights.performance,
    }
