"""Investigator recommendation engine driven by benchmark rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai.explainability.benchmark.ranking import RankingEntry, RankingResult


@dataclass(frozen=True)
class Recommendation:
    """Data-driven explainer recommendation (no hardcoded favorites)."""

    recommended_explainer: str
    overall_score: float
    rank: int
    rationale: list[str]
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_recommendation(ranking: RankingResult) -> Recommendation:
    """Recommend the top-ranked explainer and explain why."""

    if not ranking.entries:
        return Recommendation(
            recommended_explainer="",
            overall_score=0.0,
            rank=0,
            rationale=["No explainers were benchmarked."],
            caveats=["Run the benchmark suite before selecting an explainer."],
        )

    top = ranking.entries[0]
    rationale = _rationale_for(top, ranking)
    alternatives = [
        {
            "explainer": e.explainer_name,
            "rank": e.rank,
            "overall_score": e.overall_score,
            "why_consider": _alt_reason(e, top),
        }
        for e in ranking.entries[1:3]
    ]
    caveats = [
        "Recommendation is relative to this image and current weight configuration.",
        "Always review overlays beside the original evidence.",
        "Model confidence and explanation quality are different signals.",
    ]
    if top.agreement_score < 0.5:
        caveats.append(
            "Cross-explainer agreement is modest — verify highlights manually."
        )
    if top.generation_time_ms > 5000:
        caveats.append(
            "Top explainer is relatively slow on this host — consider a faster alternative for triage."
        )

    return Recommendation(
        recommended_explainer=top.explainer_name,
        overall_score=top.overall_score,
        rank=top.rank,
        rationale=rationale,
        alternatives=alternatives,
        caveats=caveats,
    )


def _rationale_for(top: RankingEntry, ranking: RankingResult) -> list[str]:
    lines = [
        f"`{top.explainer_name}` ranked #{top.rank} with overall score "
        f"{top.overall_score:.1f}/100 under the configured weights.",
        f"Explanation quality {top.quality_score:.1f}/100; focus {top.focus_score:.1f}; "
        f"localization {top.localization_score:.1f}.",
        f"Coverage {100.0 * top.coverage:.1f}%; mean agreement "
        f"{100.0 * top.agreement_score:.1f}%; generation {top.generation_time_ms:.1f} ms; "
        f"peak RAM {top.peak_ram_mb:.1f} MiB.",
    ]
    # Highlight decisive advantages vs runner-up when present
    if len(ranking.entries) >= 2:
        second = ranking.entries[1]
        delta = top.overall_score - second.overall_score
        if delta >= 1.0:
            lines.append(
                f"Leads `{second.explainer_name}` by {delta:.1f} overall points."
            )
        if top.quality_score >= second.quality_score + 5:
            lines.append("Strongest explanation quality among compared methods.")
        if top.generation_time_ms + 50 < second.generation_time_ms and top.rank == 1:
            lines.append("Competitive runtime relative to the next-ranked method.")
    return lines


def _alt_reason(entry: RankingEntry, top: RankingEntry) -> str:
    if entry.generation_time_ms + 100 < top.generation_time_ms:
        return "Faster generation — useful for quick triage."
    if entry.agreement_score > top.agreement_score + 0.05:
        return "Higher average agreement with peer explainers."
    if entry.focus_score > top.focus_score + 5:
        return "Sharper focus metrics on this sample."
    return "Next-best overall score under the same weights."
