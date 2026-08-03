"""Persist explainability benchmark reports (JSON + Markdown)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ai.explainability.benchmark.recommendation import Recommendation
from ai.explainability.benchmark.ranking import RankingResult
from ai.explainability.benchmark.utils import ensure_dir

logger = logging.getLogger("maya.ai.explainability.benchmark.report")


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_benchmark_reports(
    *,
    benchmark_payload: dict[str, Any],
    ranking: RankingResult,
    recommendation: Recommendation,
    artifact_dir: Path,
) -> dict[str, Path]:
    out = ensure_dir(artifact_dir)
    paths = {
        "benchmark_json": write_json(benchmark_payload, out / "benchmark.json"),
        "leaderboard_json": write_json(ranking.to_dict(), out / "leaderboard.json"),
        "recommendation_json": write_json(
            recommendation.to_dict(), out / "recommendation.json"
        ),
        "summary_md": _write_summary(ranking, recommendation, out / "summary.md"),
        "benchmark_report_md": _write_full_report(
            benchmark_payload, ranking, recommendation, out / "benchmark_report.md"
        ),
    }
    logger.info("Benchmark reports written under %s", out)
    return paths


def _write_summary(
    ranking: RankingResult,
    recommendation: Recommendation,
    path: Path,
) -> Path:
    lines = [
        "# Explainability Benchmark Summary",
        "",
        f"**Recommended explainer:** `{recommendation.recommended_explainer}` "
        f"(score {recommendation.overall_score:.1f}/100)",
        "",
        "## Top ranks",
        "",
    ]
    for e in ranking.entries[:5]:
        lines.append(
            f"{e.rank}. `{e.explainer_name}` — {e.overall_score:.1f} "
            f"(quality {e.quality_score:.1f}, {e.generation_time_ms:.0f} ms)"
        )
    lines.extend(["", "## Why", ""])
    lines.extend(f"- {r}" for r in recommendation.rationale)
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {c}" for c in recommendation.caveats)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_full_report(
    payload: dict[str, Any],
    ranking: RankingResult,
    recommendation: Recommendation,
    path: Path,
) -> Path:
    stats = payload.get("statistics", {})
    lines = [
        "# MAYA Explainability Benchmark Report",
        "",
        f"- Image: `{payload.get('image_name', '')}`",
        f"- Prediction: **{payload.get('prediction', '')}**",
        f"- Device: `{payload.get('device', '')}`",
        f"- Timestamp (UTC): `{payload.get('timestamp', '')}`",
        f"- Explainers: `{', '.join(payload.get('explainer_names', []))}`",
        "",
        "## Recommendation",
        "",
        f"**`{recommendation.recommended_explainer}`** "
        f"(rank #{recommendation.rank}, score {recommendation.overall_score:.1f})",
        "",
    ]
    lines.extend(f"- {r}" for r in recommendation.rationale)
    lines.extend(["", "### Alternatives", ""])
    for alt in recommendation.alternatives:
        lines.append(
            f"- `{alt.get('explainer')}` (rank {alt.get('rank')}, "
            f"score {alt.get('overall_score')}): {alt.get('why_consider')}"
        )
    lines.extend(["", "## Leaderboard", ""])
    for e in ranking.entries:
        lines.append(
            f"{e.rank}. `{e.explainer_name}` — overall **{e.overall_score:.1f}** | "
            f"quality {e.quality_score:.1f} | focus {e.focus_score:.1f} | "
            f"loc {e.localization_score:.1f} | coverage {100 * e.coverage:.1f}% | "
            f"agreement {100 * e.agreement_score:.1f}% | "
            f"{e.generation_time_ms:.1f} ms | {e.peak_ram_mb:.1f} MiB"
        )
    lines.extend(
        [
            "",
            "## Suite statistics",
            "",
            f"- Mean quality: `{stats.get('mean_quality', 0)}`",
            f"- Mean time (ms): `{stats.get('mean_generation_time_ms', 0)}`",
            f"- Mean peak RAM (MiB): `{stats.get('mean_peak_ram_mb', 0)}`",
            f"- Fastest: `{stats.get('fastest_explainer', '')}`",
            f"- Leanest: `{stats.get('leanest_explainer', '')}`",
            f"- Highest quality: `{stats.get('highest_quality_explainer', '')}`",
            "",
            "## Weights",
            "",
        ]
    )
    for k, v in ranking.weights.items():
        lines.append(f"- {k}: `{v:.4f}`")
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `leaderboard.png`, `performance_chart.png`, `quality_chart.png`",
            "- `agreement_matrix.png`, `memory_usage.png`, `comparison_dashboard.png`",
            "- `benchmark.json`, `leaderboard.json`, `recommendation.json`",
            "- `summary.md`, `benchmark_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
