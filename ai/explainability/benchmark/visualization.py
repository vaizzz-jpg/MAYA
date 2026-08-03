"""Benchmark visualizations for explainer comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from ai.explainability.benchmark.agreement import AgreementResult
from ai.explainability.benchmark.ranking import RankingResult
from ai.explainability.benchmark.utils import ensure_dir

logger = logging.getLogger("maya.ai.explainability.benchmark.visualization")


def write_benchmark_visualizations(
    ranking: RankingResult,
    agreement: AgreementResult,
    artifact_dir: Path,
) -> dict[str, Path]:
    out = ensure_dir(artifact_dir)
    paths = {
        "leaderboard": _leaderboard(ranking, out / "leaderboard.png"),
        "performance_chart": _performance(ranking, out / "performance_chart.png"),
        "quality_chart": _quality(ranking, out / "quality_chart.png"),
        "agreement_matrix": _agreement_matrix(
            agreement, out / "agreement_matrix.png"
        ),
        "memory_usage": _memory(ranking, out / "memory_usage.png"),
        "comparison_dashboard": _dashboard(
            ranking, agreement, out / "comparison_dashboard.png"
        ),
    }
    return paths


def _leaderboard(ranking: RankingResult, path: Path) -> Path:
    names = [e.explainer_name for e in ranking.entries][::-1]
    scores = [e.overall_score for e in ranking.entries][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, scores, color="#264653")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall score")
    ax.set_title("Explainer Leaderboard")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _performance(ranking: RankingResult, path: Path) -> Path:
    names = [e.explainer_name for e in ranking.entries]
    times = [e.generation_time_ms for e in ranking.entries]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, times, color="#2a9d8f")
    ax.set_ylabel("Generation time (ms)")
    ax.set_title("Performance")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _quality(ranking: RankingResult, path: Path) -> Path:
    names = [e.explainer_name for e in ranking.entries]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, [e.quality_score for e in ranking.entries], width, label="Quality")
    ax.bar(x, [e.focus_score for e in ranking.entries], width, label="Focus")
    ax.bar(
        x + width,
        [e.localization_score for e in ranking.entries],
        width,
        label="Localization",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _agreement_matrix(agreement: AgreementResult, path: Path) -> Path:
    names = list(agreement.matrix.keys())
    if not names:
        fig, ax = plt.subplots()
        ax.set_title("Agreement Matrix (empty)")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    mat = np.array([[agreement.matrix[a][b] for b in names] for a in names], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Agreement Matrix (cosine)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _memory(ranking: RankingResult, path: Path) -> Path:
    names = [e.explainer_name for e in ranking.entries]
    peaks = [e.peak_ram_mb for e in ranking.entries]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, peaks, color="#e76f51")
    ax.set_ylabel("Peak RSS (MiB)")
    ax.set_title("Memory Usage")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _dashboard(
    ranking: RankingResult,
    agreement: AgreementResult,
    path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    names = [e.explainer_name for e in ranking.entries]

    axes[0, 0].bar(names, [e.overall_score for e in ranking.entries], color="#264653")
    axes[0, 0].set_title("Overall")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].bar(names, [e.generation_time_ms for e in ranking.entries], color="#2a9d8f")
    axes[0, 1].set_title("Time (ms)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    axes[1, 0].bar(names, [e.quality_score for e in ranking.entries], color="#e9c46a")
    axes[1, 0].set_title("Quality")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_ylim(0, 100)

    keys = list(agreement.matrix.keys())
    if keys:
        mat = np.array(
            [[agreement.matrix[a][b] for b in keys] for a in keys], dtype=float
        )
        axes[1, 1].imshow(mat, cmap="viridis", vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(keys)))
        axes[1, 1].set_yticks(range(len(keys)))
        axes[1, 1].set_xticklabels(keys, rotation=25, ha="right", fontsize=8)
        axes[1, 1].set_yticklabels(keys, fontsize=8)
    axes[1, 1].set_title("Agreement")

    fig.suptitle("Explainability Benchmark Dashboard")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved comparison dashboard → %s", path)
    return path
