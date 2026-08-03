"""Analytics visualizations for explanation heatmaps."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from ai.explainability.analytics.focus_metrics import FocusMetrics
from ai.explainability.analytics.localization import LocalizationMetrics
from ai.explainability.analytics.statistics import ActivationStatistics

logger = logging.getLogger("maya.ai.explainability.analytics.visualization")


def save_activation_histogram(
    heatmap: np.ndarray,
    path: Path,
    *,
    bins: int = 40,
    title: str = "Activation Histogram",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(heatmap, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=bins, color="#3d5a80", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Activation")
    ax.set_ylabel("Pixel count")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved activation histogram → %s", path)
    return path


def save_coverage_map(
    heatmap: np.ndarray,
    localization: LocalizationMetrics,
    path: Path,
    *,
    title: str = "Coverage Map",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(heatmap, dtype=np.float64)
    mask = (arr >= localization.threshold_used).astype(np.float64)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    r0, c0, r1, c1 = localization.bbox
    rect = plt.Rectangle(
        (c0, r0),
        c1 - c0,
        r1 - r0,
        fill=False,
        edgecolor="#e63946",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    cy, cx = localization.center_of_attention
    ax.plot([cx], [cy], marker="+", color="#f4a261", markersize=12, markeredgewidth=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved coverage map → %s", path)
    return path


def save_focus_distribution(
    heatmap: np.ndarray,
    focus: FocusMetrics,
    stats: ActivationStatistics,
    path: Path,
    *,
    title: str = "Focus Distribution",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(heatmap, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im = axes[0].imshow(arr, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("Attention map")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    labels = ["coverage", "peak", "mean", "entropy", "std"]
    values = [
        focus.coverage,
        focus.peak_activation,
        focus.mean_activation,
        focus.activation_entropy,
        min(1.0, stats.std),
    ]
    axes[1].barh(labels, values, color="#2a9d8f")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Focus metrics")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved focus distribution → %s", path)
    return path


def write_analytics_visualizations(
    heatmap: np.ndarray,
    focus: FocusMetrics,
    localization: LocalizationMetrics,
    stats: ActivationStatistics,
    artifact_dir: Path,
) -> dict[str, Path]:
    """Write the Sprint 4.3 visualization trio."""

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "activation_histogram": save_activation_histogram(
            heatmap, out / "activation_histogram.png"
        ),
        "coverage_map": save_coverage_map(
            heatmap, localization, out / "coverage_map.png"
        ),
        "focus_distribution": save_focus_distribution(
            heatmap, focus, stats, out / "focus_distribution.png"
        ),
    }
