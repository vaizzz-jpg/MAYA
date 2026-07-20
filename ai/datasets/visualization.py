"""Offline visualization helpers for dataset QA reports."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from ai.datasets.dataset_config import ClassName, DatasetConfig
from ai.datasets.inventory import ImageRecord
from ai.datasets.statistics import DatasetStatistics

logger = logging.getLogger("maya.datasets.visualization")


def save_class_distribution(
    stats: DatasetStatistics,
    output_path: Path,
) -> Path:
    """Save a class-count bar chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(stats.class_counts.keys())
    values = [stats.class_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#3d8bfd", "#d9485f"][: len(labels)])
    ax.set_title("MAYA Dataset — Class Distribution (readable images)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    for idx, value in enumerate(values):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote class distribution plot → %s", output_path)
    return output_path


def save_sample_grid(
    readable_records: list[ImageRecord],
    config: DatasetConfig,
    output_path: Path,
) -> Path:
    """Save a small REAL/FAKE thumbnail grid for qualitative QA."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.random_seed)
    per_class = config.sample_grid_per_class

    selected: list[tuple[str, Path]] = []
    for cls in ClassName:
        pool = [r.path for r in readable_records if r.class_name == cls]
        if not pool:
            continue
        take = min(per_class, len(pool))
        selected.extend((cls.value, path) for path in rng.sample(pool, take))

    if not selected:
        logger.warning("No samples available for grid visualization")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No readable samples", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    cols = min(8, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax in axes_list:
        ax.axis("off")

    for ax, (label, path) in zip(axes_list, selected):
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                ax.imshow(rgb)
                ax.set_title(label, fontsize=8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            ax.set_title("error", fontsize=8)
            logger.debug("Grid sample failed for %s: %s", path, exc)
        ax.axis("off")

    fig.suptitle("MAYA Dataset — Sample Grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote sample grid → %s", output_path)
    return output_path
