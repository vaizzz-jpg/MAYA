"""Matplotlib visualizations for MAYA benchmark artefacts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai.benchmark.consistency import ConsistencyReport
from ai.benchmark.latency import LatencyReport
from ai.benchmark.memory import MemoryReport
from ai.benchmark.system_info import SystemInfo
from ai.benchmark.throughput import ThroughputReport

logger = logging.getLogger("maya.ai.benchmark.plots")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Wrote plot → %s", path)
    return path


def plot_latency_distribution(report: LatencyReport, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(report.latencies_ms, bins=min(10, max(3, report.runs)), color="#3d8bfd", edgecolor="white")
    ax.axvline(report.average_ms, color="#d9485f", linestyle="--", label=f"avg={report.average_ms:.1f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency Distribution")
    ax.legend()
    return _save(fig, path)


def plot_memory_usage(report: MemoryReport, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(len(report.samples_mb)))
    ax.plot(xs, report.samples_mb, marker="o", color="#3d8bfd", label="RSS MiB")
    ax.axhline(report.peak_rss_mb, color="#d9485f", linestyle="--", label=f"peak={report.peak_rss_mb:.1f}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("RSS (MiB)")
    ax.set_title("Process Memory During Inference")
    ax.legend()
    return _save(fig, path)


def plot_throughput(report: ThroughputReport, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [p.image_count for p in report.points]
    ips = [p.images_per_second for p in report.points]
    ax.bar([str(c) for c in counts], ips, color="#2b8a3e")
    ax.set_xlabel("Image count")
    ax.set_ylabel("Images / second")
    ax.set_title("Throughput by Batch Size")
    return _save(fig, path)


def plot_consistency(report: ConsistencyReport, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].plot(range(1, report.runs + 1), report.confidences, marker="o", color="#3d8bfd")
    axes[0].set_title("Confidence Stability")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("Confidence %")
    labels = sorted(set(report.predictions))
    counts = [report.predictions.count(label) for label in labels]
    axes[1].bar(labels, counts, color="#d9485f")
    axes[1].set_title(f"Predictions ({report.consistency_percentage:.0f}% consistent)")
    axes[1].set_ylabel("Count")
    return _save(fig, path)


def plot_system_summary(info: SystemInfo, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    text = (
        f"OS: {info.operating_system}\n"
        f"CPU: {info.cpu}\n"
        f"RAM: {info.available_ram_gb:.1f} / {info.total_ram_gb:.1f} GB available\n"
        f"Python: {info.python_version}\n"
        f"PyTorch: {info.pytorch_version}\n"
        f"Torchvision: {info.torchvision_version}\n"
        f"Device: {info.active_device}"
    )
    ax.text(0.05, 0.5, text, va="center", family="monospace", fontsize=11)
    ax.set_title("System Summary")
    return _save(fig, path)
