"""Small helpers for the explainability benchmark suite."""

from __future__ import annotations

import logging
from pathlib import Path

import psutil

logger = logging.getLogger("maya.ai.explainability.benchmark.utils")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rss_mb() -> float:
    """Current process RSS in MiB."""

    return psutil.Process().memory_info().rss / (1024**2)


def cpu_percent(interval: float = 0.0) -> float:
    """Process CPU percent (non-blocking when interval=0)."""

    return float(psutil.Process().cpu_percent(interval=interval))


def resolve_sample_image(project_root: Path, explicit: Path | None = None) -> Path:
    """Pick an explicit image or the first processed test jpg."""

    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Sample image not found: {path}")
        return path

    root = Path(project_root)
    for folder in (
        root / "dataset" / "processed" / "test" / "FAKE",
        root / "dataset" / "processed" / "test" / "REAL",
        root / "dataset" / "processed" / "validation" / "FAKE",
    ):
        if not folder.exists():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            hits = sorted(folder.glob(pattern))
            if hits:
                logger.info("Using sample image %s", hits[0])
                return hits[0]
    raise FileNotFoundError(
        f"No sample image found under {root / 'dataset' / 'processed'}"
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
