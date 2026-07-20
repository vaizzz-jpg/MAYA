"""Shared inference helpers (timers, path validation, logging helpers).

Purpose
-------
Small reusable utilities shared by pipeline / batch / CLI without duplication.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("maya.ai.inference.utils")


@contextmanager
def timing_ms() -> Iterator[dict[str, float]]:
    """Context manager that records elapsed milliseconds in ``store['ms']``."""

    store: dict[str, float] = {"ms": 0.0}
    started = time.perf_counter()
    try:
        yield store
    finally:
        store["ms"] = (time.perf_counter() - started) * 1000.0


def ensure_file(path: Path) -> Path:
    """Validate that ``path`` exists and is a file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    return path


def list_images(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    """List image files in a folder (non-recursive) using supported extensions."""

    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    ext_set = {e.lower() for e in extensions}
    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ext_set
    )
    logger.info("Found %s images in %s", len(images), folder)
    return images


def write_text(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
