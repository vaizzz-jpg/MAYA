"""Shared image helpers (Pillow-based, memory-light)."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("maya.datasets.utils.images")


def probe_size(path: Path) -> tuple[int, int] | None:
    """Read width/height without keeping pixel buffers after close."""

    try:
        with Image.open(path) as img:
            return img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.debug("Dimension probe failed for %s: %s", path, exc)
        return None


def open_rgb(path: Path) -> Image.Image:
    """Open an image as RGB, closing the underlying file handle promptly.

    Caller owns the returned image and should discard it after use.
    """

    with Image.open(path) as img:
        return img.convert("RGB")
