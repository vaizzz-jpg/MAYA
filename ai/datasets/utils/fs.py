"""Shared filesystem helpers for the MAYA dataset pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("maya.datasets.utils.fs")


def is_supported_image(path: Path, extensions: tuple[str, ...]) -> bool:
    """Return True when ``path`` is a file with a supported image extension."""

    try:
        return path.is_file() and path.suffix.lower() in {e.lower() for e in extensions}
    except OSError as exc:
        logger.warning("Cannot inspect %s: %s", path, exc)
        return False


def iter_files(
    root: Path,
    *,
    extensions: tuple[str, ...] | None = None,
) -> Iterator[Path]:
    """Yield files under ``root`` one at a time (optional extension filter)."""

    if not root.exists():
        logger.error("Path does not exist: %s", root)
        return

    ext_set = {e.lower() for e in extensions} if extensions else None
    try:
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if ext_set is not None and path.suffix.lower() not in ext_set:
                    continue
                yield path
            except OSError as exc:
                logger.warning("Skipping path due to OS error: %s (%s)", path, exc)
    except PermissionError as exc:
        logger.error("Permission denied while scanning %s: %s", root, exc)


def count_images(directory: Path, extensions: tuple[str, ...]) -> int:
    """Count image files in a single directory (non-recursive)."""

    if not directory.exists():
        return 0
    count = 0
    try:
        for path in directory.iterdir():
            if is_supported_image(path, extensions):
                count += 1
    except (OSError, PermissionError) as exc:
        logger.warning("Cannot count images in %s: %s", directory, exc)
    return count


def unique_destination(directory: Path, source: Path) -> Path:
    """Return a collision-safe destination path inside ``directory``."""

    candidate = directory / source.name
    if not candidate.exists():
        return candidate

    stem = source.stem
    suffix = source.suffix.lower() or ".jpg"
    index = 1
    while True:
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
