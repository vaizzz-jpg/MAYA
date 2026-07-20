"""Configure logging for MAYA training (writes ``logs/training.log``)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_training_logging(
    log_dir: Path,
    log_level: str = "INFO",
) -> logging.Logger:
    """Attach file + console handlers under the ``maya.ai`` logger tree."""

    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("maya.ai")
    root.setLevel(level)

    # Avoid duplicate handlers across CLI re-entries
    marker = "maya_training_file"
    if any(getattr(h, "maya_marker", None) == marker for h in root.handlers):
        return root

    file_handler = RotatingFileHandler(
        log_dir / "training.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler.maya_marker = marker  # type: ignore[attr-defined]

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    console.maya_marker = marker  # type: ignore[attr-defined]

    root.addHandler(file_handler)
    root.addHandler(console)
    root.propagate = False
    root.info("Training logging configured → %s", log_dir / "training.log")
    return root
