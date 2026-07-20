"""Logging configuration for MAYA.

Provides structured file + console logging so application code never
relies on print() for operational output.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Configure root application logging once.

    Args:
        log_dir: Directory where rotating log files are stored.
        log_level: Logging level name (e.g. INFO, DEBUG).
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "maya.log"

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Avoid duplicate handlers when create_app is called repeatedly in tests.
        root_logger.setLevel(level)
        return

    root_logger.setLevel(level)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("maya").info("Logging configured at level %s", log_level.upper())
