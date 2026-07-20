"""Centralized logging for MAYA dataset operations (no Flask dependency)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_dataset_logging(
    log_dir: Path,
    log_level: str = "INFO",
    *,
    logger_name: str = "maya.datasets",
) -> logging.Logger:
    """Configure rotating file + console logging for the dataset package.

    Safe to call multiple times; additional handlers are not attached when the
    named logger already has handlers.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    package_logger = logging.getLogger(logger_name)
    package_logger.setLevel(level)

    if package_logger.handlers:
        return package_logger

    file_handler = RotatingFileHandler(
        log_dir / "maya_datasets.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    package_logger.addHandler(file_handler)
    package_logger.addHandler(console_handler)
    package_logger.propagate = False

    package_logger.info("Dataset logging configured at level %s", log_level.upper())
    return package_logger
