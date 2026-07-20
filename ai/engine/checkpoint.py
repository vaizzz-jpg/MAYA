"""Checkpoint save/load with resume support for MAYA training.

Why this module exists
----------------------
Persist model + optimizer + scheduler + epoch + metrics + config so runs
can resume after interruption on long CPU jobs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger("maya.ai.engine.checkpoint")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> Path:
    """Write a resume-capable checkpoint atomically-ish via torch.save."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "config": config,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved epoch=%s → %s", epoch, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model (and optionally optimizer/scheduler)."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if (
        scheduler is not None
        and payload.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    logger.info(
        "Checkpoint loaded epoch=%s from %s",
        payload.get("epoch"),
        path,
    )
    return payload
