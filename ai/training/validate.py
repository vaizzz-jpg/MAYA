"""Validation loop for MAYA classifiers.

Why this module exists
----------------------
Own validation-only logic (no optimizer steps). Returns scalar metrics so the
trainer and callbacks remain interchangeable.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger("maya.ai.training.validate")


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` and return loss/accuracy metrics.

    Processes one batch at a time; never materializes the full set of
    predictions in RAM.
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        predictions = outputs.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size

        # Promptly drop batch references (helps on 8 GB hosts)
        del images, labels, outputs, loss, predictions

    if total_samples == 0:
        logger.warning("Validation loader produced zero samples")
        return {"val_loss": float("nan"), "val_accuracy": float("nan")}

    metrics = {
        "val_loss": total_loss / total_samples,
        "val_accuracy": total_correct / total_samples,
    }
    logger.info(
        "Validation metrics: loss=%.6f acc=%.4f n=%s",
        metrics["val_loss"],
        metrics["val_accuracy"],
        total_samples,
    )
    return metrics
