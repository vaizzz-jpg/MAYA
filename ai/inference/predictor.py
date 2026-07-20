"""Raw probability predictor (no thresholding).

Purpose
-------
Run a single forward pass under ``torch.no_grad`` and return softmax
probabilities only — threshold/confidence live elsewhere.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger("maya.ai.inference.predictor")


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    batch_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return softmax probabilities with shape ``(N, num_classes)``.

    Does **not** apply a decision threshold.
    """

    model.eval()
    tensor = batch_tensor.to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    logger.debug("Predicted probabilities shape=%s", tuple(probs.shape))
    # Free intermediate logits promptly on CPU-bound hosts
    del logits
    return probs.detach().cpu()
