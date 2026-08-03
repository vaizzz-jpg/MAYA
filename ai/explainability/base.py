"""Abstract explainer contract for MAYA Explainable Intelligence.

Purpose
-------
Every future explainer (Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM, …) inherits
from ``Explainer`` and returns structured explainability data — never images.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class ExplainerOutput:
    """Raw explainer payload before heatmap / overlay post-processing."""

    raw_heatmap: np.ndarray  # 2-D float array (H, W) of the CAM
    target_class: int
    target_layer_name: str
    logits: np.ndarray
    probabilities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class Explainer(ABC):
    """Abstract base class for class-activation explainers."""

    name: str = "base"

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @abstractmethod
    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        """Produce a raw class-activation heatmap for ``input_tensor``.

        Args:
            input_tensor: Batched image tensor ``(1, C, H, W)``.
            target_class: Class index to explain. ``None`` → argmax of logits.
            target_layer: Module whose activations/gradients are captured.
            target_layer_name: Human-readable layer path for artefacts.
        """

        raise NotImplementedError


# Sprint 4.2 alias — same contract, clearer plugin naming
BaseExplainer = Explainer
