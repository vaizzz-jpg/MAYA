"""EigenCAM explainer (Muhammad & Yeasin, 2020).

Class-agnostic localization via the first principal component of
activation maps (SVD), then ReLU for positive evidence display:

    reshape A → (K, u·v), center, SVD; CAM ← |v₁| reshaped to (u, v)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from ai.explainability.base import Explainer, ExplainerOutput
from ai.explainability.explainers.common import (
    capture_activations,
    validate_feature_maps,
)

logger = logging.getLogger("maya.ai.explainability.explainers.eigencam")


class EigenCAM(Explainer):
    """Eigen-CAM using the leading principal component of activations."""

    name = "eigencam"

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        if target_layer is None:
            raise ValueError("EigenCAM requires an explicit target_layer")

        maps = capture_activations(
            self.model,
            input_tensor,
            target_layer,
            self.device,
            target_class=target_class,
            need_gradients=False,
        )
        heatmap = compute_eigencam_map(maps.activations)

        logger.info(
            "EigenCAM class=%s layer=%s map=%s",
            maps.target_class,
            target_layer_name or type(target_layer).__name__,
            heatmap.shape,
        )
        return ExplainerOutput(
            raw_heatmap=heatmap,
            target_class=maps.target_class,
            target_layer_name=target_layer_name or type(target_layer).__name__,
            logits=maps.logits.numpy()[0],
            probabilities=maps.probabilities,
            metadata={
                "explainer": self.name,
                "paper": "Muhammad & Yeasin, 2020",
                "class_discriminative": False,
            },
        )


def compute_eigencam_map(activations: torch.Tensor) -> np.ndarray:
    """First principal component of channel×spatial activation matrix."""

    validate_feature_maps(activations)
    acts = activations[0].detach().cpu().numpy().astype(np.float64)  # (K, u, v)
    channels, height, width = acts.shape
    flat = acts.reshape(channels, -1)  # (K, u*v)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # Economy SVD; VT[0] is the leading right singular vector over spatial dims
    _u, _s, vt = np.linalg.svd(flat, full_matrices=False)
    component = vt[0].reshape(height, width)
    # Absolute value then ReLU-equivalent non-negativity for visualization
    localization = np.maximum(component, 0.0).astype(np.float32)
    # If the leading component is mostly negative, flip sign (standard PCA CAM practice)
    if float(localization.sum()) < 1e-8:
        localization = np.maximum(-component, 0.0).astype(np.float32)
    return localization
