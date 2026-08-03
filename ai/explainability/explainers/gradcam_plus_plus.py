"""Grad-CAM++ explainer (Chattopadhay et al., WACV 2018).

Uses pixel-wise gradient weighting derived from higher-order terms.
Practical closed-form used in mature native PyTorch ports:

    α_ij^k ∝ (∂y/∂A_ij^k)² /
              (2 (∂y/∂A_ij^k)² + Σ_{a,b} A_ab^k (∂y/∂A_ij^k)³)

    w^k = Σ_{i,j} α_ij^k · ReLU(∂y/∂A_ij^k)
    L   = ReLU( Σ_k w^k A^k )
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

logger = logging.getLogger("maya.ai.explainability.explainers.gradcam_plus_plus")


class GradCAMPlusPlus(Explainer):
    """WACV 2018 Grad-CAM++ for CNN classifiers."""

    name = "gradcam_plus_plus"

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        if target_layer is None:
            raise ValueError("GradCAMPlusPlus requires an explicit target_layer")

        maps = capture_activations(
            self.model,
            input_tensor,
            target_layer,
            self.device,
            target_class=target_class,
            need_gradients=True,
        )
        assert maps.gradients is not None
        heatmap = compute_gradcam_pp_map(maps.activations, maps.gradients)

        logger.info(
            "Grad-CAM++ class=%s layer=%s map=%s",
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
                "paper": "Chattopadhay et al., WACV 2018",
            },
        )


def compute_gradcam_pp_map(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Grad-CAM++ localization map from activations and first-order grads."""

    validate_feature_maps(activations, gradients)
    grads = gradients
    grads_2 = grads.pow(2)
    grads_3 = grads_2 * grads
    sum_a = activations.sum(dim=(2, 3), keepdim=True)  # (N, K, 1, 1)
    denom = 2.0 * grads_2 + sum_a * grads_3 + eps
    alpha = grads_2 / denom
    alpha = torch.where(grads != 0, alpha, torch.zeros_like(alpha))
    weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
    weighted = (weights * activations).sum(dim=1)
    localization = torch.relu(weighted[0])
    out = localization.cpu().numpy().astype(np.float32, copy=False)
    del grads_2, grads_3, sum_a, denom, alpha, weights, weighted, localization
    return out
