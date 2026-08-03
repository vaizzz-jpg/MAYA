"""Grad-CAM explainer (Selvaraju et al., ICCV 2017).

    (1)  α_k^c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A_ij^k
    (2)  L^c_Grad-CAM = ReLU( Σ_k α_k^c A^k )
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

logger = logging.getLogger("maya.ai.explainability.explainers.gradcam")


class GradCAM(Explainer):
    """ICCV 2017 Grad-CAM for CNN classifiers."""

    name = "gradcam"

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        if target_layer is None:
            raise ValueError("GradCAM requires an explicit target_layer module")

        maps = capture_activations(
            self.model,
            input_tensor,
            target_layer,
            self.device,
            target_class=target_class,
            need_gradients=True,
        )
        assert maps.gradients is not None
        heatmap = compute_gradcam_map(maps.activations, maps.gradients)

        logger.info(
            "Grad-CAM class=%s layer=%s map=%s",
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
            metadata={"explainer": self.name, "paper": "Selvaraju et al., ICCV 2017"},
        )


def compute_gradcam_map(
    activations: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    """Apply equations (1)–(2) from Selvaraju et al. (ICCV 2017)."""

    validate_feature_maps(activations, gradients)
    alpha = gradients.mean(dim=(2, 3), keepdim=True)
    weighted = (alpha * activations).sum(dim=1)
    localization = torch.relu(weighted[0])
    out = localization.cpu().numpy().astype(np.float32, copy=False)
    del alpha, weighted, localization
    return out
