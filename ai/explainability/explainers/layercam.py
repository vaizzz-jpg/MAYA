"""LayerCAM explainer (Jiang et al., IEEE TIP 2021).

Spatially weights each activation by its positive gradient:

    L^c = ReLU( Σ_k  ReLU(∂y^c/∂A^k) ⊙ A^k )
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

logger = logging.getLogger("maya.ai.explainability.explainers.layercam")


class LayerCAM(Explainer):
    """TIP 2021 LayerCAM for hierarchical / spatial gradient weighting."""

    name = "layercam"

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        if target_layer is None:
            raise ValueError("LayerCAM requires an explicit target_layer")

        maps = capture_activations(
            self.model,
            input_tensor,
            target_layer,
            self.device,
            target_class=target_class,
            need_gradients=True,
        )
        assert maps.gradients is not None
        heatmap = compute_layercam_map(maps.activations, maps.gradients)

        logger.info(
            "LayerCAM class=%s layer=%s map=%s",
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
            metadata={"explainer": self.name, "paper": "Jiang et al., IEEE TIP 2021"},
        )


def compute_layercam_map(
    activations: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    """LayerCAM: element-wise positive-gradient weighting then channel sum."""

    validate_feature_maps(activations, gradients)
    spatial_weights = torch.relu(gradients)
    weighted = (spatial_weights * activations).sum(dim=1)
    localization = torch.relu(weighted[0])
    out = localization.cpu().numpy().astype(np.float32, copy=False)
    del spatial_weights, weighted, localization
    return out
