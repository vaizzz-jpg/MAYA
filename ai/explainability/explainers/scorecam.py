"""Score-CAM explainer (Wang et al., CVPRW 2020).

Gradient-free channel weighting via forward confidence:

    For each channel k:
      M^k = upsample(normalize(A^k))
      S^k = f(X ⊙ M^k)_c − f(X_baseline)_c   (CIC / score increase)
    α = softmax(S)
    L = ReLU( Σ_k α_k A^k )

Batched masked forwards keep CPU memory bounded (8 GB hosts).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ai.explainability.base import Explainer, ExplainerOutput
from ai.explainability.explainers.common import (
    capture_activations,
    validate_feature_maps,
)

logger = logging.getLogger("maya.ai.explainability.explainers.scorecam")


class ScoreCAM(Explainer):
    """CVPRW 2020 Score-CAM (score-weighted activation maps)."""

    name = "scorecam"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        batch_size: int = 8,
    ) -> None:
        super().__init__(model, device)
        self.batch_size = max(1, int(batch_size))

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
        target_layer: nn.Module | None = None,
        target_layer_name: str = "",
    ) -> ExplainerOutput:
        if target_layer is None:
            raise ValueError("ScoreCAM requires an explicit target_layer")

        maps = capture_activations(
            self.model,
            input_tensor,
            target_layer,
            self.device,
            target_class=target_class,
            need_gradients=False,
        )
        heatmap = compute_scorecam_map(
            self.model,
            input_tensor,
            maps.activations,
            maps.target_class,
            self.device,
            batch_size=self.batch_size,
        )

        logger.info(
            "Score-CAM class=%s layer=%s map=%s batch=%s",
            maps.target_class,
            target_layer_name or type(target_layer).__name__,
            heatmap.shape,
            self.batch_size,
        )
        return ExplainerOutput(
            raw_heatmap=heatmap,
            target_class=maps.target_class,
            target_layer_name=target_layer_name or type(target_layer).__name__,
            logits=maps.logits.numpy()[0],
            probabilities=maps.probabilities,
            metadata={
                "explainer": self.name,
                "paper": "Wang et al., CVPRW 2020",
                "batch_size": self.batch_size,
            },
        )


def compute_scorecam_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    activations: torch.Tensor,
    target_class: int,
    device: torch.device,
    *,
    batch_size: int = 8,
    eps: float = 1e-8,
) -> np.ndarray:
    """Score-CAM localization via batched masked forwards."""

    validate_feature_maps(activations)
    model.eval()
    x = input_tensor.to(device).detach()
    _, _, img_h, img_w = x.shape
    acts = activations[0].detach()  # (K, u, v)
    channels = acts.shape[0]

    # Channel-wise upsample + [0,1] normalize → masks
    up = F.interpolate(
        acts.unsqueeze(0),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    )[0]  # (K, H, W)
    flat_min = up.amin(dim=(1, 2), keepdim=True)
    flat_max = up.amax(dim=(1, 2), keepdim=True)
    denom = (flat_max - flat_min).clamp_min(eps)
    masks = (up - flat_min) / denom  # (K, H, W)

    with torch.no_grad():
        baseline = float(model(x)[0, target_class].item())

    scores = torch.zeros(channels, device=device, dtype=torch.float32)
    for start in range(0, channels, batch_size):
        end = min(start + batch_size, channels)
        batch_masks = masks[start:end]  # (B, H, W)
        # Broadcast mask over RGB: (B, 1, H, W) * (1, 3, H, W)
        masked = x * batch_masks.unsqueeze(1)
        with torch.no_grad():
            logits = model(masked)
            scores[start:end] = logits[:, target_class] - baseline
        del masked, logits, batch_masks

    weights = torch.softmax(scores, dim=0)  # (K,)
    # Weighted sum of original (coarse) activation maps
    weighted = (weights.view(-1, 1, 1) * acts).sum(dim=0)
    localization = torch.relu(weighted)
    out = localization.cpu().numpy().astype(np.float32, copy=False)

    del up, masks, scores, weights, weighted, localization, acts, x
    return out
