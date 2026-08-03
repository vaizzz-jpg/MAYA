"""Shared capture helpers for CAM explainers (native PyTorch).

Keeps hook/forward/backward boilerplate out of each algorithm module
without introducing managers or engine coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ai.explainability.hooks import ActivationGradientHooks


@dataclass
class CapturedMaps:
    """Activations (and optional gradients) plus classifier scores."""

    activations: torch.Tensor
    gradients: torch.Tensor | None
    logits: torch.Tensor
    probabilities: np.ndarray
    target_class: int


def select_target_class(
    logits: torch.Tensor,
    target_class: int | None,
) -> int:
    """Resolve class index; ``None`` → argmax of pre-softmax scores."""

    if target_class is not None:
        idx = int(target_class)
    else:
        idx = int(torch.argmax(logits, dim=1).item())
    if idx < 0 or idx >= logits.shape[1]:
        raise ValueError(f"target_class {idx} out of range for logits {tuple(logits.shape)}")
    return idx


def capture_activations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    device: torch.device,
    *,
    target_class: int | None = None,
    need_gradients: bool = True,
) -> CapturedMaps:
    """One forward (and optional backward) with automatic hook cleanup.

    For gradient-based CAMs, ``input_tensor`` is cloned with ``requires_grad``
    so frozen backbones still expose ``∂y/∂A``.
    """

    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            f"Expected input shape (1, C, H, W), got {tuple(input_tensor.shape)}"
        )
    if target_layer is None:
        raise ValueError("target_layer is required")

    model.eval()
    if need_gradients:
        tensor = input_tensor.to(device).detach().requires_grad_(True)
    else:
        tensor = input_tensor.to(device).detach()

    with ActivationGradientHooks(
        target_layer, capture_gradients=need_gradients
    ) as hooks:
        if need_gradients:
            logits = model(tensor)
        else:
            with torch.no_grad():
                logits = model(tensor)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected logits shape {tuple(logits.shape)}; expected (1, C)"
            )

        class_idx = select_target_class(logits, target_class)

        gradients: torch.Tensor | None = None
        if need_gradients:
            y_c = logits[0, class_idx]
            model.zero_grad(set_to_none=True)
            y_c.backward(retain_graph=False)
            if hooks.activations is None or hooks.gradients is None:
                raise RuntimeError(
                    "Failed to capture A / ∂y/∂A from target layer hooks."
                )
            activations = hooks.activations.detach()
            gradients = hooks.gradients.detach()
            del y_c
        else:
            if hooks.activations is None:
                raise RuntimeError("Failed to capture activations from target layer.")
            activations = hooks.activations.detach()

        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
        logits_cpu = logits.detach().cpu()

    del tensor
    return CapturedMaps(
        activations=activations,
        gradients=gradients,
        logits=logits_cpu,
        probabilities=probs,
        target_class=class_idx,
    )


def validate_feature_maps(
    activations: torch.Tensor,
    gradients: torch.Tensor | None = None,
) -> None:
    if activations.ndim != 4:
        raise RuntimeError(
            f"Expected (N, K, u, v) feature maps, got {tuple(activations.shape)}"
        )
    if gradients is not None and activations.shape != gradients.shape:
        raise RuntimeError(
            f"A / ∂y/∂A shape mismatch: {tuple(activations.shape)} vs "
            f"{tuple(gradients.shape)}"
        )
