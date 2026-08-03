"""Activation / gradient hook capture for CAM explainers.

Registers forward (+ optional full-backward) hooks on one module and always
removes them after use to avoid memory leaks on CPU hosts.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("maya.ai.explainability.hooks")


class ActivationGradientHooks:
    """Capture ``A`` (activations) and optionally ``∂y/∂A`` (gradients)."""

    def __init__(
        self,
        target_layer: nn.Module,
        *,
        capture_gradients: bool = True,
    ) -> None:
        self.target_layer = target_layer
        self.capture_gradients = capture_gradients
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    def _on_forward(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        # Clone when gradients are needed so inplace ops in later modules
        # cannot invalidate the full-backward hook view (PyTorch constraint).
        if self.capture_gradients and torch.is_tensor(output):
            self.activations = output.clone()
        else:
            self.activations = output

    def _on_backward(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0]

    def register(self) -> None:
        self.remove()
        self._handles.append(
            self.target_layer.register_forward_hook(self._on_forward)
        )
        if self.capture_gradients:
            self._handles.append(
                self.target_layer.register_full_backward_hook(self._on_backward)
            )
        logger.debug(
            "Hooks registered on %s (gradients=%s)",
            type(self.target_layer).__name__,
            self.capture_gradients,
        )

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hook removal failed: %s", exc)
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def __enter__(self) -> ActivationGradientHooks:
        self.register()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove()
