"""Automatic target-layer resolution for CAM explainers.

Purpose
-------
Select the last spatial feature stage without hardcoding layer strings in
Grad-CAM. Future CNN families register resolvers here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from torch import nn

from ai.models.model_config import ModelName

logger = logging.getLogger("maya.ai.explainability.target_layer")

LayerResolver = Callable[[nn.Module], tuple[nn.Module, str]]


class UnsupportedModelForExplainability(ValueError):
    """Raised when no target-layer resolver exists for a model family."""


class TargetLayerResolver:
    """Registry of model-family → target-layer resolvers."""

    _resolvers: dict[str, LayerResolver] = {}

    @classmethod
    def register(cls, model_name: str, resolver: LayerResolver) -> None:
        key = model_name.strip().lower()
        cls._resolvers[key] = resolver
        logger.debug("Registered target-layer resolver: %s", key)

    @classmethod
    def resolve(
        cls,
        model: nn.Module,
        model_name: str,
        *,
        override: str | None = None,
    ) -> tuple[nn.Module, str]:
        """Return ``(module, dotted_name)`` for Grad-CAM capture.

        Args:
            model: Loaded classifier.
            model_name: Logical model id (e.g. ``efficientnet_b0``).
            override: Optional dotted attribute path (e.g. ``features.8``).
        """

        if override:
            layer = _resolve_dotted(model, override)
            logger.info("Using overridden target layer: %s", override)
            return layer, override

        key = model_name.strip().lower()
        resolver = cls._resolvers.get(key)
        if resolver is None:
            supported = ", ".join(sorted(cls._resolvers)) or "(none)"
            raise UnsupportedModelForExplainability(
                f"No target-layer resolver for '{model_name}'. "
                f"Supported: {supported}. Pass target_layer_override to force a path."
            )
        layer, name = resolver(model)
        logger.info("Resolved target layer for %s → %s", model_name, name)
        return layer, name

    @classmethod
    def clear(cls) -> None:
        cls._resolvers.clear()


def _resolve_dotted(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            children = list(module.children())
            idx = int(part)
            if idx < 0 or idx >= len(children):
                raise UnsupportedModelForExplainability(
                    f"Index {idx} out of range while resolving '{path}'"
                )
            module = children[idx]
        else:
            if not hasattr(module, part):
                raise UnsupportedModelForExplainability(
                    f"Attribute '{part}' not found while resolving '{path}'"
                )
            module = getattr(module, part)
            if not isinstance(module, nn.Module):
                raise UnsupportedModelForExplainability(
                    f"Resolved '{path}' but '{part}' is not an nn.Module"
                )
    return module


def _efficientnet_b0_target(model: nn.Module) -> tuple[nn.Module, str]:
    """Last feature stage of torchvision EfficientNet-B0 (spatial maps)."""

    if not hasattr(model, "features"):
        raise UnsupportedModelForExplainability(
            "EfficientNet-B0 target requires model.features"
        )
    features = model.features
    if not isinstance(features, nn.Sequential) or len(features) == 0:
        raise UnsupportedModelForExplainability(
            "EfficientNet-B0 features must be a non-empty Sequential"
        )
    # features[-1] = final Conv2dNormActivation before avgpool/classifier
    index = len(features) - 1
    return features[index], f"features.{index}"


def register_default_target_layers() -> None:
    """Register Sprint 4.1 built-in resolvers."""

    if ModelName.EFFICIENTNET_B0.value not in TargetLayerResolver._resolvers:
        TargetLayerResolver.register(
            ModelName.EFFICIENTNET_B0.value,
            _efficientnet_b0_target,
        )


# Auto-register on import
register_default_target_layers()
