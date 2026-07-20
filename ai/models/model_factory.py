"""Model factory — single entry point for constructing MAYA classifiers.

Why this module exists
----------------------
The future training engine must never ``import efficientnet_b0`` directly.
A factory lets us swap EfficientNet-B0 → B1/B2/ResNet/ViT by config alone.

Architecture role
-----------------
AI Analysis Layer factory. Maps ``ModelConfig.model_name`` → builder.
Unsupported names raise ``UnsupportedModelError`` with a clear message.

Design decisions
----------------
* Explicit registry dict for O(1) lookup and easy extension.
* Future model enums are listed but not registered until implemented.
* No device placement here (caller / engine decides) to avoid silent copies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

from torch import nn

from ai.models.efficientnet import build_efficientnet_b0
from ai.models.model_config import ModelConfig, ModelName, get_model_config

logger = logging.getLogger("maya.ai.models.factory")

Builder = Callable[[ModelConfig], nn.Module]


class UnsupportedModelError(ValueError):
    """Raised when ``ModelFactory`` is asked for an unimplemented model."""


_REGISTRY: Final[dict[str, Builder]] = {
    ModelName.EFFICIENTNET_B0.value: build_efficientnet_b0,
}

# Reserved identifiers (documented for expansion; not callable yet)
_PLANNED: Final[frozenset[str]] = frozenset(
    {
        ModelName.EFFICIENTNET_B1.value,
        ModelName.EFFICIENTNET_B2.value,
        ModelName.RESNET18.value,
        ModelName.RESNET50.value,
        ModelName.MOBILENET_V3.value,
        ModelName.VIT.value,
    }
)


class ModelFactory:
    """Create MAYA classification models by name without coupling to builders."""

    @staticmethod
    def supported_models() -> tuple[str, ...]:
        """Return currently implemented model identifiers."""

        return tuple(sorted(_REGISTRY.keys()))

    @staticmethod
    def planned_models() -> tuple[str, ...]:
        """Return reserved-but-unimplemented identifiers."""

        return tuple(sorted(_PLANNED))

    @staticmethod
    def create(
        model_name: str | None = None,
        *,
        config: ModelConfig | None = None,
    ) -> nn.Module:
        """Instantiate a model from the registry.

        Args:
            model_name: Optional override; defaults to ``config.model_name``.
            config: Optional ``ModelConfig``; defaults to ``get_model_config()``.

        Raises:
            UnsupportedModelError: unknown or not-yet-implemented model id.
        """

        cfg = config or get_model_config()
        name = (model_name or cfg.model_name).strip().lower()
        builder = _REGISTRY.get(name)

        if builder is None:
            if name in _PLANNED:
                raise UnsupportedModelError(
                    f"Model '{name}' is planned for a later sprint but not "
                    f"implemented yet. Implemented: {ModelFactory.supported_models()}"
                )
            raise UnsupportedModelError(
                f"Unknown model '{name}'. "
                f"Supported now: {ModelFactory.supported_models()}. "
                f"Planned later: {ModelFactory.planned_models()}"
            )

        logger.info("ModelFactory.create(%s)", name)
        return builder(cfg)
