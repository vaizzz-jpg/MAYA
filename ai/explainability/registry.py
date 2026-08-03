"""Explainer registry — plug-in explainers without if/elif chains.

Future explainers: implement ``Explainer``, then register a factory.
The engine never hardcodes algorithm names beyond reading configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from torch import nn

from ai.explainability.base import Explainer

logger = logging.getLogger("maya.ai.explainability.registry")

ExplainerFactory = Callable[[nn.Module, object], Explainer]
T = TypeVar("T", bound=Explainer)

# Canonical Sprint 4.2 plugin set (order used for comparisons)
DEFAULT_EXPLAINER_NAMES: tuple[str, ...] = (
    "gradcam",
    "gradcam_plus_plus",
    "layercam",
    "scorecam",
    "eigencam",
)


class ExplainerRegistry:
    """Global name → factory registry for explainability backends."""

    _registry: dict[str, ExplainerFactory] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: ExplainerFactory | None = None,
    ) -> Callable[[ExplainerFactory], ExplainerFactory] | ExplainerFactory:
        """Register an explainer factory. Usable as a decorator."""

        def decorator(fn: ExplainerFactory) -> ExplainerFactory:
            key = name.strip().lower()
            cls._registry[key] = fn
            logger.debug("Registered explainer: %s", key)
            return fn

        if factory is not None:
            return decorator(factory)
        return decorator

    @classmethod
    def get(cls, name: str) -> ExplainerFactory:
        key = name.strip().lower()
        if key not in cls._registry:
            available = ", ".join(cls.list_available()) or "(none)"
            raise KeyError(f"Unknown explainer '{name}'. Available: {available}")
        return cls._registry[key]

    @classmethod
    def create(cls, name: str, model: nn.Module, device: object) -> Explainer:
        """Instantiate a registered explainer for ``model`` on ``device``."""

        factory = cls.get(name)
        return factory(model, device)

    @classmethod
    def list_available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry.keys()))

    @classmethod
    def clear(cls) -> None:
        """Test helper — wipe registry."""

        cls._registry.clear()


def register_default_explainers() -> None:
    """Register Sprint 4.1 + 4.2 built-in explainers if missing."""

    from ai.explainability.explainers.eigencam import EigenCAM
    from ai.explainability.explainers.gradcam import GradCAM
    from ai.explainability.explainers.gradcam_plus_plus import GradCAMPlusPlus
    from ai.explainability.explainers.layercam import LayerCAM
    from ai.explainability.explainers.scorecam import ScoreCAM

    builtins: dict[str, ExplainerFactory] = {
        "gradcam": lambda m, d: GradCAM(m, d),  # type: ignore[arg-type, return-value]
        "gradcam_plus_plus": lambda m, d: GradCAMPlusPlus(m, d),  # type: ignore[arg-type, return-value]
        "gradcam++": lambda m, d: GradCAMPlusPlus(m, d),  # type: ignore[arg-type, return-value]
        "layercam": lambda m, d: LayerCAM(m, d),  # type: ignore[arg-type, return-value]
        "scorecam": lambda m, d: ScoreCAM(m, d),  # type: ignore[arg-type, return-value]
        "eigencam": lambda m, d: EigenCAM(m, d),  # type: ignore[arg-type, return-value]
    }
    for name, factory in builtins.items():
        if name not in ExplainerRegistry._registry:
            ExplainerRegistry.register(name, factory)
