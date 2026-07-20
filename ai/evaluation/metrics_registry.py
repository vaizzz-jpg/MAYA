"""Metric registry — register-once, calculate-all without hardcoding lists.

Purpose
-------
Allow future metrics to plug in via ``MetricRegistry.register`` without
changing ``Evaluator`` architecture.

Architecture
------------
Registry Layer between pure ``metrics`` functions and ``Evaluator``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ai.evaluation import metrics as metric_fns

logger = logging.getLogger("maya.ai.evaluation.registry")

MetricFn = Callable[..., float]


class MetricRegistry:
    """Global registry of named metric callables."""

    _registry: dict[str, MetricFn] = {}

    @classmethod
    def register(cls, name: str, fn: MetricFn | None = None) -> Callable[[MetricFn], MetricFn] | MetricFn:
        """Register a metric function. Usable as ``@MetricRegistry.register('x')``."""

        def decorator(func: MetricFn) -> MetricFn:
            key = name.strip().lower()
            cls._registry[key] = func
            logger.debug("Registered metric: %s", key)
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry.keys()))

    @classmethod
    def clear(cls) -> None:
        """Test helper — wipe registry (re-call ``register_default_metrics`` after)."""

        cls._registry.clear()

    @classmethod
    def calculate(
        cls,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_score: np.ndarray | None = None,
        positive_class: int = 1,
    ) -> float:
        key = name.strip().lower()
        if key not in cls._registry:
            raise KeyError(f"Unknown metric '{name}'. Registered: {cls.names()}")
        return float(
            cls._registry[key](
                y_true,
                y_pred,
                y_score=y_score,
                positive_class=positive_class,
            )
        )

    @classmethod
    def calculate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_score: np.ndarray | None = None,
        positive_class: int = 1,
    ) -> dict[str, float]:
        """Iterate every registered metric (no hardcoded metric list)."""

        results: dict[str, float] = {}
        for name in cls.names():
            try:
                results[name] = cls.calculate(
                    name,
                    y_true,
                    y_pred,
                    y_score=y_score,
                    positive_class=positive_class,
                )
            except Exception as exc:  # noqa: BLE001 — never abort whole eval
                logger.warning("Metric %s failed: %s", name, exc)
                results[name] = float("nan")
        return results


def register_default_metrics() -> None:
    """Register Sprint 3.3 built-in metrics if the registry is empty/missing them."""

    defaults: dict[str, MetricFn] = {
        "accuracy": metric_fns.accuracy,
        "precision": metric_fns.precision,
        "recall": metric_fns.recall,
        "f1": metric_fns.f1_score,
        "roc_auc": metric_fns.roc_auc,
        "balanced_accuracy": metric_fns.balanced_accuracy,
        "matthews_corrcoef": metric_fns.matthews_corrcoef,
        "cohen_kappa": metric_fns.cohen_kappa,
        "sensitivity": metric_fns.sensitivity,
        "specificity": metric_fns.specificity,
        "false_positive_rate": metric_fns.false_positive_rate,
        "false_negative_rate": metric_fns.false_negative_rate,
    }
    for name, fn in defaults.items():
        if name not in MetricRegistry._registry:
            MetricRegistry.register(name, fn)


# Auto-register on import
register_default_metrics()
