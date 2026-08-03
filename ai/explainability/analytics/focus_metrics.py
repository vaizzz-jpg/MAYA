"""Focus metrics for explanation heatmaps (coverage, peak, entropy, …)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ai.explainability.analytics.statistics import _as_2d_float


@dataclass(frozen=True)
class FocusMetrics:
    """Quantitative focus characteristics of a single heatmap."""

    coverage: float  # fraction of pixels above activation threshold
    peak_activation: float
    mean_activation: float
    activation_variance: float
    activation_entropy: float  # normalized Shannon entropy in [0, 1]
    threshold_used: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_focus_metrics(
    heatmap: np.ndarray,
    *,
    coverage_threshold: float = 0.2,
    eps: float = 1e-12,
) -> FocusMetrics:
    """Measure how concentrated / diffuse an explanation map is."""

    arr = _as_2d_float(heatmap)
    peak = float(arr.max()) if arr.size else 0.0
    mean = float(arr.mean()) if arr.size else 0.0
    variance = float(arr.var()) if arr.size else 0.0
    coverage = float(np.mean(arr >= coverage_threshold)) if arr.size else 0.0
    entropy = _normalized_entropy(arr, eps=eps)
    return FocusMetrics(
        coverage=round(coverage, 6),
        peak_activation=round(peak, 6),
        mean_activation=round(mean, 6),
        activation_variance=round(variance, 6),
        activation_entropy=round(entropy, 6),
        threshold_used=float(coverage_threshold),
    )


def _normalized_entropy(arr: np.ndarray, *, eps: float = 1e-12) -> float:
    """Shannon entropy of the activation distribution, scaled to ``[0, 1]``."""

    flat = np.clip(arr.ravel(), 0.0, None)
    total = float(flat.sum())
    if total <= eps:
        return 0.0
    probs = flat / total
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    raw = float(-np.sum(probs * np.log(probs + eps)))
    max_entropy = float(np.log(flat.size))
    if max_entropy <= eps:
        return 0.0
    return float(np.clip(raw / max_entropy, 0.0, 1.0))
