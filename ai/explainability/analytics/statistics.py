"""Descriptive statistics for explanation activation maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActivationStatistics:
    """Summary statistics over a 2-D activation / heatmap array."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    p25: float
    p75: float
    p90: float
    p99: float
    sparsity: float  # fraction of near-zero values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_activation_statistics(
    heatmap: np.ndarray,
    *,
    zero_eps: float = 1e-6,
) -> ActivationStatistics:
    """Compute descriptive statistics for a 2-D heatmap."""

    arr = _as_2d_float(heatmap)
    flat = arr.ravel()
    return ActivationStatistics(
        count=int(flat.size),
        minimum=float(flat.min()) if flat.size else 0.0,
        maximum=float(flat.max()) if flat.size else 0.0,
        mean=float(flat.mean()) if flat.size else 0.0,
        median=float(np.median(flat)) if flat.size else 0.0,
        std=float(flat.std()) if flat.size else 0.0,
        p25=float(np.percentile(flat, 25)) if flat.size else 0.0,
        p75=float(np.percentile(flat, 75)) if flat.size else 0.0,
        p90=float(np.percentile(flat, 90)) if flat.size else 0.0,
        p99=float(np.percentile(flat, 99)) if flat.size else 0.0,
        sparsity=float(np.mean(flat <= zero_eps)) if flat.size else 1.0,
    )


def _as_2d_float(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {arr.shape}")
    return arr
