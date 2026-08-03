"""Heatmap normalization and spatial upsample (visualization only).

Paper note (Selvaraju et al., ICCV 2017): after computing L^c_Grad-CAM,
upsample to input resolution (bilinear) and scale for display. This module
does not implement Grad-CAM equations.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ai.explainability.config import InterpolationMethod

logger = logging.getLogger("maya.ai.explainability.heatmap")

_PIL_RESAMPLE = {
    InterpolationMethod.BILINEAR: Image.Resampling.BILINEAR,
    InterpolationMethod.BICUBIC: Image.Resampling.BICUBIC,
    InterpolationMethod.NEAREST: Image.Resampling.NEAREST,
}


def upsample_heatmap(
    heatmap: np.ndarray,
    size: tuple[int, int],
    *,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> np.ndarray:
    """Bilinear (default) upsample of a coarse CAM to ``(height, width)``."""

    height, width = size
    if height < 1 or width < 1:
        raise ValueError(f"Invalid upsample size: {size}")
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Heatmap must be 2-D, got shape {arr.shape}")
    img = Image.fromarray(arr, mode="F")
    resample = _PIL_RESAMPLE.get(interpolation, Image.Resampling.BILINEAR)
    return np.asarray(img.resize((width, height), resample=resample), dtype=np.float32)


def normalize_heatmap(
    heatmap: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Min-max scale a CAM to ``[0, 1]`` for stable visualization."""

    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Heatmap must be 2-D, got shape {arr.shape}")
    lo = float(arr.min())
    hi = float(arr.max())
    denom = hi - lo
    if denom < eps:
        logger.warning("Near-constant heatmap (range=%.3e) — zeros", denom)
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (denom + eps), 0.0, 1.0).astype(np.float32, copy=False)


def prepare_heatmap(
    raw_heatmap: np.ndarray,
    output_size: tuple[int, int],
    *,
    eps: float = 1e-8,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
) -> np.ndarray:
    """Upsample then normalize — paper visualization order."""

    upsampled = upsample_heatmap(raw_heatmap, output_size, interpolation=interpolation)
    return normalize_heatmap(upsampled, eps=eps)
