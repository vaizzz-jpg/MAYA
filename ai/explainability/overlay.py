"""Overlay a normalized heatmap onto an RGB image — no Grad-CAM math."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import colormaps
from PIL import Image

from ai.explainability.config import ColorMap

logger = logging.getLogger("maya.ai.explainability.overlay")


def apply_colormap(
    heatmap: np.ndarray,
    color_map: ColorMap = ColorMap.JET,
) -> np.ndarray:
    """Map a ``[0, 1]`` heatmap to an RGB uint8 image ``(H, W, 3)``."""

    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Heatmap must be 2-D, got {arr.shape}")
    cmap = colormaps.get_cmap(color_map.value)
    colored = cmap(np.clip(arr, 0.0, 1.0))  # (H, W, 4) RGBA float
    rgb = (colored[..., :3] * 255.0).astype(np.uint8)
    return rgb


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    *,
    opacity: float = 0.45,
    color_map: ColorMap = ColorMap.JET,
) -> Image.Image:
    """Blend coloured heatmap onto the original RGB image.

    Args:
        image: PIL image or HxWx3 uint8 array.
        heatmap: Normalized 2-D map matching the image spatial size (or resized).
        opacity: Heatmap blend weight in ``[0, 1]``.
        color_map: Matplotlib colormap name.
    """

    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")

    if isinstance(image, Image.Image):
        base = image.convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {arr.shape}")
        base = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.shape[:2] != (base.height, base.width):
        heat_img = Image.fromarray(heat, mode="F").resize(
            base.size, resample=Image.Resampling.BILINEAR
        )
        heat = np.asarray(heat_img, dtype=np.float32)

    colored = apply_colormap(heat, color_map=color_map)
    heat_pil = Image.fromarray(colored, mode="RGB")
    blended = Image.blend(base, heat_pil, alpha=opacity)
    logger.debug("Overlay generated size=%s opacity=%.2f", blended.size, opacity)
    return blended
