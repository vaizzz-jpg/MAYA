"""Visualization artefact writers for explainability outputs.

Saves original / heatmap / overlay / side-by-side comparison images.
Does not compute Grad-CAM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai.explainability.config import ColorMap
from ai.explainability.overlay import apply_colormap

logger = logging.getLogger("maya.ai.explainability.visualization")


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_rgb(path: Path | str) -> Image.Image:
    path = Path(path)
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        raise


def save_heatmap_image(
    heatmap: np.ndarray,
    path: Path,
    *,
    color_map: ColorMap = ColorMap.JET,
) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    rgb = apply_colormap(heatmap, color_map=color_map)
    Image.fromarray(rgb, mode="RGB").save(path)
    logger.info("Saved heatmap → %s", path)
    return path


def save_overlay_image(overlay: Image.Image, path: Path) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    overlay.convert("RGB").save(path)
    logger.info("Saved overlay → %s", path)
    return path


def save_original_image(image: Image.Image, path: Path) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    image.convert("RGB").save(path)
    return path


def save_comparison_figure(
    original: Image.Image,
    heatmap_rgb: Image.Image,
    overlay: Image.Image,
    path: Path,
) -> Path:
    """Side-by-side Original | Heatmap | Overlay."""

    path = Path(path)
    _ensure_dir(path.parent)

    panels = [
        original.convert("RGB"),
        heatmap_rgb.convert("RGB"),
        overlay.convert("RGB"),
    ]
    height = max(p.height for p in panels)
    resized: list[Image.Image] = []
    for panel in panels:
        if panel.height != height:
            new_w = int(panel.width * (height / panel.height))
            panel = panel.resize((new_w, height), Image.Resampling.BILINEAR)
        resized.append(panel)

    gap = 8
    total_w = sum(p.width for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, height), color=(245, 245, 245))
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap

    canvas.save(path)
    logger.info("Saved comparison figure → %s", path)
    return path


def write_visualization_bundle(
    *,
    source_image: Path | Image.Image,
    heatmap: np.ndarray,
    overlay: Image.Image,
    artifact_dir: Path,
    color_map: ColorMap = ColorMap.JET,
    output_size: int | None = None,
) -> dict[str, Path]:
    """Write original.png, heatmap.png, overlay.png, comparison.png."""

    out = _ensure_dir(artifact_dir)
    if isinstance(source_image, Image.Image):
        original = source_image.convert("RGB")
    else:
        original = _load_rgb(source_image)

    if output_size is not None:
        original = original.resize(
            (output_size, output_size), Image.Resampling.BILINEAR
        )

    heat_rgb = Image.fromarray(
        apply_colormap(heatmap, color_map=color_map), mode="RGB"
    )
    if heat_rgb.size != original.size:
        heat_rgb = heat_rgb.resize(original.size, Image.Resampling.BILINEAR)
    if overlay.size != original.size:
        overlay = overlay.resize(original.size, Image.Resampling.BILINEAR)

    return {
        "original": save_original_image(original, out / "original.png"),
        "heatmap": save_heatmap_image(
            heatmap, out / "heatmap.png", color_map=color_map
        ),
        "overlay": save_overlay_image(overlay, out / "overlay.png"),
        "comparison": save_comparison_figure(
            original, heat_rgb, overlay, out / "comparison.png"
        ),
    }
