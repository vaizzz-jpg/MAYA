"""SHAP contribution visualizations (not Grad-CAM attention maps)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from ai.explainability.config import ColorMap
from ai.explainability.overlay import apply_colormap, overlay_heatmap
from ai.explainability.visualization import (
    save_comparison_figure,
    save_heatmap_image,
    save_overlay_image,
)

logger = logging.getLogger("maya.ai.explainability.shap.visualization")


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label_banner(image: Image.Image, text: str) -> Image.Image:
    """Add a small caption strip so SHAP is not mistaken for CAM attention."""

    rgb = image.convert("RGB")
    banner_h = 28
    canvas = Image.new("RGB", (rgb.width, rgb.height + banner_h), (250, 250, 250))
    canvas.paste(rgb, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((6, 6), text, fill=(30, 30, 30), font=font)
    return canvas


def write_shap_visualizations(
    *,
    original: Image.Image,
    abs_map: np.ndarray,
    positive_map: np.ndarray,
    negative_map: np.ndarray,
    artifact_dir: Path,
    opacity: float = 0.45,
    color_map: ColorMap = ColorMap.JET,
) -> dict[str, Path]:
    """Write shap_heatmap / overlay / comparison (+ pos/neg panels)."""

    out = _ensure_dir(artifact_dir)
    original = original.convert("RGB")

    overlay = overlay_heatmap(
        original, abs_map, opacity=opacity, color_map=color_map
    )
    overlay = _label_banner(overlay, "SHAP contribution (not Grad-CAM attention)")

    heat_path = save_heatmap_image(
        abs_map, out / "shap_heatmap.png", color_map=color_map
    )
    overlay_path = save_overlay_image(overlay, out / "shap_overlay.png")

    heat_rgb = Image.fromarray(apply_colormap(abs_map, color_map=color_map), mode="RGB")
    heat_rgb = _label_banner(heat_rgb, "SHAP |abs| contribution")
    comparison_path = save_comparison_figure(
        _label_banner(original, "Original evidence"),
        heat_rgb,
        overlay,
        out / "shap_comparison.png",
    )

    # Positive vs negative contribution strip
    pos_rgb = Image.fromarray(
        apply_colormap(positive_map, color_map=ColorMap.HOT), mode="RGB"
    )
    neg_rgb = Image.fromarray(
        apply_colormap(negative_map, color_map=ColorMap.VIRIDIS), mode="RGB"
    )
    pos_rgb = _label_banner(pos_rgb, "Positive contribution")
    neg_rgb = _label_banner(neg_rgb, "Negative contribution")

    signed_path = out / "shap_signed_contributions.png"
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(pos_rgb)
    axes[1].set_title("Positive SHAP")
    axes[2].imshow(neg_rgb)
    axes[2].set_title("Negative SHAP")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("SHAP feature contribution (not attention)", fontsize=10)
    fig.tight_layout()
    fig.savefig(signed_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote SHAP visualizations under %s", out)
    return {
        "heatmap": heat_path,
        "overlay": overlay_path,
        "comparison": comparison_path,
        "signed": signed_path,
    }
