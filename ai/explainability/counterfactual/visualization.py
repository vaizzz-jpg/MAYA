"""Counterfactual comparison visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from ai.datasets.dataset_config import IMAGENET_MEAN, IMAGENET_STD
from ai.explainability.counterfactual.result import CounterfactualExperiment

logger = logging.getLogger("maya.ai.explainability.counterfactual.visualization")


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected (1,C,H,W), got {tuple(tensor.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(
        1, 3, 1, 1
    )
    denorm = (tensor * std + mean).clamp(0.0, 1.0)[0].detach().cpu().permute(1, 2, 0)
    rgb = (denorm.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def write_counterfactual_comparison(
    *,
    original_image: Image.Image,
    input_tensor: torch.Tensor,
    preview_tensors: dict[str, torch.Tensor],
    experiments: list[CounterfactualExperiment],
    artifact_dir: Path,
) -> Path:
    """Side-by-side original vs perturbed previews with confidence deltas."""

    if not experiments:
        raise ValueError(
            "Cannot generate counterfactual_comparison.png without experiments"
        )

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "counterfactual_comparison.png"

    panels = [("original", original_image.convert("RGB"), None)]
    by_method = {e.perturbation_strategy: e for e in experiments}
    for method, tensor in preview_tensors.items():
        exp = by_method.get(method)
        panels.append((method, _tensor_to_pil(tensor), exp))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (title, img, exp) in zip(axes, panels):
        ax.imshow(img)
        if exp is None:
            ax.set_title("Original")
        else:
            ax.set_title(
                f"{title}\n"
                f"{exp.perturbed_prediction} {exp.perturbed_confidence:.1f}%\n"
                f"Δconf {exp.confidence_delta:+.1f}pp"
            )
        ax.axis("off")
    fig.suptitle(
        "Model counterfactual perturbation experiments\n"
        "(sensitivity under controlled edits — not real-world counterfactuals)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote counterfactual comparison → %s", path)
    # Keep input_tensor referenced so callers know original path was used
    _ = input_tensor
    return path
