"""Shared perturbation helpers for faithfulness and counterfactual modules."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter


def importance_ranking(importance_map: np.ndarray) -> np.ndarray:
    """Return flat indices sorted by importance descending."""

    arr = np.asarray(importance_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"importance_map must be 2-D, got {arr.shape}")
    return np.argsort(arr.ravel())[::-1]


def build_cumulative_mask(
    shape: tuple[int, int],
    ranked_indices: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """Boolean HxW mask selecting the top ``count`` ranked pixels."""

    h, w = shape
    mask = np.zeros(h * w, dtype=bool)
    n = max(0, min(int(count), ranked_indices.size))
    if n:
        mask[ranked_indices[:n]] = True
    return mask.reshape(h, w)


def perturb_tensor(
    tensor: torch.Tensor,
    mask: np.ndarray,
    *,
    method: str = "mask",
    blur_radius: int = 8,
    neutral_value: float = 0.0,
) -> torch.Tensor:
    """Perturb ImageNet-normalized tensor pixels where ``mask`` is True.

    Methods:
    - mask: set selected pixels to ``neutral_value``
    - neutral: same as mask (explicit naming)
    - blur: approximate blur on selected region via PIL on denormalized RGB
    """

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected tensor (1,C,H,W), got {tuple(tensor.shape)}")
    method = method.lower().strip()
    if method not in {"mask", "blur", "neutral"}:
        raise ValueError(f"Unsupported perturbation method: {method}")

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != tuple(tensor.shape[-2:]):
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match tensor "
            f"{tuple(tensor.shape[-2:])}"
        )

    out = tensor.detach().clone()
    if method in {"mask", "neutral"} or not mask_bool.any():
        mask_t = torch.from_numpy(mask_bool).to(device=out.device)
        out[:, :, mask_t] = float(neutral_value)
        return out

    # blur path: denorm → PIL blur → renorm, then keep only masked pixels
    from ai.datasets.dataset_config import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    denorm = (out * std + mean).clamp(0.0, 1.0)
    rgb = (denorm[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=max(1, int(blur_radius))))
    blur_arr = np.asarray(blurred, dtype=np.float32) / 255.0
    blur_t = torch.from_numpy(blur_arr).permute(2, 0, 1).unsqueeze(0).to(
        device=out.device, dtype=out.dtype
    )
    blur_norm = (blur_t - mean) / std
    mask_t = torch.from_numpy(mask_bool).to(device=out.device)
    out[:, :, mask_t] = blur_norm[:, :, mask_t]
    return out


def important_region_mask(
    importance_map: np.ndarray,
    *,
    threshold: float = 0.35,
) -> np.ndarray:
    """Boolean mask of pixels at/above an importance threshold."""

    arr = np.asarray(importance_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"importance_map must be 2-D, got {arr.shape}")
    if arr.max() - arr.min() > 1e-8:
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        norm = np.zeros_like(arr)
    mask = norm >= float(threshold)
    if not mask.any():
        # Fall back to top 10% so experiments remain measurable
        k = max(1, int(0.10 * arr.size))
        ranked = importance_ranking(norm)
        mask = build_cumulative_mask(arr.shape, ranked, count=k)
    return mask
