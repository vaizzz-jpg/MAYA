"""Process raw SHAP values into attribution maps and statistics."""

from __future__ import annotations

import logging

import numpy as np

from ai.explainability.heatmap import normalize_heatmap
from ai.explainability.shap.result import ShapAttributionStats

logger = logging.getLogger("maya.ai.explainability.shap.processor")


def extract_class_attribution(
    shap_values: np.ndarray,
    *,
    target_class: int,
) -> np.ndarray:
    """Reduce SHAP output to a single H×W attribution map for ``target_class``.

    Accepts common SHAP image output layouts:
    - (H, W, C)
    - (H, W, C, classes)
    - (1, H, W, C)
    - (1, H, W, C, classes)
    - list/tuple of per-class arrays
    """

    if isinstance(shap_values, (list, tuple)):
        if target_class < 0 or target_class >= len(shap_values):
            raise ValueError(
                f"target_class {target_class} out of range for "
                f"{len(shap_values)} SHAP outputs"
            )
        arr = np.asarray(shap_values[target_class], dtype=np.float64)
    else:
        arr = np.asarray(shap_values, dtype=np.float64)

    # Drop leading batch dim of 1
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4:
        # (H, W, C, classes) or (H, W, C, 1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            if target_class < 0 or target_class >= arr.shape[-1]:
                raise ValueError(
                    f"target_class {target_class} out of range for "
                    f"SHAP classes dim {arr.shape[-1]}"
                )
            arr = arr[..., target_class]

    if arr.ndim == 3:
        # Sum channel contributions → spatial map
        spatial = arr.sum(axis=-1)
    elif arr.ndim == 2:
        spatial = arr
    else:
        raise ValueError(
            f"Unsupported SHAP value shape after reduction: {arr.shape}"
        )

    return np.asarray(spatial, dtype=np.float32)


def split_signed_maps(
    attribution: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (abs_normalized, positive_normalized, negative_normalized)."""

    arr = np.asarray(attribution, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Attribution must be 2-D, got {arr.shape}")

    positive = np.clip(arr, 0.0, None)
    negative = np.clip(-arr, 0.0, None)
    abs_map = np.abs(arr)

    return (
        normalize_heatmap(abs_map, eps=eps),
        normalize_heatmap(positive, eps=eps),
        normalize_heatmap(negative, eps=eps),
    )


def compute_attribution_stats(attribution: np.ndarray) -> ShapAttributionStats:
    """Compute summary statistics over a signed attribution map."""

    arr = np.asarray(attribution, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty attribution map")
    abs_arr = np.abs(arr)
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    return ShapAttributionStats(
        mean_abs=round(float(abs_arr.mean()), 8),
        max_abs=round(float(abs_arr.max()), 8),
        mean_positive=round(float(pos.mean()) if pos.size else 0.0, 8),
        mean_negative=round(float(neg.mean()) if neg.size else 0.0, 8),
        positive_fraction=round(float((arr > 0).mean()), 6),
        negative_fraction=round(float((arr < 0).mean()), 6),
    )
