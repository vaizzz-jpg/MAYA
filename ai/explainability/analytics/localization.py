"""Spatial localization metrics for explanation heatmaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ai.explainability.analytics.statistics import _as_2d_float


@dataclass(frozen=True)
class LocalizationMetrics:
    """Where attention concentrates on the evidence image."""

    bbox: tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max) inclusive
    coverage_percentage: float
    largest_region_area: int
    largest_region_fraction: float
    center_of_attention: tuple[float, float]  # (row, col) weighted centroid
    threshold_used: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["center_of_attention"] = list(self.center_of_attention)
        return data


def compute_localization_metrics(
    heatmap: np.ndarray,
    *,
    activation_threshold: float = 0.2,
) -> LocalizationMetrics:
    """Bounding box, connected region, and attention centroid."""

    arr = _as_2d_float(heatmap)
    height, width = arr.shape
    mask = arr >= activation_threshold
    coverage_pct = float(100.0 * mask.mean()) if mask.size else 0.0

    if not mask.any():
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        return LocalizationMetrics(
            bbox=(0, 0, max(height - 1, 0), max(width - 1, 0)),
            coverage_percentage=round(coverage_pct, 4),
            largest_region_area=0,
            largest_region_fraction=0.0,
            center_of_attention=(round(cy, 4), round(cx, 4)),
            threshold_used=float(activation_threshold),
        )

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    largest_area, largest_frac = _largest_connected_region(mask)
    center = _center_of_attention(arr)

    return LocalizationMetrics(
        bbox=bbox,
        coverage_percentage=round(coverage_pct, 4),
        largest_region_area=int(largest_area),
        largest_region_fraction=round(float(largest_frac), 6),
        center_of_attention=(round(center[0], 4), round(center[1], 4)),
        threshold_used=float(activation_threshold),
    )


def _center_of_attention(arr: np.ndarray) -> tuple[float, float]:
    total = float(arr.sum())
    if total <= 1e-12:
        h, w = arr.shape
        return ((h - 1) / 2.0, (w - 1) / 2.0)
    rows = np.arange(arr.shape[0], dtype=np.float64)[:, None]
    cols = np.arange(arr.shape[1], dtype=np.float64)[None, :]
    cy = float((arr * rows).sum() / total)
    cx = float((arr * cols).sum() / total)
    return (cy, cx)


def _largest_connected_region(mask: np.ndarray) -> tuple[int, float]:
    """4-connected component area via iterative flood fill (no SciPy required)."""

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = 0
    total_on = int(mask.sum())
    if total_on == 0:
        return 0, 0.0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            area = 0
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                area += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area > best:
                best = area

    return best, best / float(mask.size)
