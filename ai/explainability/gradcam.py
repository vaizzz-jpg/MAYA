"""Backward-compatible Grad-CAM import path (Sprint 4.1).

Canonical implementation lives in ``ai.explainability.explainers.gradcam``.
"""

from __future__ import annotations

from ai.explainability.explainers.gradcam import GradCAM, compute_gradcam_map

__all__ = ["GradCAM", "compute_gradcam_map"]
