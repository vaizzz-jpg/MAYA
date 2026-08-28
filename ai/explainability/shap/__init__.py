"""SHAP attribution engine for MAYA Phase 4 Sprint 4.5.

Provides feature-contribution estimates (not CAM activation maps).
Uses the maintained ``shap`` library — does not reimplement SHAP math.
Requires OpenCV (``opencv-python-headless``) for ``shap.maskers.Image``.
"""

from __future__ import annotations

from ai.explainability.shap.explainer import ShapEngine, ShapUnavailableError
from ai.explainability.shap.masker import (
    ShapDependencyError,
    require_opencv,
    require_shap,
    require_shap_image_stack,
)
from ai.explainability.shap.result import ShapAttributionStats, ShapResult

__all__ = [
    "ShapAttributionStats",
    "ShapDependencyError",
    "ShapEngine",
    "ShapResult",
    "ShapUnavailableError",
    "require_opencv",
    "require_shap",
    "require_shap_image_stack",
]
