"""Image masker helpers for SHAP PartitionExplainer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("maya.ai.explainability.shap.masker")

_INSTALL_HINT = (
    "Install MAYA XAI dependencies with: "
    "python -m pip install \"shap>=0.44.0\" \"opencv-python-headless>=4.8.0\" "
    "(from the active project virtualenv), or: python -m pip install -r requirements.txt"
)


class ShapDependencyError(RuntimeError):
    """Raised when SHAP image explanation dependencies are unavailable."""


def require_shap() -> Any:
    """Import and return the ``shap`` module, or raise a clear error."""

    try:
        import shap
    except ImportError as exc:
        raise ShapDependencyError(
            "The 'shap' package is required for SHAP explanations. " + _INSTALL_HINT
        ) from exc
    return shap


def require_opencv() -> Any:
    """Import OpenCV (``cv2``) required by ``shap.maskers.Image``.

    SHAP's blur/inpaint image maskers import OpenCV at masker construction time.
    MAYA uses ``opencv-python-headless`` (no GUI) for CPU/Windows environments.
    """

    try:
        import cv2
    except ImportError as exc:
        raise ShapDependencyError(
            "OpenCV (cv2) is required for SHAP image masking via "
            "shap.maskers.Image. MAYA expects opencv-python-headless. "
            + _INSTALL_HINT
        ) from exc
    return cv2


def require_shap_image_stack() -> Any:
    """Ensure both ``shap`` and OpenCV are available for image SHAP."""

    shap = require_shap()
    require_opencv()
    return shap


def build_image_masker(
    image_hwc: np.ndarray,
    *,
    blur_kernel: int = 32,
) -> Any:
    """Build a ``shap.maskers.Image`` for an H×W×C float image in [0, 1].

    Uses blur-based masking (CPU-friendly) rather than generative inpainting.
    Requires ``shap`` and OpenCV (``cv2``).
    """

    shap = require_shap_image_stack()
    arr = np.asarray(image_hwc, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] not in {1, 3, 4}:
        raise ValueError(
            f"Expected HWC image with 1/3/4 channels, got shape {arr.shape}"
        )
    # shap Image masker blur token: "blur(k,k)" — implemented via OpenCV
    k = max(1, int(blur_kernel))
    try:
        masker = shap.maskers.Image(f"blur({k},{k})", arr.shape)
    except ModuleNotFoundError as exc:
        # SHAP may raise ModuleNotFoundError for cv2 from inside its package
        missing = str(exc)
        if "cv2" in missing or "opencv" in missing.lower():
            raise ShapDependencyError(
                "OpenCV (cv2) is required for SHAP image masking via "
                "shap.maskers.Image. MAYA expects opencv-python-headless. "
                + _INSTALL_HINT
            ) from exc
        raise ShapDependencyError(
            f"SHAP image masker dependency missing: {exc}. " + _INSTALL_HINT
        ) from exc
    logger.debug("Built SHAP Image masker blur(%s,%s) for shape %s", k, k, arr.shape)
    return masker
