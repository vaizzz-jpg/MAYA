"""Image preprocessing for MAYA investigation inference.

Purpose
-------
Safely load an evidence image, resize to the project size, convert to an
ImageNet-normalized tensor ready for the classifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from ai.datasets.dataset_config import IMAGENET_MEAN, IMAGENET_STD
from ai.inference.inference_config import InferenceConfig, get_inference_config

logger = logging.getLogger("maya.ai.inference.preprocessing")


@dataclass
class PreprocessedImage:
    """Tensor plus lightweight metadata (no pixel copies beyond the tensor)."""

    tensor: torch.Tensor  # shape (1, 3, H, W)
    image_name: str
    image_size: tuple[int, int]  # (width, height) after resize


class ImagePreprocessError(RuntimeError):
    """Raised when an evidence image cannot be prepared for inference."""


def build_infer_transform(image_size: int) -> transforms.Compose:
    """Match training/eval normalization (Resize → Tensor → ImageNet Normalize)."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(
    path: Path | str,
    config: InferenceConfig | None = None,
) -> PreprocessedImage:
    """Load and preprocess a single image for inference.

    Raises:
        ImagePreprocessError: unsupported extension, corrupt/unreadable file.
    """

    cfg = config or get_inference_config()
    path = Path(path)
    if path.suffix.lower() not in {e.lower() for e in cfg.supported_extensions}:
        raise ImagePreprocessError(
            f"Unsupported image format '{path.suffix}' for {path.name}"
        )
    if not path.exists():
        raise ImagePreprocessError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            transform = build_infer_transform(cfg.image_size)
            tensor = transform(rgb).unsqueeze(0)
            size = (cfg.image_size, cfg.image_size)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Failed to preprocess %s: %s", path, exc)
        raise ImagePreprocessError(f"Unreadable or corrupted image: {path}") from exc

    logger.info("Preprocessed %s → tensor %s", path.name, tuple(tensor.shape))
    return PreprocessedImage(tensor=tensor, image_name=path.name, image_size=size)
