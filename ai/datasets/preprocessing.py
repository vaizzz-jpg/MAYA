"""Image preprocessing for the processed corpus.

Purpose:
    Write RGB 224×224 derivatives while leaving the raw dataset untouched.
    Does NOT apply mean/std normalization (that stays in PyTorch transforms).
"""

from __future__ import annotations

import logging

from PIL import Image, ImageOps, UnidentifiedImageError

from ai.datasets.dataset_config import DatasetConfig
from ai.datasets.sampling import SplitAssignment
from ai.datasets.utils.fs import unique_destination

logger = logging.getLogger("maya.datasets.preprocessing")


def resize_preserve_aspect_rgb(image: Image.Image, size: int) -> Image.Image:
    """Convert to RGB, scale preserving aspect ratio, center-crop to ``size``.

    Flow:
        1. RGB conversion
        2. Resize so the shorter side equals ``size``
        3. Center-crop to ``size × size``
    """

    rgb = image.convert("RGB")
    width, height = rgb.size
    if width == 0 or height == 0:
        raise ValueError("Invalid image dimensions")

    if width < height:
        new_w = size
        new_h = max(size, int(round(height * (size / width))))
    else:
        new_h = size
        new_w = max(size, int(round(width * (size / height))))

    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return ImageOps.fit(
        resized,
        (size, size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )


def materialize_processed_split(
    assignment: SplitAssignment,
    config: DatasetConfig,
) -> tuple[int, int]:
    """Preprocess and write one split/class assignment.

    Returns:
        (written_count, skipped_count)
    """

    target_dir = config.class_dir(assignment.split, assignment.class_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for source in assignment.paths:
        try:
            with Image.open(source) as img:
                processed = resize_preserve_aspect_rgb(img, config.image_size)
                # Standardize extension to JPEG for compact processed corpus
                destination = unique_destination(target_dir, source.with_suffix(".jpg"))
                processed.save(destination, format="JPEG", quality=95, optimize=True)
            written += 1
        except (UnidentifiedImageError, OSError, ValueError, PermissionError) as exc:
            skipped += 1
            logger.warning("Skip preprocessing %s: %s", source, exc)
            continue

    logger.info(
        "Processed %s/%s → wrote=%s skipped=%s dir=%s",
        assignment.split.value,
        assignment.class_name.value,
        written,
        skipped,
        target_dir,
    )
    return written, skipped


def materialize_processed_dataset(
    assignments: list[SplitAssignment],
    config: DatasetConfig,
) -> dict[str, int]:
    """Create the full processed corpus from split assignments."""

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    totals = {"written": 0, "skipped": 0}
    for assignment in assignments:
        written, skipped = materialize_processed_split(assignment, config)
        totals["written"] += written
        totals["skipped"] += skipped
    return totals
