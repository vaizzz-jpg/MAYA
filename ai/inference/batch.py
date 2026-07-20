"""Batch-ready investigation runners (single image or folder).

Purpose
-------
Provide a future-proof entry API for one image or a folder of evidence files
while reusing a shared ``InferencePipeline`` (and cached model).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai.inference.inference_config import InferenceConfig, get_inference_config
from ai.inference.pipeline import InferencePipeline
from ai.inference.preprocessing import ImagePreprocessError
from ai.inference.result import InvestigationResult
from ai.inference.utils import list_images

logger = logging.getLogger("maya.ai.inference.batch")


class InvestigationBatchRunner:
    """Run inference on a single path or every image in a folder."""

    def __init__(
        self,
        config: InferenceConfig | None = None,
        *,
        pipeline: InferencePipeline | None = None,
    ) -> None:
        self.config = config or get_inference_config()
        self.pipeline = pipeline or InferencePipeline(self.config)

    def run_one(self, image_path: Path | str) -> InvestigationResult:
        return self.pipeline.run(image_path)

    def run_folder(self, folder: Path | str) -> list[InvestigationResult]:
        """Process all supported images in ``folder`` (non-recursive).

        Failures on individual files are logged and skipped so a corrupted
        sample does not abort the remainder of the batch.
        """

        images = list_images(Path(folder), self.config.supported_extensions)
        results: list[InvestigationResult] = []
        for path in images:
            try:
                results.append(self.pipeline.run(path))
            except (ImagePreprocessError, OSError, ValueError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
        logger.info("Folder inference complete ok=%s / total=%s", len(results), len(images))
        return results
