"""Single-image investigation inference pipeline (orchestration only).

Purpose
-------
Wire model load → preprocess → predict → confidence → InvestigationResult →
artefacts without embedding those concerns inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai.engine.experiment import read_dataset_version
from ai.inference.artifacts import write_inference_artifacts
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig, get_inference_config
from ai.inference.investigation_id import InvestigationIDGenerator
from ai.inference.model_loader import ModelLoader
from ai.inference.predictor import predict_probabilities
from ai.inference.preprocessing import ImagePreprocessError, preprocess_image
from ai.inference.result import InvestigationResult
from ai.inference.utils import ensure_file, timing_ms

logger = logging.getLogger("maya.ai.inference.pipeline")


class InferencePipeline:
    """Reusable end-to-end inference coordinator with cached model loading."""

    def __init__(
        self,
        config: InferenceConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.config = config or get_inference_config()
        self.loader = model_loader or ModelLoader(self.config)
        self.id_gen = InvestigationIDGenerator(self.config.id_state_path)

    def run(self, image_path: Path | str) -> InvestigationResult:
        """Execute the full investigation workflow for one image."""

        path = ensure_file(Path(image_path))
        logger.info("Inference pipeline start image=%s", path)

        loaded = self.loader.load()
        status = "success"
        try:
            with timing_ms() as timer:
                prepared = preprocess_image(path, self.config)
                probs = predict_probabilities(
                    loaded.model, prepared.tensor, loaded.device
                )
                # Single-image batch → first row
                row = probs[0]
                decision = decide_from_probabilities(row, self.config)
                # Release batch tensor promptly
                del prepared.tensor
                del probs
        except ImagePreprocessError as exc:
            status = "failed"
            logger.error("Preprocess failed: %s", exc)
            raise
        except Exception:
            status = "failed"
            logger.exception("Inference failed for %s", path)
            raise

        result = InvestigationResult(
            investigation_id=self.id_gen.next_id(),
            prediction=decision.predicted_label,
            confidence=decision.confidence,
            confidence_level=decision.confidence_level,
            real_probability=decision.real_probability,
            fake_probability=decision.fake_probability,
            threshold=decision.threshold,
            model_name=loaded.model_name,
            model_version=loaded.model_version,
            dataset_version=read_dataset_version(self.config.project_root),
            prediction_time_ms=round(timer["ms"], 3),
            timestamp=InvestigationResult.utc_now(),
            image_name=path.name,
            image_size=prepared.image_size,
            processing_device=str(loaded.device),
            processing_status=status,
            threshold_decision=decision.threshold_decision,
        )

        write_inference_artifacts(result, self.config.artifact_dir)
        logger.info(
            "Pipeline complete id=%s prediction=%s conf=%.2f%%",
            result.investigation_id,
            result.prediction,
            result.confidence,
        )
        return result
