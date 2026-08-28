"""AI bridge — orchestrates existing inference / XAI without duplicating logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.explainability import ExplainabilityConfig, ExplainabilityEngine
from ai.inference import InferenceConfig, InferencePipeline
from ai.inference.result import InvestigationResult
from ai.models.model_config import _project_root

logger = logging.getLogger("maya.backend.integrations.ai")


@dataclass
class AiAnalysisBundle:
    investigation: InvestigationResult
    explanation: dict[str, Any] | None = None


_pipeline: InferencePipeline | None = None


def get_inference_pipeline() -> InferencePipeline:
    """Process-level cached pipeline (reuses ModelLoader cache)."""

    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline(
            InferenceConfig(
                project_root=_project_root(),
                device_preference="cpu",
            )
        )
    return _pipeline


def reset_inference_pipeline() -> None:
    """Test helper."""

    global _pipeline
    _pipeline = None


def run_inference(image_path: Path) -> InvestigationResult:
    pipeline = get_inference_pipeline()
    logger.info("Calling InferencePipeline.run path=%s", image_path)
    return pipeline.run(image_path)


def run_explanation(
    image_path: Path,
    *,
    investigation_id: str,
    explainer: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    cfg = ExplainabilityConfig(
        project_root=_project_root(),
        device_preference="cpu",
        explainer_name=explainer,
        artifact_dir=artifact_dir,
        checkpoint_path=_project_root() / "artifacts" / "checkpoints" / "best.pt",
    )
    engine = ExplainabilityEngine(cfg)
    # Reuse investigation ID from inference; isolate artefacts
    result = engine.explain(
        image_path,
        investigation_id=investigation_id,
        artifact_dir=artifact_dir,
    )
    return {
        "explainer": result.explainer_name,
        "heatmap": result.heatmap_path,
        "overlay": result.overlay_path,
        "comparison": result.comparison_path,
        "explanation_json": str(Path(artifact_dir) / "explanation_result.json"),
        "prediction": result.prediction,
        "confidence": result.confidence,
        "target_class": result.target_class,
    }
