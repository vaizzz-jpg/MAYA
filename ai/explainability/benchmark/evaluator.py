"""Per-explainer evaluation: generate map + measure performance + quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ai.explainability.benchmark.performance import (
    PerformanceMetrics,
    measure_explainer_performance,
)
from ai.explainability.benchmark.quality import QualityBundle, evaluate_heatmap_quality
from ai.explainability.heatmap import prepare_heatmap
from ai.explainability.registry import ExplainerRegistry
from ai.explainability.target_layer import TargetLayerResolver

logger = logging.getLogger("maya.ai.explainability.benchmark.evaluator")


@dataclass
class ExplainerEvalResult:
    """Raw evaluation outcome for a single registered explainer."""

    explainer_name: str
    heatmap: np.ndarray
    target_layer: str
    target_class: int
    performance: PerformanceMetrics
    quality: QualityBundle
    logits: np.ndarray
    probabilities: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "explainer_name": self.explainer_name,
            "target_layer": self.target_layer,
            "target_class": self.target_class,
            "performance": self.performance.to_dict(),
            "quality": self.quality.to_dict(),
            "probabilities": self.probabilities.tolist(),
        }


def evaluate_explainer(
    name: str,
    model: nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
    *,
    model_name: str,
    coverage_threshold: float = 0.2,
    output_resolution: int = 224,
    target_layer_override: str | None = None,
    target_class: int | None = None,
    warmup_runs: int = 0,
    timed_runs: int = 1,
    scorecam_batch_size: int = 8,
) -> ExplainerEvalResult:
    """Run one registry explainer and collect performance + quality metrics."""

    layer, layer_name = TargetLayerResolver.resolve(
        model, model_name, override=target_layer_override
    )
    explainer = ExplainerRegistry.create(name, model, device)
    if name == "scorecam" and hasattr(explainer, "batch_size"):
        explainer.batch_size = scorecam_batch_size  # type: ignore[attr-defined]

    def _generate():
        return explainer.generate(
            input_tensor,
            target_class=target_class,
            target_layer=layer,
            target_layer_name=layer_name,
        )

    cam, perf = measure_explainer_performance(
        _generate, warmup_runs=warmup_runs, timed_runs=timed_runs
    )
    heatmap = prepare_heatmap(
        cam.raw_heatmap,
        (output_resolution, output_resolution),
    )
    quality = evaluate_heatmap_quality(
        heatmap, coverage_threshold=coverage_threshold
    )
    logger.info("Evaluated explainer=%s time=%.1f ms", name, perf.generation_time_ms)
    return ExplainerEvalResult(
        explainer_name=name,
        heatmap=heatmap,
        target_layer=cam.target_layer_name,
        target_class=cam.target_class,
        performance=perf,
        quality=quality,
        logits=cam.logits,
        probabilities=cam.probabilities,
    )
