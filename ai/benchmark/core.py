"""Shared core inference helper for MAYA benchmarks (no artefact side-effects).

Purpose
-------
Reuse Sprint 4 model_loader / preprocessing / predictor / confidence without
writing investigation IDs or JSON on every timed run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ai.inference.confidence import ConfidenceDecision, decide_from_probabilities
from ai.inference.inference_config import InferenceConfig, get_inference_config
from ai.inference.model_loader import LoadedModel, ModelLoader
from ai.inference.predictor import predict_probabilities
from ai.inference.preprocessing import preprocess_image
from ai.inference.utils import timing_ms

logger = logging.getLogger("maya.ai.benchmark.core")


@dataclass
class CoreInferenceOutcome:
    """Lightweight outcome of one timed core inference."""

    decision: ConfidenceDecision
    latency_ms: float


class CoreInferencer:
    """Cached-model inferencer for latency / consistency / throughput tests."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or get_inference_config()
        self.loader = ModelLoader(self.config)
        self._loaded: LoadedModel | None = None

    def ensure_loaded(self) -> LoadedModel:
        if self._loaded is None:
            self._loaded = self.loader.load()
            logger.info("Benchmark model ready on %s", self._loaded.device)
        return self._loaded

    def infer_once(self, image_path: Path) -> CoreInferenceOutcome:
        """Run preprocess → predict → decide; return decision + wall latency."""

        loaded = self.ensure_loaded()
        with timing_ms() as timer:
            prepared = preprocess_image(image_path, self.config)
            probs = predict_probabilities(loaded.model, prepared.tensor, loaded.device)
            decision = decide_from_probabilities(probs[0], self.config)
            del prepared.tensor, probs
        return CoreInferenceOutcome(decision=decision, latency_ms=float(timer["ms"]))
