"""Counterfactual perturbation experiment results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CounterfactualExperiment:
    """One controlled model counterfactual perturbation experiment."""

    perturbation_strategy: str
    affected_region: str
    original_prediction: str
    original_confidence: float
    perturbed_prediction: str
    perturbed_confidence: float
    confidence_delta: float
    prediction_changed: bool
    original_score: float
    perturbed_score: float
    score_delta: float
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    """Bundle of model counterfactual perturbation experiments."""

    investigation_id: str
    image_name: str
    original_prediction: str
    original_confidence: float
    target_class: int
    experiments: list[CounterfactualExperiment] = field(default_factory=list)
    visualization_path: str = ""
    generation_time_ms: float = 0.0
    timestamp: str = ""
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Any) -> Any:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
