"""Faithfulness evaluation results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FaithfulnessStep:
    """One deletion / occlusion step measurement."""

    step: int
    fraction_masked: float
    important_score: float
    unimportant_score: float
    important_drop: float
    unimportant_drop: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Measured model sensitivity to highlighted regions."""

    investigation_id: str
    image_name: str
    prediction: str
    original_confidence: float
    original_score: float
    target_class: int
    perturbation_method: str
    steps: int
    deletion_score: float
    insertion_proxy_score: float
    faithfulness_score: float
    confidence_drop_important: float
    confidence_drop_unimportant: float
    step_results: list[FaithfulnessStep] = field(default_factory=list)
    visualization_path: str = ""
    generation_time_ms: float = 0.0
    timestamp: str = ""
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

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
