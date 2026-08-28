"""Structured SHAP attribution results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ShapAttributionStats:
    """Summary statistics over a SHAP attribution map."""

    mean_abs: float
    max_abs: float
    mean_positive: float
    mean_negative: float
    positive_fraction: float
    negative_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShapResult:
    """Investigator-facing SHAP attribution outcome."""

    investigation_id: str
    explainer_name: str
    model_name: str
    model_version: str
    dataset_version: str
    prediction: str
    confidence: float
    target_class: int
    attribution_stats: ShapAttributionStats
    generation_time_ms: float
    device: str
    visualization_path: str
    timestamp: str
    heatmap_path: str = ""
    overlay_path: str = ""
    comparison_path: str = ""
    image_name: str = ""
    real_probability: float = 0.0
    fake_probability: float = 0.0
    max_evaluations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

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
