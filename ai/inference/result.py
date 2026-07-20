"""InvestigationResult dataclass and JSON serialization.

Purpose
-------
Canonical investigator-facing prediction payload for MAYA inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class InvestigationResult:
    """Structured single-image investigation outcome."""

    investigation_id: str
    prediction: str
    confidence: float
    confidence_level: str
    real_probability: float
    fake_probability: float
    threshold: float
    model_name: str
    model_version: str
    dataset_version: str
    prediction_time_ms: float
    timestamp: str
    image_name: str
    image_size: tuple[int, int]
    processing_device: str
    processing_status: str
    threshold_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
