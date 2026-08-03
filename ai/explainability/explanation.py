"""Explainability result dataclasses and report serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExplanationResult:
    """Structured investigator-facing explanation outcome."""

    investigation_id: str
    explainer_name: str
    model_name: str
    model_version: str
    dataset_version: str
    prediction: str
    confidence: float
    target_class: int
    target_layer: str
    generation_time_ms: float
    device: str
    heatmap_path: str
    overlay_path: str
    timestamp: str
    comparison_path: str = ""
    original_path: str = ""
    image_name: str = ""
    real_probability: float = 0.0
    fake_probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def to_explanation_json_payload(self) -> dict[str, Any]:
        """Compact artefact schema matching Sprint 4.1 contract."""

        return {
            "investigation_id": self.investigation_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "explainer": self.explainer_name,
            "target_layer": self.target_layer,
            "generation_time_ms": self.generation_time_ms,
            "device": self.device,
            "heatmap": self.heatmap_path,
            "overlay": self.overlay_path,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


def write_explanation_json(result: ExplanationResult, path: Path) -> Path:
    """Write the compact ``explanation.json`` artefact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_explanation_json_payload(), indent=2),
        encoding="utf-8",
    )
    return path


def write_explanation_report(result: ExplanationResult, path: Path) -> Path:
    """Write investigator-facing ``explanation_report.md``."""

    lines = [
        "# MAYA Explanation Report",
        "",
        f"- Investigation ID: **{result.investigation_id}**",
        f"- Image: `{result.image_name}`",
        f"- Prediction: **{result.prediction}**",
        f"- Confidence: **{result.confidence:.2f}%**",
        f"- REAL probability: `{result.real_probability:.6f}`",
        f"- FAKE probability: `{result.fake_probability:.6f}`",
        f"- Explainer: `{result.explainer_name}`",
        f"- Target Layer: `{result.target_layer}`",
        f"- Target Class Index: `{result.target_class}`",
        f"- Generation Time: `{result.generation_time_ms:.2f} ms`",
        f"- Model: `{result.model_name}` / `{result.model_version}`",
        f"- Dataset Version: `{result.dataset_version}`",
        f"- Device: `{result.device}`",
        f"- Timestamp (UTC): `{result.timestamp}`",
        "",
        "## Generated Files",
        "",
        f"- Original: `{result.original_path}`",
        f"- Heatmap: `{result.heatmap_path}`",
        f"- Overlay: `{result.overlay_path}`",
        f"- Comparison: `{result.comparison_path}`",
        "",
        "## Investigator Notes",
        "",
        "Grad-CAM highlights spatial regions that most influenced the classifier",
        "score for the predicted class. Treat the map as an **indicator**, not",
        "legal certainty. Review the overlay beside the original evidence image.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_pipeline_summary(path: Path) -> Path:
    """Write ``pipeline_summary.md`` describing the explainability pipeline."""

    lines = [
        "# Explainability Pipeline Summary",
        "",
        "1. Validate input image",
        "2. Load / reuse cached model (eval mode)",
        "3. Preprocess (resize + ImageNet normalize)",
        "4. Resolve target layer (or apply override)",
        "5. Load explainer from registry (Grad-CAM)",
        "6. Forward + backward with activation/gradient hooks",
        "7. Normalize / resize heatmap",
        "8. Overlay heatmap onto original",
        "9. Persist visualization artefacts + JSON / Markdown reports",
        "10. Remove hooks and release tensors",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
