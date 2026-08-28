"""Fused explanation and trust result dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExplanationTrustResult:
    """MAYA Explanation Trust Score — engineering composite, not a probability."""

    trust_score: float  # 0–100
    grade: str
    components: dict[str, float]
    weights_used: dict[str, float]
    missing_components: list[str] = field(default_factory=list)
    methodology_note: str = (
        "MAYA Explanation Trust Score summarizes measured explanation "
        "characteristics with configurable weights. It is not a probability "
        "that the explanation is scientifically true."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdvancedInvestigatorSummary:
    """Plain-language summary across CAM / SHAP / faithfulness / counterfactual."""

    prediction: str
    model_confidence: float
    primary_visual_evidence: str
    shap_contribution: str
    faithfulness: str
    counterfactual_sensitivity: str
    explanation_assessment: str
    caveats: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Advanced XAI Investigator Summary",
                "",
                f"**Prediction:** {self.prediction}",
                f"**Model confidence:** {self.model_confidence:.1f}%",
                "",
                "## Primary visual evidence",
                self.primary_visual_evidence,
                "",
                "## SHAP contribution",
                self.shap_contribution,
                "",
                "## Faithfulness",
                self.faithfulness,
                "",
                "## Counterfactual sensitivity",
                self.counterfactual_sensitivity,
                "",
                "## Explanation assessment",
                self.explanation_assessment,
                "",
                "## Caveats",
                self.caveats,
                "",
            ]
        )


@dataclass
class FusedExplanationResult:
    """Combined Sprint 4.5 advanced explanation package."""

    investigation_id: str
    image_name: str
    prediction: str
    confidence: float
    trust: ExplanationTrustResult
    summary: AdvancedInvestigatorSummary
    visual_evidence: dict[str, Any] = field(default_factory=dict)
    shap: dict[str, Any] | None = None
    faithfulness: dict[str, Any] | None = None
    counterfactual: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "image_name": self.image_name,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "trust": self.trust.to_dict(),
            "summary": self.summary.to_dict(),
            "visual_evidence": dict(self.visual_evidence),
            "shap": self.shap,
            "faithfulness": self.faithfulness,
            "counterfactual": self.counterfactual,
            "artifact_paths": dict(self.artifact_paths),
            "timestamp": self.timestamp,
        }

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
