"""XAI audit trail for Sprint 4.5 (no SHA-256 in this sprint)."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("maya.ai.explainability.fusion.audit")


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for name in ("torch", "numpy", "PIL", "shap", "matplotlib"):
        try:
            if name == "PIL":
                import PIL

                versions["pillow"] = getattr(PIL, "__version__", "unknown")
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name if name != "PIL" else "pillow"] = "unavailable"
    return versions


@dataclass
class XaiAuditRecord:
    """Reproducibility metadata for an advanced explanation run."""

    investigation_id: str
    image_identifier: str
    model_name: str
    model_version: str
    dataset_version: str
    explainer: str
    configuration: dict[str, Any]
    target_class: int | None
    target_layer: str | None
    generation_timestamp: str
    device: str
    generation_time_ms: float
    output_artifact_paths: dict[str, str] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=dict)
    stages_completed: list[str] = field(default_factory=list)
    notes: str = (
        "Audit trail for reproducibility. Digital integrity hashing "
        "(e.g. SHA-256) is deferred to a later phase."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Wrote XAI audit → %s", path)
        return path

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


def build_audit_record(
    *,
    investigation_id: str,
    image_identifier: str,
    model_name: str,
    model_version: str,
    dataset_version: str,
    configuration: dict[str, Any],
    device: str,
    generation_time_ms: float,
    target_class: int | None = None,
    target_layer: str | None = None,
    explainer: str = "advanced_xai",
    output_artifact_paths: dict[str, str] | None = None,
    stages_completed: list[str] | None = None,
    timestamp: str | None = None,
) -> XaiAuditRecord:
    """Create a structured audit record capturing run metadata."""

    required = {
        "investigation_id": investigation_id,
        "image_identifier": image_identifier,
        "model_name": model_name,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "device": device,
    }
    for key, value in required.items():
        if value is None or (isinstance(value, str) and not str(value).strip()):
            raise ValueError(f"Audit record missing required field: {key}")

    return XaiAuditRecord(
        investigation_id=investigation_id,
        image_identifier=image_identifier,
        model_name=model_name,
        model_version=model_version,
        dataset_version=dataset_version,
        explainer=explainer,
        configuration=dict(configuration),
        target_class=target_class,
        target_layer=target_layer,
        generation_timestamp=timestamp or XaiAuditRecord.utc_now(),
        device=device,
        generation_time_ms=float(generation_time_ms),
        output_artifact_paths=dict(output_artifact_paths or {}),
        software_versions=_library_versions(),
        stages_completed=list(stages_completed or []),
    )
