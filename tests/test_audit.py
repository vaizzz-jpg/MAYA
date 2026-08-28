"""Tests for MAYA Phase 4 Sprint 4.5 XAI audit trail."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.fusion.audit import XaiAuditRecord, build_audit_record


def test_audit_required_metadata(tmp_path: Path) -> None:
    record = build_audit_record(
        investigation_id="INV-2026-000001",
        image_identifier="evidence.jpg",
        model_name="efficientnet_b0",
        model_version="sprint3.2-best",
        dataset_version="MAYA-Kaggle-Balanced-v1",
        configuration={"shap": {"max_evaluations": 32}},
        device="CPU",
        generation_time_ms=123.4,
        target_class=1,
        target_layer="features.8",
        stages_completed=["shap", "faithfulness"],
        output_artifact_paths={"shap_overlay": "a.png"},
    )
    assert record.investigation_id.startswith("INV-")
    assert record.generation_timestamp
    assert "shap" in record.configuration
    assert "torch" in record.software_versions
    assert record.stages_completed == ["shap", "faithfulness"]

    path = record.save_json(tmp_path / "xai_audit.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "efficientnet_b0"
    assert payload["device"] == "CPU"
    assert "SHA-256" not in record.notes or "deferred" in record.notes.lower()


def test_audit_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        build_audit_record(
            investigation_id="",
            image_identifier="x.png",
            model_name="m",
            model_version="v",
            dataset_version="d",
            configuration={},
            device="CPU",
            generation_time_ms=1.0,
        )


def test_audit_json_roundtrip() -> None:
    record = XaiAuditRecord(
        investigation_id="INV-1",
        image_identifier="a.png",
        model_name="m",
        model_version="v",
        dataset_version="d",
        explainer="advanced_xai",
        configuration={"a": 1},
        target_class=0,
        target_layer=None,
        generation_timestamp="t",
        device="CPU",
        generation_time_ms=1.0,
    )
    data = json.loads(record.to_json())
    assert data["explainer"] == "advanced_xai"
