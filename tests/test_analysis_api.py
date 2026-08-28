"""Analysis API + product integration tests (AI mocked)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = ROOT
    application.config["SECRET_KEY"] = "test-secret-key"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_investigation():
    from ai.inference.result import InvestigationResult

    return InvestigationResult(
        investigation_id="INV-2026-009999",
        prediction="REAL",
        confidence=68.1,
        confidence_level="Low",
        real_probability=0.681,
        fake_probability=0.319,
        threshold=0.5,
        model_name="efficientnet_b0",
        model_version="sprint3.2-best",
        dataset_version="MAYA-Kaggle-Balanced-v1",
        prediction_time_ms=12.0,
        timestamp="t",
        image_name="sample.png",
        image_size=(224, 224),
        processing_device="CPU",
        processing_status="success",
    )


def test_product_flow_with_mocked_ai(client, tmp_path: Path) -> None:
    register_and_login(client, username="analyst")
    case = client.post("/api/cases", json={"title": "AI Case"}).get_json()["data"]
    case_id = case["case_id"]
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "face.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 201
    evidence_id = up.get_json()["data"]["evidence_id"]

    expl = {
        "explainer": "gradcam",
        "heatmap": str(tmp_path / "heatmap.png"),
        "overlay": str(tmp_path / "overlay.png"),
        "explanation_json": str(tmp_path / "explanation_result.json"),
        "prediction": "REAL",
        "confidence": 68.1,
        "target_class": 0,
    }

    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ) as mock_inf, patch(
        "backend.app.services.analysis_service.run_explanation",
        return_value=expl,
    ) as mock_xai:
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "explainer": "gradcam"},
        )
        assert resp.status_code == 201, resp.get_json()
        mock_inf.assert_called_once()
        mock_xai.assert_called_once()

    data = resp.get_json()["data"]
    assert data["investigation_id"] == "INV-2026-009999"
    assert data["prediction"] == "REAL"
    assert data["confidence"] == pytest.approx(68.1)
    assert data["analysis_status"] == "COMPLETED"
    assert data["explanation"]["explainer"] == "gradcam"
    assert data["explanation"]["overlay"]

    got = client.get(f"/api/analysis/{data['analysis_id']}")
    assert got.status_code == 200
    assert got.get_json()["data"]["investigation_id"] == "INV-2026-009999"

    audit = client.get("/api/audit")
    assert audit.status_code == 200
    events = {e["event_type"] for e in audit.get_json()["data"]}
    assert "ANALYSIS_COMPLETED" in events
    assert "EVIDENCE_UPLOADED" in events


def test_analysis_failure_persisted(client) -> None:
    register_and_login(client, username="failuser")
    case_id = client.post("/api/cases", json={"title": "Fail"}).get_json()["data"][
        "case_id"
    ]
    evidence_id = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "x.png", "image/png")},
        content_type="multipart/form-data",
    ).get_json()["data"]["evidence_id"]

    with patch(
        "backend.app.services.analysis_service.run_inference",
        side_effect=RuntimeError("boom"),
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 500 or resp.status_code >= 400
    # Evidence should be marked failed after commit inside service
    meta = client.get(f"/api/evidence/{evidence_id}")
    assert meta.get_json()["data"]["analysis_status"] == "FAILED"
