"""Genuine E2E backend test (non-mocked AI where available).

Covers: Register → Login → Create Case → Upload Evidence →
Analyze Evidence (real AI inference) → (optional XAI) → Database
persistence → Audit record → Artifact creation → Retrieve results.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = ROOT
    application.config["REPORT_DIR"] = tmp_path / "reports"
    application.config["REPORT_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["SECRET_KEY"] = "e2e-test-secret"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _png_rgb(size=(96, 96), color=(30, 100, 170)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_e2e_auth_case_evidence_analysis(client) -> None:
    """End-to-end flow with real EfficientNet-B0 inference."""
    user = register_and_login(client, username="e2einvestigator", email="e2e@maya.test")
    assert user["id"] and user["role"] == "INVESTIGATOR"

    created = client.post(
        "/api/cases",
        json={
            "title": "E2E Forensic Case",
            "description": "Automated E2E product validation.",
            "priority": "HIGH",
        },
    )
    assert created.status_code == 201, created.get_json()
    case = created.get_json()["data"]
    case_id = case["case_id"]
    assert case["case_number"].startswith("CASE-")
    assert case["status"] == "OPEN"

    png = _png_rgb()
    upload = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(png), "forensic_evidence.png", "image/png"),
              "notes": "E2E test artifact"},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201, upload.get_json()
    ev = upload.get_json()["data"]
    evidence_id = ev["evidence_id"]
    assert len(ev["sha256"]) == 64
    assert ev["status"] == "UPLOADED"

    listed = client.get(f"/api/evidence/cases/{case_id}")
    assert listed.status_code == 200
    assert len(listed.get_json()["data"]) == 1

    integrity = client.post(f"/api/evidence/{evidence_id}/verify-integrity")
    assert integrity.status_code == 200
    assert integrity.get_json()["data"]["integrity_status"] == "VALID"

    analyze = client.post(
        f"/api/evidence/{evidence_id}/analyze",
        json={
            "generate_explanation": False,
            "verify_before_analyze": True,
        },
    )
    assert analyze.status_code == 201, analyze.get_json()
    result = analyze.get_json()["data"]

    assert result["evidence_id"] == evidence_id
    assert result["case_id"] == case_id
    assert result["investigation_id"].startswith("INV-"), result
    assert result["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= float(result["confidence"]) <= 100.0
    assert result["analysis_status"] == "COMPLETED"
    assert result["model_name"] == "efficientnet_b0"
    assert result["investigation_id"].startswith("INV-")
    assert result["artifact_dir"]

    retrieved = client.get(f"/api/analysis/{result['analysis_id']}")
    assert retrieved.status_code == 200
    assert retrieved.get_json()["data"]["investigation_id"] == result["investigation_id"]

    audit_resp = client.get("/api/audit")
    assert audit_resp.status_code == 200
    events = audit_resp.get_json()["data"]
    event_types = [e["event_type"] for e in events]
    assert "USER_REGISTERED" in event_types
    assert "USER_LOGIN" in event_types
    assert "CASE_CREATED" in event_types
    assert "EVIDENCE_UPLOADED" in event_types
    assert "EVIDENCE_VERIFIED" in event_types
    assert "ANALYSIS_STARTED" in event_types
    assert "ANALYSIS_COMPLETED" in event_types
    for e in events:
        if e["event_type"] == "ANALYSIS_COMPLETED":
            details = e["details"] or {}
            assert "investigation_id" in details
            assert details["investigation_id"] == result["investigation_id"]

    artifact_dir = Path(result["artifact_dir"])
    assert artifact_dir.exists(), f"Missing artifact dir: {artifact_dir}"


def test_e2e_authorization_enforced(client) -> None:
    """Ensure cross-user access is blocked at every layer."""
    register_and_login(client, username="owner_a", email="a@maya.test")
    case_a = client.post("/api/cases", json={"title": "A's case"}).get_json()["data"]
    case_a_id = case_a["case_id"]
    png = _png_rgb()
    upload = client.post(
        f"/api/evidence/cases/{case_a_id}",
        data={"file": (io.BytesIO(png), "a.png", "image/png")},
        content_type="multipart/form-data",
    )
    ev_a_id = upload.get_json()["data"]["evidence_id"]
    analyze = client.post(
        f"/api/evidence/{ev_a_id}/analyze",
        json={"generate_explanation": False, "verify_before_analyze": False},
    )
    analysis_a_id = analyze.get_json()["data"]["analysis_id"]
    client.post("/api/auth/logout")

    register_and_login(client, username="owner_b", email="b@maya.test")
    assert client.get(f"/api/cases/{case_a_id}").status_code == 403
    assert client.get(f"/api/evidence/{ev_a_id}").status_code == 403
    assert client.get(f"/api/analysis/{analysis_a_id}").status_code == 403
    others_audit = client.get("/api/audit").get_json()["data"]
    for e in others_audit:
        assert e["user_id"] != 1


def test_e2e_integrity_modified_detection(client, app) -> None:
    """After writing tampered bytes, integrity check must report MODIFIED."""
    from backend.app.models import Evidence
    from backend.app.extensions import db
    from pathlib import Path as _P

    register_and_login(client, username="integrity_user", email="int@maya.test")
    case_id = client.post("/api/cases", json={"title": "Integrity"}).get_json()["data"]["case_id"]
    png = _png_rgb()
    ev_id = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(png), "clean.png", "image/png")},
        content_type="multipart/form-data",
    ).get_json()["data"]["evidence_id"]

    clean = client.post(f"/api/evidence/{ev_id}/verify-integrity")
    assert clean.get_json()["data"]["integrity_status"] == "VALID"

    with app.app_context():
        ev = db.session.get(Evidence, ev_id)
        full = _P(app.config["UPLOAD_DIR"]) / ev.storage_path
        full.write_bytes(b"\x00\x01\x02CORRUPT" + full.read_bytes())

    tampered = client.post(f"/api/evidence/{ev_id}/verify-integrity")
    assert tampered.status_code == 200
    assert tampered.get_json()["data"]["integrity_status"] == "MODIFIED"
