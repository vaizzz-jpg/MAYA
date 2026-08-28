"""Case management API tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    application.config["SECRET_KEY"] = "test-secret-key"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_case_crud_and_close(client) -> None:
    register_and_login(client, username="caseowner")
    created = client.post(
        "/api/cases",
        json={"title": "Case Alpha", "description": "desc", "priority": "HIGH"},
    )
    assert created.status_code == 201
    case = created.get_json()["data"]
    case_id = case["case_id"]
    assert case["case_number"].startswith("CASE-")
    assert case["status"] == "OPEN"

    listed = client.get("/api/cases")
    assert listed.status_code == 200
    assert len(listed.get_json()["data"]) == 1

    got = client.get(f"/api/cases/{case_id}")
    assert got.status_code == 200
    assert got.get_json()["data"]["title"] == "Case Alpha"

    patched = client.patch(
        f"/api/cases/{case_id}",
        json={"title": "Case Alpha Updated", "status": "IN_PROGRESS"},
    )
    assert patched.status_code == 200
    assert patched.get_json()["data"]["title"] == "Case Alpha Updated"
    assert patched.get_json()["data"]["status"] == "IN_PROGRESS"

    closed = client.post(f"/api/cases/{case_id}/close")
    assert closed.status_code == 200
    assert closed.get_json()["data"]["status"] == "CLOSED"
    assert closed.get_json()["data"]["closed_at"] is not None


def test_unauthorized_case_access(client) -> None:
    register_and_login(client, username="owner_a")
    created = client.post("/api/cases", json={"title": "Private"})
    case_id = created.get_json()["data"]["case_id"]
    client.post("/api/auth/logout")

    register_and_login(client, username="owner_b", email="owner_b@maya.test")
    denied = client.get(f"/api/cases/{case_id}")
    assert denied.status_code == 403
    listed = client.get("/api/cases")
    assert listed.get_json()["data"] == []
