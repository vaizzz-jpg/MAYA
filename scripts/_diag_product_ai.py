"""Quick diagnostic: real AI inference through product API."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import create_app
from backend.app.extensions import db


def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="maya-diag-")
    print(f"tmpdir = {tmpdir}")

    app = create_app("testing")
    app.config["UPLOAD_DIR"] = Path(tmpdir) / "uploads"
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["ROOT_DIR"] = ROOT
    app.config["SECRET_KEY"] = "test-secret"

    with app.app_context():
        db.create_all()
        client = app.test_client()

        r = client.post(
            "/api/auth/register",
            json={
                "email": "a@maya.test",
                "username": "analyst",
                "password": "securepass1",
            },
        )
        print("register:", r.status_code)
        assert r.status_code == 201, r.get_json()

        r = client.post(
            "/api/auth/login",
            json={"login": "analyst", "password": "securepass1"},
        )
        print("login:", r.status_code)
        assert r.status_code == 200

        r = client.post("/api/cases", json={"title": "Diagnostic Case"})
        print("case:", r.status_code)
        case_id = r.get_json()["data"]["case_id"]

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")
        png_bytes = buf.getvalue()
        data = {"file": (io.BytesIO(png_bytes), "sample.png", "image/png")}
        r = client.post(
            f"/api/evidence/cases/{case_id}",
            data=data,
            content_type="multipart/form-data",
        )
        print("evidence:", r.status_code)
        assert r.status_code == 201, r.get_json()
        evidence_id = r.get_json()["data"]["evidence_id"]
        sha = r.get_json()["data"]["sha256"]
        print(f"  sha256 = {sha[:16]}...")

        print("\n--- Running REAL inference (no explanation) ---")
        r = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={
                "generate_explanation": False,
                "verify_before_analyze": False,
            },
        )
        print("analyze status:", r.status_code)
        body = r.get_json()
        import json

        print(json.dumps(body, indent=2, default=str))

        if r.status_code == 201:
            data = body["data"]
            print("\n--- Inference success ---")
            print(f"  investigation_id: {data['investigation_id']}")
            print(f"  prediction:       {data['prediction']}")
            print(f"  confidence:       {data['confidence']}")
            print(f"  analysis_status:  {data['analysis_status']}")

            # verify audit log
            r2 = client.get("/api/audit")
            events = [e["event_type"] for e in r2.get_json()["data"]]
            print(f"  audit events:     {events}")
            assert "ANALYSIS_COMPLETED" in events
            print("  audit OK")

            # integrity verify
            r3 = client.post(f"/api/evidence/{evidence_id}/verify-integrity")
            print(f"  integrity verify: {r3.get_json()['data']['integrity_status']}")
            assert r3.get_json()["data"]["integrity_status"] == "VALID"

            db.drop_all()
            print("\n*** DIAGNOSTIC PASSED ***")
            return 0

    db.drop_all()
    print("\n*** DIAGNOSTIC FAILED ***")
    return 1


if __name__ == "__main__":
    sys.exit(main())
