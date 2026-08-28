"""JSON response helpers and lightweight schema builders."""

from __future__ import annotations

from typing import Any

from flask import jsonify
from werkzeug.wrappers import Response

from backend.app.models.entities import AnalysisRun, Case, Evidence, User


def api_success(data: Any = None, *, status: int = 200, message: str | None = None) -> Response:
    body: dict[str, Any] = {"ok": True}
    if message is not None:
        body["message"] = message
    if data is not None:
        body["data"] = data
    return jsonify(body), status


def api_error(
    message: str,
    *,
    status: int = 400,
    error_code: str = "error",
) -> Response:
    return jsonify({"ok": False, "error": error_code, "message": message}), status


def user_to_dict(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def case_to_dict(case: Case) -> dict[str, Any]:
    return {
        "id": case.id,
        "case_id": case.id,
        "case_number": case.case_number,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "priority": case.priority,
        "created_by": case.created_by_user_id,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
    }


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.id,
        "case_id": evidence.case_id,
        "original_filename": evidence.original_filename,
        "stored_filename": evidence.stored_filename,
        "media_type": evidence.media_type,
        "mime_type": evidence.mime_type,
        "file_size": evidence.file_size_bytes,
        "sha256": evidence.sha256_hash,
        "status": evidence.status,
        "analysis_status": evidence.analysis_status,
        "uploaded_by": evidence.uploaded_by_user_id,
        "created_at": evidence.uploaded_at.isoformat() if evidence.uploaded_at else None,
        "storage_path": evidence.storage_path,
        "notes": evidence.notes,
    }


def analysis_to_dict(run: AnalysisRun) -> dict[str, Any]:
    explanation = None
    if run.generate_explanation or run.overlay_path or run.heatmap_path:
        explanation = {
            "explainer": run.explainer_name,
            "heatmap": run.heatmap_path,
            "overlay": run.overlay_path,
            "explanation_json": run.explanation_json_path,
        }
    return {
        "analysis_id": run.id,
        "investigation_id": run.investigation_id,
        "evidence_id": run.evidence_id,
        "case_id": run.case_id,
        "prediction": run.prediction,
        "confidence": run.confidence,
        "analysis_status": run.status,
        "model_name": run.model_name,
        "model_version": run.model_version,
        "dataset_version": run.dataset_version,
        "artifact_dir": run.artifact_dir,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "explanation": explanation,
    }
