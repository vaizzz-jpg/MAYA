"""Analysis / investigation JSON API."""

from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user

from backend.app.schemas import analysis_to_dict, api_success
from backend.app.security import login_required_api
from backend.app.services import analysis_service

analysis_bp = Blueprint("api_analysis", __name__, url_prefix="/api")


@analysis_bp.post("/evidence/<int:evidence_id>/analyze")
@login_required_api
def analyze_evidence(evidence_id: int):
    payload = request.get_json(silent=True) or {}
    generate = payload.get("generate_explanation", True)
    explainer = str(payload.get("explainer", "gradcam"))
    try:
        run = analysis_service.analyze_evidence(
            current_user,
            evidence_id,
            generate_explanation=bool(generate),
            explainer=explainer,
            verify_before_analyze=bool(payload.get("verify_before_analyze", True)),
        )
    except Exception as exc:
        from backend.app.exceptions import MayaProductError
        from backend.app.schemas import api_error

        if isinstance(exc, MayaProductError):
            raise
        return api_error(
            f"Analysis failed: {exc}",
            status=500,
            error_code="analysis_failed",
        )
    return api_success(analysis_to_dict(run), status=201)


@analysis_bp.get("/analysis/<int:analysis_id>")
@login_required_api
def get_analysis(analysis_id: int):
    run = analysis_service.get_analysis(current_user, analysis_id)
    return api_success(analysis_to_dict(run))


@analysis_bp.get("/investigations/<int:analysis_id>")
@login_required_api
def get_investigation(analysis_id: int):
    run = analysis_service.get_analysis(current_user, analysis_id)
    return api_success(analysis_to_dict(run))
