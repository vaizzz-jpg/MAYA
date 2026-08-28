"""Audit JSON API (ADMIN list; authenticated users see own events)."""

from __future__ import annotations

import json

from flask import Blueprint
from flask_login import current_user

from backend.app.models.entities import AuditLog
from backend.app.models.enums import UserRole
from backend.app.schemas import api_success
from backend.app.security import login_required_api

audit_bp = Blueprint("api_audit", __name__, url_prefix="/api/audit")


@audit_bp.get("")
@login_required_api
def list_audit():
    q = AuditLog.query.order_by(AuditLog.timestamp.desc())
    if current_user.role != UserRole.ADMIN.value:
        q = q.filter_by(user_id=current_user.id)
    rows = q.limit(200).all()
    data = [
        {
            "audit_id": r.id,
            "user_id": r.user_id,
            "case_id": r.case_id,
            "evidence_id": r.evidence_id,
            "analysis_id": r.analysis_id,
            "event_type": r.event_type,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "details": json.loads(r.details_json or "{}"),
        }
        for r in rows
    ]
    return api_success(data)
