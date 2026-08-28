"""Append-only product audit logging."""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.app.extensions import db
from backend.app.models.entities import AuditLog
from backend.app.models.enums import AuditEventType

logger = logging.getLogger("maya.backend.audit")


def record_audit(
    event_type: AuditEventType | str,
    *,
    user_id: int | None = None,
    case_id: int | None = None,
    evidence_id: int | None = None,
    analysis_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    event = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
    # Never persist secrets
    safe = {k: v for k, v in (details or {}).items() if k.lower() not in {
        "password", "token", "secret", "authorization"
    }}
    row = AuditLog(
        user_id=user_id,
        case_id=case_id,
        evidence_id=evidence_id,
        analysis_id=analysis_id,
        event_type=event,
        details_json=json.dumps(safe, default=str),
    )
    db.session.add(row)
    db.session.flush()
    logger.info("Audit %s user=%s case=%s evidence=%s", event, user_id, case_id, evidence_id)
    return row
