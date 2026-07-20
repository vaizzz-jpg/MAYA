"""Health blueprint — platform liveness only.

Why this exists:
    Provides a dependency-light endpoint for local verification and
    future reverse-proxy health checks without exposing business data.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    """Return a simple JSON liveness payload."""

    return jsonify(
        {
            "status": "ok",
            "service": "maya",
            "phase": 1,
        }
    )
