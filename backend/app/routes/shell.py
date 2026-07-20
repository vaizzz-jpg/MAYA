"""Presentation shell routes (Phase 1 foundation only).

Why this exists:
    Verifies template/static wiring and investigator chrome without
    implementing authentication, cases, or analysis.
"""

from __future__ import annotations

from flask import Blueprint, render_template

shell_bp = Blueprint("shell", __name__)


@shell_bp.get("/")
def home():
    """Render the foundation landing shell."""

    return render_template("home.html")
