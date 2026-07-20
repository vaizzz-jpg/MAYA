"""Database initialization helpers.

Why this exists:
    Schema bootstrap belongs in infrastructure code, not in HTTP routes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from backend.app.extensions import db

logger = logging.getLogger("maya.database")


def init_database(app: Flask) -> None:
    """Bind SQLAlchemy and create tables for any imported models.

    Phase 1 note:
        No business models are registered yet. ``create_all()`` remains
        safe and establishes the Phase 2 pathway.
    """

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()
        logger.info(
            "Database initialized (%s)",
            app.config.get("SQLALCHEMY_DATABASE_URI"),
        )
