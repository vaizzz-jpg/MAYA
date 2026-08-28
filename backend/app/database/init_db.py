"""Database initialization helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from backend.app.extensions import db

logger = logging.getLogger("maya.database")


def init_database(app: Flask) -> None:
    """Bind SQLAlchemy and create tables for registered models."""

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        # Ensure models are registered on metadata
        import backend.app.models  # noqa: F401

        db.create_all()
        logger.info(
            "Database initialized (%s)",
            app.config.get("SQLALCHEMY_DATABASE_URI"),
        )
