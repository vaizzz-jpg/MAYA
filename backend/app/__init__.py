"""MAYA Flask application factory.

Why this exists:
    An application factory enables testing, multiple configs, and deferred
    extension initialization without a global app singleton.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, render_template

from backend.app.config import get_config
from backend.app.database import init_database
from backend.app.routes import register_blueprints
from backend.app.utils.logging_config import configure_logging


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the MAYA Flask application.

    Args:
        config_name: Optional environment name (development/testing/production).

    Returns:
        A fully configured Flask app instance.
    """

    config_object = get_config(config_name)
    frontend_dir: Path = config_object.FRONTEND_DIR

    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir / "static"),
        static_url_path="/static",
        instance_path=str(config_object.ROOT_DIR / "backend" / "instance"),
    )
    app.config.from_object(config_object)

    _ensure_runtime_directories(config_object)
    configure_logging(config_object.LOG_DIR, config_object.LOG_LEVEL)
    init_database(app)
    register_blueprints(app)
    _register_error_handlers(app)

    logging.getLogger("maya").info("MAYA application created (Phase 1 foundation)")
    return app


def _ensure_runtime_directories(config_object: type) -> None:
    """Create runtime directories used by later phases."""

    for path in (
        config_object.LOG_DIR,
        config_object.UPLOAD_DIR,
        config_object.REPORT_DIR,
        config_object.ROOT_DIR / "backend" / "instance",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def _register_error_handlers(app: Flask) -> None:
    """Register professional HTML error pages."""

    @app.errorhandler(403)
    def forbidden(error):  # noqa: ANN001, ARG001
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(error):  # noqa: ANN001, ARG001
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(error):  # noqa: ANN001, ARG001
        return render_template("errors/500.html"), 500
