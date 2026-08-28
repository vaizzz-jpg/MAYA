"""MAYA Flask application factory.

Why this exists:
    An application factory enables testing, multiple configs, and deferred
    extension initialization without a global app singleton.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, render_template

from backend.app.api import register_api_blueprints
from backend.app.config import get_config
from backend.app.database import init_database
from backend.app.extensions import login_manager
from backend.app.routes import register_blueprints
from backend.app.utils.logging_config import configure_logging


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the MAYA Flask application."""

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

    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        from backend.app.models import User

        try:
            return User.query.get(int(user_id))
        except (TypeError, ValueError):
            return None

    @login_manager.unauthorized_handler
    def unauthorized():
        from backend.app.schemas import api_error

        return api_error("Authentication required", status=401, error_code="authentication_error")

    register_blueprints(app)
    register_api_blueprints(app)
    _register_error_handlers(app)

    logging.getLogger("maya").info("MAYA application created (Phase 3 product)")
    return app


def _ensure_runtime_directories(config_object: type) -> None:
    for path in (
        config_object.LOG_DIR,
        config_object.UPLOAD_DIR,
        config_object.REPORT_DIR,
        config_object.ROOT_DIR / "backend" / "instance",
        config_object.ROOT_DIR / "artifacts" / "investigations",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(403)
    def forbidden(error):  # noqa: ANN001, ARG001
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(error):  # noqa: ANN001, ARG001
        # Prefer JSON for /api paths
        from flask import request

        if request.path.startswith("/api/"):
            from backend.app.schemas import api_error

            return api_error("Not found", status=404, error_code="not_found")
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(error):  # noqa: ANN001, ARG001
        from flask import request

        if request.path.startswith("/api/"):
            from backend.app.schemas import api_error

            return api_error("Internal server error", status=500, error_code="server_error")
        return render_template("errors/500.html"), 500
