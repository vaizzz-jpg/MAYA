"""Environment-based configuration for MAYA.

Why this module exists:
    Centralizes settings so paths, database URLs, and secrets are not
    scattered across the codebase or hardcoded into routes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# MAYA/backend/app/config/config.py → parents[3] == repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
INSTANCE_DIR = BACKEND_DIR / "instance"
LOG_DIR = ROOT_DIR / "logs"
UPLOAD_DIR = ROOT_DIR / "uploads"
REPORT_DIR = ROOT_DIR / "reports"

load_dotenv(ROOT_DIR / ".env")


class BaseConfig:
    """Shared configuration for all environments."""

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(INSTANCE_DIR / 'maya.db').as_posix()}",
    )

    ROOT_DIR: Path = ROOT_DIR
    FRONTEND_DIR: Path = FRONTEND_DIR
    LOG_DIR: Path = LOG_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR
    REPORT_DIR: Path = REPORT_DIR

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
    # Session cookie settings for Flask-Login
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    REMEMBER_COOKIE_HTTPONLY: bool = True
    ALLOW_PUBLIC_REGISTRATION: bool = os.getenv(
        "ALLOW_PUBLIC_REGISTRATION", "true"
    ).lower() in {"1", "true", "yes", "on"}


class DevelopmentConfig(BaseConfig):
    """Local development settings."""

    DEBUG: bool = True


class TestingConfig(BaseConfig):
    """Isolated in-memory settings for automated tests."""

    TESTING: bool = True
    DEBUG: bool = True
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    WTF_CSRF_ENABLED: bool = False
    # Isolated upload root set by tests via app.config override when needed


class ProductionConfig(BaseConfig):
    """Production defaults (DEBUG off)."""

    DEBUG: bool = False


CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    """Resolve a configuration class from FLASK_ENV or an explicit name."""

    env_name = (name or os.getenv("FLASK_ENV", "development")).lower()
    return CONFIG_MAP.get(env_name, DevelopmentConfig)
