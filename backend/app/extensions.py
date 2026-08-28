"""Shared Flask extensions (initialized in create_app)."""

from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = None  # JSON APIs return 401, not redirect
