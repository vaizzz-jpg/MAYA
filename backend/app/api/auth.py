"""Auth JSON API (Flask-Login sessions)."""

from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user

from backend.app.exceptions import ValidationError
from backend.app.schemas import api_success, user_to_dict
from backend.app.security import login_required_api
from backend.app.services import auth_service

auth_bp = Blueprint("api_auth", __name__, url_prefix="/api/auth")


@auth_bp.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    user = auth_service.register_user(
        email=str(payload.get("email", "")),
        username=str(payload.get("username", "")),
        password=str(payload.get("password", "")),
        full_name=str(payload.get("full_name", "")),
    )
    return api_success(user_to_dict(user), status=201, message="Registered")


@auth_bp.post("/login")
def login():
    payload = request.get_json(silent=True) or {}
    login_id = payload.get("login") or payload.get("email") or payload.get("username")
    if not login_id:
        raise ValidationError("login (email or username) is required")
    user = auth_service.authenticate_user(
        login=str(login_id),
        password=str(payload.get("password", "")),
    )
    return api_success(user_to_dict(user), message="Logged in")


@auth_bp.post("/logout")
@login_required_api
def logout():
    auth_service.logout_current_user()
    return api_success(message="Logged out")


@auth_bp.get("/me")
@login_required_api
def me():
    return api_success(user_to_dict(current_user))
