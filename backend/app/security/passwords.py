"""Password hashing and authentication helpers (Werkzeug)."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from flask import g
from flask_login import current_user
from werkzeug.security import check_password_hash, generate_password_hash

from backend.app.exceptions import AuthenticationError, AuthorizationError
from backend.app.models.enums import UserRole

F = TypeVar("F", bound=Callable[..., Any])


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def login_required_api(view: F) -> F:
    """Require an authenticated Flask-Login user for JSON APIs."""

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        if not current_user.is_authenticated:
            raise AuthenticationError("Authentication required")
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def roles_required(*roles: UserRole | str) -> Callable[[F], F]:
    allowed = {
        r.value if isinstance(r, UserRole) else str(r).upper() for r in roles
    }

    def decorator(view: F) -> F:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any):
            if not current_user.is_authenticated:
                raise AuthenticationError("Authentication required")
            if current_user.role not in allowed:
                raise AuthorizationError("Insufficient role for this operation")
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def get_current_user_id() -> int:
    if not current_user.is_authenticated:
        raise AuthenticationError("Authentication required")
    return int(current_user.id)
