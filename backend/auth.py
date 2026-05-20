from __future__ import annotations

import logging
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

COOKIE_NAME = "cb_session"
_MAGIC_SALT = "magic-link"
_SESSION_SALT = "session"
_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days


def _secret() -> str:
    secret = os.getenv("APP_SECRET", "")
    if not secret:
        logger.warning("APP_SECRET is not set — using insecure development default")
        return "dev-insecure-secret-do-not-use-in-production"
    return secret


def _serializer(salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret(), salt=salt)


def generate_magic_token(email: str) -> str:
    return _serializer(_MAGIC_SALT).dumps(email.strip().lower())


def verify_magic_token(token: str, ttl_seconds: int) -> str | None:
    try:
        return _serializer(_MAGIC_SALT).loads(token, max_age=ttl_seconds)
    except (BadSignature, SignatureExpired):
        return None


def make_session_cookie(resident_id: str) -> str:
    return _serializer(_SESSION_SALT).dumps(resident_id)


def decode_session_cookie(cookie_value: str) -> str | None:
    try:
        return _serializer(_SESSION_SALT).loads(cookie_value, max_age=_SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_current_resident_id(request: object) -> str | None:
    cookie_value = getattr(request, "cookies", {}).get(COOKIE_NAME)
    if not cookie_value:
        return None
    return decode_session_cookie(cookie_value)
