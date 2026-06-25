"""Authentication routes: magic-link, code login/signup, logout, identity.

Extracted from ``web_app`` (F3). ``web_app`` mounts ``router`` and re-exports
the handlers + request models so existing call sites and tests that reference
``web_app.request_magic_link`` / ``web_app.RequestLinkRequest`` keep working.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator

from backend.auth import (
    COOKIE_NAME,
    generate_magic_token,
    get_current_resident_id,
    make_session_cookie,
    verify_magic_token,
)
from backend.persistence import (
    consume_magic_link,
    create_resident_with_code,
    find_resident_by_code,
    get_or_create_resident,
    get_resident,
    store_magic_link,
)
from backend.web_deps import limiter, set_auth_cookie as _set_auth_cookie

logger = logging.getLogger(__name__)

router = APIRouter()


class RequestLinkRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) > 254:
            raise ValueError("email too long")
        return v


class ResidentLoginRequest(BaseModel):
    resident_code: str
    last_name: str

    @field_validator("resident_code", "last_name")
    @classmethod
    def validate_fields(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 100:
            raise ValueError("value too long")
        return v


@router.post("/api/auth/request-link")
@limiter.limit("5/minute")
def request_magic_link(request: Request, body: RequestLinkRequest) -> dict[str, str]:
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="valid email required")

    ttl_minutes = int(os.getenv("MAGIC_LINK_TTL_MINUTES", "15"))
    token = generate_magic_token(email)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
    resident_id = get_or_create_resident(email)
    store_magic_link(token, resident_id, expires_at)

    link = f"/api/auth/verify?token={token}"
    logger.info("Magic link for %s: %s", email, link)

    return {"message": "Check your email for a login link.", "dev_link": link}


@router.get("/api/auth/verify")
def verify_magic_link(token: str) -> RedirectResponse:
    ttl_minutes = int(os.getenv("MAGIC_LINK_TTL_MINUTES", "15"))
    email = verify_magic_token(token, ttl_minutes * 60)
    if email is None:
        raise HTTPException(status_code=400, detail="invalid or expired link")

    resident_id = consume_magic_link(token)
    if resident_id is None:
        raise HTTPException(status_code=400, detail="link already used or not found")

    cookie_value = make_session_cookie(resident_id)
    redirect = RedirectResponse(url="/home", status_code=302)
    redirect.set_cookie(
        COOKIE_NAME,
        cookie_value,
        httponly=True,
        samesite="strict",
        max_age=30 * 24 * 3600,
    )
    return redirect


@router.post("/api/auth/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.post("/api/auth/test-login")
def test_login(response: Response) -> dict[str, Any]:
    resident_id = get_or_create_resident("test@casesbasis.dev")
    _set_auth_cookie(response, resident_id)
    return {"ok": True, "resident_id": resident_id}


@router.post("/api/auth/login")
@limiter.limit("10/minute")
def login_resident(request: Request, body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
    resident_id = find_resident_by_code(body.resident_code, body.last_name)
    if resident_id is None:
        raise HTTPException(status_code=404, detail="not_found")
    _set_auth_cookie(response, resident_id)
    return {"ok": True}


@router.post("/api/auth/signup")
@limiter.limit("5/minute")
def signup_resident(request: Request, body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
    existing = find_resident_by_code(body.resident_code, body.last_name)
    if existing:
        _set_auth_cookie(response, existing)
        return {"ok": True, "existed": True}
    resident_id = create_resident_with_code(body.resident_code, body.last_name)
    _set_auth_cookie(response, resident_id)
    return {"ok": True, "existed": False}


@router.get("/api/auth/me")
def me(request: Request) -> dict[str, str | None]:
    resident_id = get_current_resident_id(request)
    if resident_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    resident = get_resident(resident_id)
    if resident is None:
        raise HTTPException(status_code=401, detail="resident not found")
    return {
        "resident_id": resident.id,
        "email": resident.email,
        "display_name": resident.display_name,
    }
