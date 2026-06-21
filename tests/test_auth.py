from __future__ import annotations

import pytest
from tests.helpers import make_request


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "auth_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-for-auth-tests")
    from backend.db import run_migrations
    run_migrations()
    yield path


# ── Token helpers ─────────────────────────────────────────────────────────────

def test_magic_token_round_trip(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import generate_magic_token, verify_magic_token

    token = generate_magic_token("alice@example.com")
    result = verify_magic_token(token, ttl_seconds=900)
    assert result == "alice@example.com"


def test_magic_token_normalises_email(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import generate_magic_token, verify_magic_token

    token = generate_magic_token("  Alice@Example.COM  ")
    result = verify_magic_token(token, ttl_seconds=900)
    assert result == "alice@example.com"


def test_magic_token_invalid_returns_none(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import verify_magic_token

    assert verify_magic_token("not-a-real-token", ttl_seconds=900) is None


def test_magic_token_wrong_secret_returns_none(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "secret-a")
    from backend import auth as auth_mod
    import importlib
    importlib.reload(auth_mod)
    token = auth_mod.generate_magic_token("bob@example.com")

    monkeypatch.setenv("APP_SECRET", "secret-b")
    importlib.reload(auth_mod)
    assert auth_mod.verify_magic_token(token, ttl_seconds=900) is None


def test_session_cookie_round_trip(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import decode_session_cookie, make_session_cookie

    cookie = make_session_cookie("resident-abc-123")
    result = decode_session_cookie(cookie)
    assert result == "resident-abc-123"


def test_session_cookie_tampered_returns_none(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import decode_session_cookie, make_session_cookie

    cookie = make_session_cookie("resident-xyz")
    tampered = cookie[:-4] + "XXXX"
    assert decode_session_cookie(tampered) is None


def test_get_current_resident_id_from_cookie(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import COOKIE_NAME, get_current_resident_id, make_session_cookie

    cookie_value = make_session_cookie("res-001")

    class FakeRequest:
        cookies = {COOKIE_NAME: cookie_value}

    assert get_current_resident_id(FakeRequest()) == "res-001"


def test_get_current_resident_id_no_cookie(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    from backend.auth import get_current_resident_id

    class FakeRequest:
        cookies: dict = {}

    assert get_current_resident_id(FakeRequest()) is None


# ── Resident DB helpers ───────────────────────────────────────────────────────

def test_get_or_create_resident_creates_new(db_path):
    from backend.persistence import get_or_create_resident, get_resident

    resident_id = get_or_create_resident("carol@example.com")
    assert resident_id

    resident = get_resident(resident_id)
    assert resident is not None
    assert resident.email == "carol@example.com"
    assert resident.display_name == "Carol"
    assert resident.created_at


def test_get_or_create_resident_is_idempotent(db_path):
    from backend.persistence import get_or_create_resident

    id1 = get_or_create_resident("dan@example.com")
    id2 = get_or_create_resident("dan@example.com")
    assert id1 == id2


def test_get_or_create_resident_normalises_email(db_path):
    from backend.persistence import get_or_create_resident

    id1 = get_or_create_resident("Eve@Example.COM")
    id2 = get_or_create_resident("eve@example.com")
    assert id1 == id2


def test_get_resident_returns_none_for_unknown(db_path):
    from backend.persistence import get_resident

    assert get_resident("does-not-exist") is None


def test_display_name_derived_from_email(db_path):
    from backend.persistence import get_or_create_resident, get_resident

    rid = get_or_create_resident("john.doe@hospital.org")
    resident = get_resident(rid)
    assert resident is not None
    assert resident.display_name == "John Doe"


# ── Magic link DB helpers ─────────────────────────────────────────────────────

def test_store_and_consume_magic_link(db_path):
    from backend.db import utcnow
    from backend.persistence import consume_magic_link, get_or_create_resident, store_magic_link
    from datetime import datetime, timedelta, timezone

    resident_id = get_or_create_resident("frank@example.com")
    token = "test-token-abc"
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    store_magic_link(token, resident_id, expires_at)

    result = consume_magic_link(token)
    assert result == resident_id


def test_consume_magic_link_can_only_be_used_once(db_path):
    from backend.persistence import consume_magic_link, get_or_create_resident, store_magic_link
    from datetime import datetime, timedelta, timezone

    resident_id = get_or_create_resident("grace@example.com")
    token = "one-time-token"
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    store_magic_link(token, resident_id, expires_at)

    first = consume_magic_link(token)
    second = consume_magic_link(token)
    assert first == resident_id
    assert second is None


def test_consume_expired_magic_link_returns_none(db_path):
    from backend.persistence import consume_magic_link, get_or_create_resident, store_magic_link
    from datetime import datetime, timedelta, timezone

    resident_id = get_or_create_resident("harry@example.com")
    token = "expired-token"
    expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store_magic_link(token, resident_id, expires_at)

    assert consume_magic_link(token) is None


def test_consume_unknown_magic_link_returns_none(db_path):
    from backend.persistence import consume_magic_link

    assert consume_magic_link("ghost-token") is None


# ── Auth endpoint functions ───────────────────────────────────────────────────

def test_request_magic_link_endpoint_rejects_invalid_email(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    with pytest.raises(HTTPException) as exc_info:
        web_app.request_magic_link(make_request(), web_app.RequestLinkRequest(email="not-an-email"))
    assert exc_info.value.status_code == 422


def test_request_magic_link_endpoint_creates_resident_and_link(db_path):
    import backend.web_app as web_app
    from backend.persistence import get_or_create_resident, get_resident

    result = web_app.request_magic_link(make_request(), web_app.RequestLinkRequest(email="ivan@example.com"))
    assert "message" in result
    assert "dev_link" in result
    assert "/api/auth/verify?token=" in result["dev_link"]


def test_verify_magic_link_endpoint_sets_cookie_and_redirects(db_path):
    import backend.web_app as web_app
    from fastapi.responses import RedirectResponse

    link_result = web_app.request_magic_link(make_request(), web_app.RequestLinkRequest(email="julia@example.com"))
    token = link_result["dev_link"].split("token=")[1]

    response = web_app.verify_magic_link(token)
    assert isinstance(response, RedirectResponse)
    assert response.status_code == 302
    cookie_header = response.headers.get("set-cookie", "")
    assert "cb_session=" in cookie_header
    assert "httponly" in cookie_header.lower()


def test_verify_magic_link_invalid_token_raises_400(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    with pytest.raises(HTTPException) as exc_info:
        web_app.verify_magic_link("bad-token")
    assert exc_info.value.status_code == 400


def test_verify_magic_link_already_used_raises_400(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    link_result = web_app.request_magic_link(make_request(), web_app.RequestLinkRequest(email="kim@example.com"))
    token = link_result["dev_link"].split("token=")[1]

    web_app.verify_magic_link(token)
    with pytest.raises(HTTPException) as exc_info:
        web_app.verify_magic_link(token)
    assert exc_info.value.status_code == 400


def test_me_endpoint_returns_resident_when_authed(db_path, monkeypatch):
    import backend.web_app as web_app
    from backend.auth import COOKIE_NAME, make_session_cookie
    from backend.persistence import get_or_create_resident

    resident_id = get_or_create_resident("leo@example.com")
    cookie_value = make_session_cookie(resident_id)

    class FakeRequest:
        cookies = {COOKIE_NAME: cookie_value}

    result = web_app.me(FakeRequest())
    assert result["email"] == "leo@example.com"
    assert result["resident_id"] == resident_id


def test_me_endpoint_raises_401_when_not_authed(db_path):
    from fastapi import HTTPException
    import backend.web_app as web_app

    class FakeRequest:
        cookies: dict = {}

    with pytest.raises(HTTPException) as exc_info:
        web_app.me(FakeRequest())
    assert exc_info.value.status_code == 401
