"""Cross-cutting web primitives shared by ``web_app`` and the route modules.

A leaf module by design: it imports only :mod:`backend.auth` (and FastAPI /
slowapi), never ``web_app`` or any router. That keeps it free of import cycles
so every ``backend.routers.*`` module can depend on it for the two things every
route needs — the shared rate limiter and the "who is this resident" guard —
without reaching back into the composition root.

``web_app`` re-exports these under their historical names (``_limiter``,
``_set_auth_cookie``) so existing call sites and tests are unaffected.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.auth import COOKIE_NAME, get_current_resident_id, make_session_cookie

# One limiter instance for the whole app; ``web_app`` registers it on
# ``app.state.limiter`` and every route decorates with ``@limiter.limit(...)``.
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


def require_resident(request: Request) -> str:
    """The authenticated resident id, or 401. The guard nearly every route opens with."""
    resident_id = get_current_resident_id(request)
    if resident_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return resident_id


def set_auth_cookie(response: Response, resident_id: str) -> None:
    """Write the signed 30-day session cookie after login / signup / magic-link."""
    response.set_cookie(
        COOKIE_NAME,
        make_session_cookie(resident_id),
        httponly=True,
        samesite="strict",
        max_age=30 * 24 * 3600,
    )
