from __future__ import annotations

from starlette.requests import Request


def make_request(method: str = "POST", path: str = "/") -> Request:
    """Return a minimal Starlette Request suitable for direct endpoint calls in tests."""
    return Request(scope={
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 0),
    })
