from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable slowapi rate limiting for direct endpoint function calls in tests."""
    try:
        import backend.web_app as _wapp
        prev = _wapp._limiter.enabled
        _wapp._limiter.enabled = False
        yield
        _wapp._limiter.enabled = prev
    except Exception:
        yield


@pytest.fixture(autouse=True)
def reset_thread_local_db():
    """Reset the per-thread SQLite connection before and after each test.

    The thread-local connection persists across tests within the same thread.
    When a test's db_path fixture changes DB_PATH and calls run_migrations(),
    get_db() would return the previous test's connection (pointing at a
    different file) unless we clear it here.
    """
    import backend.db as _db

    def _reset():
        conn = getattr(_db._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _db._local.conn = None

    _reset()
    yield
    _reset()
