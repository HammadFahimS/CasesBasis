"""LangWatch tracing for every LLM call in the app.

``langwatch.setup()`` reads ``LANGWATCH_API_KEY`` from the environment, so it is
run lazily on the first traced call: by then every entry point (``cli``,
``web_app``, ``simulation_runner``, ...) has already called ``load_dotenv()``.
When the SDK is missing or no key is set, the decorators are pass-throughs, so
tests and untraced runs pay nothing and tracing can never break a turn.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

try:
    import langwatch as _langwatch
except Exception:  # SDK not installed → tracing disabled
    _langwatch = None

_setup_lock = threading.Lock()
_setup_done = False
_enabled = False


def langwatch_enabled() -> bool:
    """Initialise LangWatch once, on first use. True when traces are being sent."""
    global _setup_done, _enabled
    if _setup_done:
        return _enabled
    with _setup_lock:
        if _setup_done:
            return _enabled
        _setup_done = True
        if _langwatch is None or not os.getenv("LANGWATCH_API_KEY"):
            return False
        try:
            _langwatch.setup(
                base_attributes={"service.name": "casesbasis"},
            )
            _enabled = True
        except Exception:
            logger.warning("LangWatch setup failed; tracing disabled", exc_info=True)
    return _enabled


def trace(name: str, **trace_kwargs: Any) -> Callable[[F], F]:
    """Wrap an end-to-end operation (one case generation, one voice turn) as a trace."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not langwatch_enabled():
                return func(*args, **kwargs)
            return _langwatch.trace(name=name, **trace_kwargs)(func)(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def span(name: str, type: str = "span", **span_kwargs: Any) -> Callable[[F], F]:
    """Wrap one step (an LLM call, a validation pass) as a span under the current trace."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not langwatch_enabled():
                return func(*args, **kwargs)
            return _langwatch.span(
                name=name, type=type, ignore_missing_trace_warning=True, **span_kwargs
            )(func)(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def update_current_span(**fields: Any) -> None:
    """Attach model / input / output / metadata to the active span. Never raises."""
    if not _enabled:
        return
    try:
        current = _langwatch.get_current_span()
        if current is not None:
            current.update(**fields)
    except Exception:
        logger.debug("LangWatch span update failed", exc_info=True)


def update_current_trace(**fields: Any) -> None:
    """Attach metadata (thread_id, user_id, labels) to the active trace. Never raises."""
    if not _enabled:
        return
    try:
        current = _langwatch.get_current_trace()
        if current is not None:
            current.update(**fields)
    except Exception:
        logger.debug("LangWatch trace update failed", exc_info=True)
