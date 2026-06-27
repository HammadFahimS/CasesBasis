"""One interactive-grade seam for every voice/character LLM call.

Before this module, each voice site (nurse_brain, patient_voice, consult_voice,
voice_debrief, …) copy-pasted the same provider plumbing —
``get_llm_provider`` / ``build_llm_client`` / ``resolve_model`` then
``client.text_response(...)`` — with no shared timeout, retry, or observability.
A slow provider could stall a resident's turn for up to a minute (httpx's 60s
default, urllib's 120s). As the vision multiplies agents (Director, per-case
personas, emotional voicing), that copy-paste becomes N subtly-different call
sites — the "unmanageable agents" risk.

``AgentRuntime`` is the single entry point those sites delegate to. It is a thin
*reliability + observability + prompt-composition* wrapper, **not** an agent
framework — it adds an interactive timeout, one retry with short backoff, a
typed soft-failure the callers already handle, and a tracing hook for latency /
provider / model / outcome. The provider clients (and the Cerebras
model-fallback inside :class:`~agents.base.CerebrasClient`) are reused untouched.

Each module keeps its own ``_complete(system_prompt, user_prompt)`` shim — the
seam tests monkeypatch — which now simply calls :func:`complete` here.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# LangSmith tracing for the voice/character path. The structured-output agent
# (agents.base.OpenAIJsonAgent) is already @traceable; voice turns go through
# `complete` below, so without this they never reach LangSmith. Optional — a no-op
# when the SDK is absent, and the SDK itself only ships spans when
# LANGCHAIN_TRACING_V2 / LANGSMITH_TRACING is set, so tests and untraced runs pay
# nothing.
try:
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree as _ls_run_tree
except Exception:  # langsmith not installed / import error → tracing disabled
    _ls_traceable = None
    _ls_run_tree = None


def _traced(func):
    """Wrap `complete` as a LangSmith LLM run when the SDK is available."""
    if _ls_traceable is None:
        return func
    return _ls_traceable(run_type="llm", name="voice.complete")(func)


def _annotate_run(kind: str | None, provider: str, model: str) -> None:
    """Tag the active LangSmith run with kind/provider/model and a readable name."""
    if _ls_run_tree is None:
        return
    try:
        run = _ls_run_tree()
        if run is None:
            return
        run.name = f"voice.{kind or 'nurse'}"
        meta = run.extra.setdefault("metadata", {}) if run.extra is not None else {}
        meta.update({"kind": kind, "provider": provider, "model": model})
    except Exception:  # tracing must never break a turn
        logger.debug("langsmith run annotation failed", exc_info=True)


# Interactive budget: a hung provider should degrade within seconds, not stall a
# turn for a minute. Bounds the *caller's* wait regardless of provider (the
# underlying request can't be cancelled, but the room stops waiting on it).
DEFAULT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", "12"))
DEFAULT_RETRIES = int(os.getenv("AGENT_RETRIES", "1"))
RETRY_BACKOFF_S = float(os.getenv("AGENT_RETRY_BACKOFF_S", "0.4"))


class AgentError(RuntimeError):
    """An interactive completion failed after retries/timeout.

    Voice call sites already wrap their ``_complete`` in ``try/except`` and fail
    soft (read the raw beat, give a canned line) — this is the typed error they
    catch so the room keeps running."""


@dataclass(frozen=True)
class CompletionTrace:
    """One completion's telemetry — fed to the tracing hook (pairs with F8)."""

    kind: str | None
    provider: str
    model: str
    latency_s: float
    attempts: int
    ok: bool
    error: str | None = None


# Optional sink for per-call telemetry. The diagnostics/latency layer (F8) can
# set this; by default we just log. Kept module-level so it's trivially wired.
trace_hook: Callable[[CompletionTrace], None] | None = None

# Provider clients are stateless and reusable — build each once instead of
# rebuilding (and re-importing the SDK) on every turn. Keyed by provider so a
# test or runtime that flips ``LLM_PROVIDER`` still gets the right client.
_client_cache: dict[str, Any] = {}

# A small pool so a slow call bounds the caller's wait without blocking it; the
# orphaned future drains in the background if the provider eventually answers.
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="agent-rt"
)


def _resolve() -> tuple[str, Any, str]:
    """The app's provider, a cached client for it, and the resolved model."""
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    provider = get_llm_provider()
    client = _client_cache.get(provider)
    if client is None:
        client = build_llm_client(provider)
        _client_cache[provider] = client
    return provider, client, resolve_model(provider, None)


def _emit(trace: CompletionTrace) -> None:
    if trace.ok:
        logger.debug(
            "agent completion ok kind=%s provider=%s model=%s latency=%.2fs attempts=%d",
            trace.kind, trace.provider, trace.model, trace.latency_s, trace.attempts,
        )
    else:
        logger.warning(
            "agent completion failed kind=%s provider=%s model=%s latency=%.2fs attempts=%d: %s",
            trace.kind, trace.provider, trace.model, trace.latency_s, trace.attempts, trace.error,
        )
    if trace_hook is not None:
        try:
            trace_hook(trace)
        except Exception:  # telemetry must never break a turn
            logger.debug("agent trace_hook raised", exc_info=True)


@_traced
def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    kind: str | None = None,
    timeout: float | None = None,
    retries: int | None = None,
) -> str:
    """Run one text completion through the app's provider, interactive-grade.

    ``kind`` is the speaker kind (nurse/patient/ems/…) for tracing only. Bounds
    the wait to ``timeout`` seconds, retries once on failure with a short
    backoff, and raises :class:`AgentError` if every attempt fails — callers
    fail soft on that. Provider/model selection and the Cerebras fallback are
    unchanged.
    """
    budget = DEFAULT_TIMEOUT_S if timeout is None else timeout
    max_attempts = (DEFAULT_RETRIES if retries is None else retries) + 1

    provider, client, model = _resolve()
    _annotate_run(kind, provider, model)
    started = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            future = _executor.submit(
                client.text_response,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            text = future.result(timeout=budget)
            result = (text or "").strip()
            if not result:
                raise ValueError("empty completion")
            _emit(CompletionTrace(kind, provider, model, time.monotonic() - started, attempt, True))
            return result
        except concurrent.futures.TimeoutError as exc:
            last_exc = TimeoutError(f"provider exceeded {budget:.0f}s")
            logger.warning("agent completion timed out (kind=%s, attempt=%d/%d)", kind, attempt, max_attempts)
        except Exception as exc:
            last_exc = exc
            logger.warning("agent completion error (kind=%s, attempt=%d/%d): %s", kind, attempt, max_attempts, exc)
        if attempt < max_attempts:
            time.sleep(RETRY_BACKOFF_S * attempt)

    _emit(CompletionTrace(kind, provider, model, time.monotonic() - started, max_attempts, False, str(last_exc)))
    raise AgentError(f"completion failed after {max_attempts} attempt(s): {last_exc}") from last_exc
