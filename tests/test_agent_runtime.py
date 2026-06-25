"""The shared interactive runtime: timeout, retry, soft-failure, tracing.

These lock in the F5 guarantees the voice room relies on — a hung or flaky
provider degrades within seconds (raising the typed ``AgentError`` callers fail
soft on) rather than stalling a turn, and one retry papers over a transient
blip. The voice modules' own ``_complete`` seams are exercised by their own
tests with the seam monkeypatched; here we drive the runtime directly.
"""

from __future__ import annotations

import time

import pytest

from backend import agent_runtime
from backend.agent_runtime import AgentError, CompletionTrace, complete


class _Client:
    """A stand-in provider client with a scriptable ``text_response``."""

    def __init__(self, behaviours):
        self._behaviours = list(behaviours)
        self.calls = 0

    def text_response(self, *, model, system_prompt, user_prompt):
        b = self._behaviours[min(self.calls, len(self._behaviours) - 1)]
        self.calls += 1
        if callable(b):
            return b()
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture
def fake_provider(monkeypatch):
    """Route ``_resolve`` to a scriptable client and capture traces."""

    def install(client):
        monkeypatch.setattr(agent_runtime, "_resolve", lambda: ("test", client, "test-model"))
        traces: list[CompletionTrace] = []
        monkeypatch.setattr(agent_runtime, "trace_hook", traces.append)
        return traces

    return install


def test_returns_stripped_completion(fake_provider):
    client = _Client(["  hello, doctor  "])
    traces = fake_provider(client)

    assert complete("sys", "user", kind="nurse") == "hello, doctor"
    assert client.calls == 1
    assert traces[-1].ok and traces[-1].kind == "nurse" and traces[-1].attempts == 1


def test_retries_once_then_succeeds(fake_provider):
    client = _Client([RuntimeError("transient 503"), "recovered"])
    traces = fake_provider(client)

    assert complete("sys", "user", retries=1) == "recovered"
    assert client.calls == 2
    assert traces[-1].ok and traces[-1].attempts == 2


def test_raises_agent_error_after_exhausting_retries(fake_provider):
    client = _Client([RuntimeError("down")])
    traces = fake_provider(client)

    with pytest.raises(AgentError):
        complete("sys", "user", retries=1)
    assert client.calls == 2
    assert not traces[-1].ok and traces[-1].error


def test_empty_completion_is_treated_as_failure(fake_provider):
    client = _Client(["   "])
    fake_provider(client)

    with pytest.raises(AgentError):
        complete("sys", "user", retries=0)


def test_timeout_bounds_the_wait(fake_provider):
    def hang():
        time.sleep(5)
        return "too late"

    client = _Client([hang])
    traces = fake_provider(client)

    started = time.monotonic()
    with pytest.raises(AgentError):
        complete("sys", "user", timeout=0.2, retries=0)
    assert time.monotonic() - started < 2  # returned on the timeout, not the 5s sleep
    assert not traces[-1].ok
