from __future__ import annotations

import httpx
import pytest

from backend import voice_service


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "http://x/audio/speech"))


def test_synthesize_falls_back_to_default_voice_on_rejected_voice(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_TTS_VOICE", "british reading lady")
    calls = []

    def fake_request(text, voice, fmt):
        calls.append(voice)
        if voice == "newsman":   # provider rejects the unknown voice
            raise httpx.HTTPStatusError("400", request=None, response=_resp(400))
        return b"AUDIO"

    monkeypatch.setattr(voice_service, "_tts_request", fake_request)
    out = voice_service.synthesize("hello", "newsman")
    assert out == b"AUDIO"
    # tried the requested voice, then fell back to the default
    assert calls == ["newsman", "british reading lady"]


def test_synthesize_does_not_loop_when_default_voice_itself_fails(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_TTS_VOICE", "british reading lady")

    def always_400(text, voice, fmt):
        raise httpx.HTTPStatusError("400", request=None, response=_resp(400))

    monkeypatch.setattr(voice_service, "_tts_request", always_400)
    with pytest.raises(httpx.HTTPStatusError):
        voice_service.synthesize("hello", "british reading lady")  # already the default → no retry


def test_synthesize_passes_through_a_working_voice(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setattr(voice_service, "_tts_request", lambda t, v, f: b"OK:" + v.encode())
    assert voice_service.synthesize("hi", "professional man") == b"OK:professional man"
