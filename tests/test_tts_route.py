"""``/api/voice/tts`` serves exactly what it asked ElevenLabs for.

A reply shown but never heard (2026-10-25 diagnosis): the route sniffed the first
two bytes of the stream for an MP3 frame sync (``FF Ex``) and relabelled the
response ``audio/mpeg`` — but raw PCM-16 starts with ``FF FF`` whenever the first
sample is -1, which near-silent speech onsets often are. The browser handed raw PCM
to ``<audio>``, which cannot decode it, and the line played silently (4 of 25 lines
in one local run). The request pins ``output_format=pcm_16000`` in the query string,
so the response is PCM; a failed stream is a 502 the client retries, not a hidden
server-side fallback chain.
"""

from __future__ import annotations

import asyncio
import struct

import pytest
from fastapi import HTTPException

from backend import voice_service


@pytest.fixture()
def resident(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "tts_route.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-tts-route")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    run_migrations()
    return get_or_create_resident("tts@test.com")


class Req:
    def __init__(self, resident_id):
        from backend.auth import COOKIE_NAME, make_session_cookie
        self.cookies = {COOKIE_NAME: make_session_cookie(resident_id)}


def _pcm(first_sample: int) -> bytes:
    return struct.pack("<8h", first_sample, 3, 5, 2, -1, 0, 4, 1) * 64


def _stream_of(*chunks, calls=None, fail: Exception | None = None):
    def fake(*a, **k):
        if calls is not None:
            calls.append(a)

        def gen():
            if fail:
                raise fail
            yield from chunks
        return "audio/pcm", gen()
    return fake


def _body(resp) -> bytes:
    async def collect():
        return b"".join([c async for c in resp.body_iterator])
    return asyncio.run(collect())


def _tts(resident, **body):
    from backend.routers.voice import api_voice_tts
    return api_voice_tts({"text": "hi", "speaker": "nurse", **body}, Req(resident))


@pytest.mark.parametrize("first_sample", [0, 1, -2, -1, -257, -7937])
def test_pcm_is_served_as_pcm_whatever_its_first_sample(resident, monkeypatch, first_sample):
    # -1, -257 and -7937 start with an MP3-like frame sync (FF FF / FF FE / FF E0).
    pcm = _pcm(first_sample)
    monkeypatch.setattr(voice_service, "synthesize_stream", _stream_of(pcm[:256], pcm[256:]))
    resp = _tts(resident)
    assert resp.media_type == "audio/pcm"
    assert _body(resp) == pcm


def test_empty_stream_is_a_502_after_one_elevenlabs_call(resident, monkeypatch):
    calls = []
    monkeypatch.setattr(voice_service, "synthesize_stream", _stream_of(calls=calls))
    with pytest.raises(HTTPException) as exc:
        _tts(resident)
    assert exc.value.status_code == 502
    assert len(calls) == 1          # the client retries; the server adds no hidden fallbacks


def test_stream_error_at_start_is_a_502_naming_the_cause(resident, monkeypatch):
    monkeypatch.setattr(voice_service, "synthesize_stream",
                        _stream_of(fail=RuntimeError("429 concurrent limit")))
    with pytest.raises(HTTPException) as exc:
        _tts(resident)
    assert exc.value.status_code == 502 and "429 concurrent limit" in exc.value.detail


def test_stream_request_pins_pcm_in_the_query_string(monkeypatch):
    # The guard the byte-sniff stood in for: in the JSON body, output_format is ignored
    # and ElevenLabs answers MP3, which the PCM player renders as static.
    monkeypatch.setattr("backend.pronunciation.tts_locators", lambda: [])
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=4096):
            yield b"\x00\x00"

    class _Ctx:
        def __enter__(self):
            return _Resp()

        def __exit__(self, *exc):
            return False

    class _Client:
        def stream(self, method, url, params=None, headers=None, json=None):
            seen.update(params=params, json=json)
            return _Ctx()

    monkeypatch.setattr(voice_service, "_http", lambda: _Client())
    list(voice_service._el_tts_stream("hi", "v", 0.5, 0.75, "eleven_v4_turbo"))
    assert seen["params"] == {"output_format": "pcm_16000"}
    assert "output_format" not in seen["json"]
