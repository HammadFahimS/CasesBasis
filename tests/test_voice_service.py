from __future__ import annotations

import httpx
import pytest

from backend import voice_service


def _resp(status: int, url: str = "http://x/audio/speech") -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", url))


# ── Together / Cartesia path (ElevenLabs key absent) ──────────────────────────

def test_synthesize_falls_back_to_default_voice_on_rejected_voice(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
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
    assert calls == ["newsman", "british reading lady"]


def test_synthesize_does_not_loop_when_default_voice_itself_fails(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_TTS_VOICE", "british reading lady")

    def always_400(text, voice, fmt):
        raise httpx.HTTPStatusError("400", request=None, response=_resp(400))

    monkeypatch.setattr(voice_service, "_tts_request", always_400)
    with pytest.raises(httpx.HTTPStatusError):
        voice_service.synthesize("hello", "british reading lady")


def test_synthesize_passes_through_a_working_voice(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setattr(voice_service, "_tts_request", lambda t, v, f: b"OK:" + v.encode())
    assert voice_service.synthesize("hi", "professional man") == b"OK:professional man"


# ── ElevenLabs path ────────────────────────────────────────────────────────────

def test_synthesize_uses_elevenlabs_when_key_set(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    calls = []

    def fake_el(text, voice_id, stability, similarity, model_id, fmt):
        calls.append((voice_id, model_id))
        return b"EL_AUDIO"

    monkeypatch.setattr(voice_service, "_el_tts_request", fake_el)
    out = voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")
    assert out == b"EL_AUDIO"
    assert calls[0][1] == "eleven_flash_v2_5"


def test_synthesize_uses_nurse_default_when_voice_none(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_NURSE_VOICE_ID", raising=False)
    received = {}

    def fake_el(text, voice_id, stability, similarity, model_id, fmt):
        received["voice_id"] = voice_id
        received["stability"] = stability
        return b"NURSE_AUDIO"

    monkeypatch.setattr(voice_service, "_el_tts_request", fake_el)
    out = voice_service.synthesize("on it")
    assert out == b"NURSE_AUDIO"
    # resolved to the built-in nurse default
    assert received["voice_id"] == voice_service._EL_NURSE_DEFAULT
    # nurse gets the calibrated stability, not a per-actor default
    assert received["stability"] == voice_service._EL_NURSE_STABILITY


def test_synthesize_falls_back_to_together_when_elevenlabs_fails(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_TTS_VOICE", "british reading lady")

    def failing_el(text, voice_id, stability, similarity, model_id, fmt):
        raise httpx.HTTPStatusError(
            "503", request=None, response=_resp(503, "http://el/tts")
        )

    together_calls = []

    def fake_together(text, voice, fmt):
        together_calls.append(voice)
        return b"FALLBACK:" + voice.encode()

    monkeypatch.setattr(voice_service, "_el_tts_request", failing_el)
    monkeypatch.setattr(voice_service, "_tts_request", fake_together)

    # EL voice_id passed in — fallback must use Together default, not the EL id
    out = voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")
    assert out == b"FALLBACK:british reading lady"
    assert together_calls == ["british reading lady"]


def test_synthesize_el_per_actor_stability(monkeypatch):
    """Patient voice (low stability) is looked up from the actors registry."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_PATIENT_VOICE_ID", raising=False)
    received = {}

    def fake_el(text, voice_id, stability, similarity, model_id, fmt):
        received["stability"] = stability
        received["similarity"] = similarity
        return b"PATIENT_AUDIO"

    monkeypatch.setattr(voice_service, "_el_tts_request", fake_el)
    from backend.actors import SPEAKERS
    patient = next(s for s in SPEAKERS if s.id == "patient")
    voice_service.synthesize("I can't breathe", patient.el_default_voice)
    assert received["stability"] == pytest.approx(patient.el_stability)
    assert received["similarity"] == pytest.approx(patient.el_similarity)
