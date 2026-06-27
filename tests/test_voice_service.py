from __future__ import annotations

import httpx
import pytest

from backend import voice_service


def _resp(status: int, url: str = "http://el/tts") -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", url))


# ── ElevenLabs is the sole voice provider ────────────────────────────────────

def test_synthesize_requires_elevenlabs_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")


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


def test_synthesize_raises_on_elevenlabs_failure_no_fallback(monkeypatch):
    """ElevenLabs is the only provider — a failure surfaces, never silently
    degrades to another vendor."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")

    def failing_el(text, voice_id, stability, similarity, model_id, fmt):
        raise httpx.HTTPStatusError("503", request=None, response=_resp(503))

    monkeypatch.setattr(voice_service, "_el_tts_request", failing_el)
    with pytest.raises(httpx.HTTPStatusError):
        voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")


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


# ── Streaming TTS + Scribe STT both require the ElevenLabs key ────────────────

def test_synthesize_stream_requires_elevenlabs_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        voice_service.synthesize_stream("hello", "21m00Tcm4TlvDq8ikWAM")


def test_transcribe_requires_elevenlabs_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        voice_service.transcribe(b"\x00\x01", "audio/webm")
