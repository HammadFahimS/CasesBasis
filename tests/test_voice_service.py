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

    def fake_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
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

    def fake_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
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

    def failing_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
        raise httpx.HTTPStatusError("503", request=None, response=_resp(503))

    monkeypatch.setattr(voice_service, "_el_tts_request", failing_el)
    with pytest.raises(httpx.HTTPStatusError):
        voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")


def test_synthesize_el_per_actor_stability(monkeypatch):
    """Patient voice (low stability) is looked up from the actors registry."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_PATIENT_VOICE_ID", raising=False)
    received = {}

    def fake_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
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


# ── Extreme tier selects the ElevenLabs frontier model ───────────────────────

def test_synthesize_extreme_tier_uses_frontier_model(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    calls = []

    def fake_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
        calls.append(model_id)
        return b"EL_AUDIO"

    monkeypatch.setattr(voice_service, "_el_tts_request", fake_el)
    voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM", tier="extreme")
    voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM", tier="models")
    voice_service.synthesize("hello", "21m00Tcm4TlvDq8ikWAM")
    assert calls == [voice_service.EL_TTS_MODEL_V3,
                     voice_service.EL_TTS_MODEL,
                     voice_service.EL_TTS_MODEL]


def test_synthesize_stream_extreme_tier_passes_frontier_model(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    seen = {}

    def fake_stream(text, voice_id, stability, similarity, model_id=voice_service.EL_TTS_MODEL, style=0.0):
        seen["model_id"] = model_id
        yield b"PCM"

    monkeypatch.setattr(voice_service, "_el_tts_stream", fake_stream)
    _mt, gen = voice_service.synthesize_stream("hi", None, tier="extreme")
    list(gen)
    assert seen["model_id"] == voice_service.EL_TTS_MODEL_V3


# ── Speak-time clinical-abbreviation normalisation ───────────────────────────

def test_normalize_for_speech_units_and_acronyms():
    from backend.speech_normalize import normalize_for_speech as N
    assert N("SpO2 94% on 15 L O2") == "oxygen saturation 94 percent on 15 liters oxygen"
    assert N("BP 120/80, HR 110 bpm") == "blood pressure 120 over 80, heart rate 110 beats per minute"
    assert N("Give 5 mg IV morphine.") == "Give 5 milligrams I V morphine."
    assert N("Na 138 mmol/L, glucose 180 mg/dL") == \
        "Na 138 millimoles per liter, glucose 180 milligrams per deciliter"
    assert N("Temp 38.5°C, MAP 65 mmHg") == "Temp 38.5 degrees Celsius, MAP 65 millimeters of mercury"
    assert N("SpO₂ 88%") == "oxygen saturation 88 percent"


def test_normalize_for_speech_leaves_prose_untouched():
    from backend.speech_normalize import normalize_for_speech as N
    prose = "The patient is comfortable and talking in full sentences."
    assert N(prose) == prose
    assert N("") == ""


# ── Pronunciation dictionary (phonetic pass) is optional + graceful ──────────

def test_pronunciation_rules_well_formed():
    from backend import pronunciation
    assert pronunciation.RULES, "lexicon should not be empty"
    for r in pronunciation.RULES:
        assert r.get("string_to_replace"), r
        assert r.get("type") == "alias", r
        assert r.get("alias"), r


def test_pronunciation_locators_empty_without_key(monkeypatch):
    from backend import pronunciation
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_PRONUNCIATION_DICTIONARY_ID", raising=False)
    monkeypatch.setattr(pronunciation, "_locator", pronunciation._UNSET)
    assert pronunciation.tts_locators() == []


def test_pronunciation_env_pin_skips_network(monkeypatch):
    from backend import pronunciation
    monkeypatch.setenv("ELEVENLABS_PRONUNCIATION_DICTIONARY_ID", "pd_123")
    monkeypatch.setenv("ELEVENLABS_PRONUNCIATION_VERSION_ID", "v7")
    monkeypatch.setattr(pronunciation, "_locator", pronunciation._UNSET)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit on env pin"))
    monkeypatch.setattr(pronunciation, "_find_existing", boom)
    monkeypatch.setattr(pronunciation, "_create", boom)
    assert pronunciation.tts_locators() == [
        {"pronunciation_dictionary_id": "pd_123", "version_id": "v7"}
    ]


def test_pronunciation_resolution_failure_degrades(monkeypatch):
    from backend import pronunciation
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_PRONUNCIATION_DICTIONARY_ID", raising=False)
    monkeypatch.setattr(pronunciation, "_locator", pronunciation._UNSET)
    def boom(*a, **k):
        raise RuntimeError("elevenlabs down")
    monkeypatch.setattr(pronunciation, "_find_existing", boom)
    assert pronunciation.tts_locators() == []


# ── Expressiveness (Extreme-only) raises the TTS style ───────────────────────

def test_synthesize_expressive_sets_style(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    seen = {}

    def fake_el(text, voice_id, stability, similarity, model_id, fmt, style=0.0):
        seen["style"] = style
        return b"A"

    monkeypatch.setattr(voice_service, "_el_tts_request", fake_el)
    voice_service.synthesize("hi", "v", tier="extreme", expressive=True)
    assert seen["style"] == voice_service.EL_EXPRESSIVE_STYLE
    voice_service.synthesize("hi", "v", tier="extreme", expressive=False)
    assert seen["style"] == 0.0


def test_head_speaker_has_distinct_voice(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_HEAD_VOICE_ID", raising=False)
    from backend.actors import speaker_voice
    assert speaker_voice("head") == "onwK4e9ZLuTAKqWW03F9"


# ── Head Physician clarify only for substantive, unattributed requests ───────

def test_is_unclear_request_gating():
    from backend import routing

    class I:
        def __init__(self, op="chitchat", addressee=None):
            self.op = op
            self.addressee = addressee

    assert routing._is_unclear_request(I(), "can we please sort this out now") is True
    assert routing._is_unclear_request(I(), "what do you think about that?") is True
    # short acks / filler must NOT nag
    assert routing._is_unclear_request(I(), "okay") is False
    assert routing._is_unclear_request(I(), "hmm right") is False
    # interpreter already attributed it → not unclear
    assert routing._is_unclear_request(I(addressee="patient"), "how are you feeling today friend") is False
    # non-chitchat speech-acts are routed above, never treated as unclear here
    assert routing._is_unclear_request(I(op="order_medication"), "give five of morphine now please") is False


# ── realtime STT keyterms ────────────────────────────────────────────────────
#
# Regression guard for a live bug: ElevenLabs rejects the *entire* realtime
# connection with `invalid_request` when any single keyterm exceeds 20
# characters. The case vocabulary comes from `_stt_prompt`, which was written as
# a Whisper prompt and contains phrases like "Negative inspiratory force" (26).
# Because the browser silently falls back to the blocking upload, the streaming
# path failed on every session while the room looked perfectly healthy.

def test_keyterms_within_limit_pass_through_whole():
    assert voice_service._normalize_keyterms(["CBC", "ECG", "CT head"]) == ["CBC", "ECG", "CT head"]


def test_no_keyterm_exceeds_the_api_limit():
    """The property that actually matters — nothing over 20 chars reaches the wire."""
    raw = [
        "Myasthenia Gravis with Myasthenic Crisis",
        "Negative inspiratory force",
        "Bedside vital capacity",
        "Respiratory Therapist",
    ]
    out = voice_service._normalize_keyterms(raw)
    assert out, "an over-long phrase must yield usable terms, not be dropped wholesale"
    assert all(len(t) <= voice_service._KEYTERM_MAX_CHARS for t in out)


def test_long_phrases_are_split_keeping_distinctive_words():
    out = voice_service._normalize_keyterms(["Negative inspiratory force"])
    assert "inspiratory" in out and "Negative" in out and "force" in out


def test_stopwords_are_dropped_from_split_phrases():
    out = voice_service._normalize_keyterms(["Myasthenia Gravis with Myasthenic Crisis"])
    assert "with" not in [t.lower() for t in out]
    assert "Myasthenic" in out and "Gravis" in out


def test_keyterms_are_deduplicated_case_insensitively():
    out = voice_service._normalize_keyterms(["ECG", "ecg", "CT head", "CT chest"])
    assert out.count("ECG") == 1 and "ecg" not in out
    # "CT" appears in both phrases but each is short enough to stay whole.
    assert "CT head" in out and "CT chest" in out


def test_punctuation_is_stripped_from_split_words():
    out = voice_service._normalize_keyterms(["Severe bulbar weakness, dysphagia (marked)"])
    assert all(t == t.strip(" ()[]{},.;:") for t in out), out
    assert "dysphagia" in out and "marked" in out


def test_keyterms_are_capped_to_the_limit():
    assert len(voice_service._normalize_keyterms([f"term{i}" for i in range(80)], limit=32)) == 32


def test_empty_and_blank_keyterms_are_ignored():
    assert voice_service._normalize_keyterms(["", "   ", None]) == []
    assert voice_service._normalize_keyterms(None) == []


def test_real_case_vocabulary_is_accepted():
    """The exact 18-term list from myasthenic_crisis that broke the connection."""
    raw = [
        "Myasthenia Gravis with Myasthenic Crisis", "Patient (EK)", "EMS", "Husband",
        "Nurse", "Respiratory Therapist", "Neurology consultant", "ICU consultant",
        "Venous blood gas", "CBC", "electrolytes", "renal function", "ECG",
        "Chest X-ray", "Bedside vital capacity", "Negative inspiratory force",
        "CT head", "CT chest",
    ]
    out = voice_service._normalize_keyterms(raw)
    assert all(len(t) <= voice_service._KEYTERM_MAX_CHARS for t in out)
    assert len(out) <= 32
