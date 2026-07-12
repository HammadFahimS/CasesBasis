"""Thin model-I/O plumbing for the voice room.

Three independent legs:

* STT  — ElevenLabs Scribe (``scribe_v1``, ``/v1/speech-to-text``)
* LLM  — the app's existing provider (Cerebras) via ``agent_runtime.complete``
* TTS  — ElevenLabs Flash v2.5 (``eleven_flash_v2_5``), streamed PCM-16000

ElevenLabs is the sole voice provider — it requires ``ELEVENLABS_API_KEY``.
Flash v2.5 is used for every live turn: it targets ~75 ms latency and streams
PCM the browser plays after the first chunk. There is no second voice vendor; a
voice failure surfaces as an error rather than silently degrading. (The LLM
provider is separate and unaffected.)

API keys stay server-side; the browser only ever talks to our endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── ElevenLabs — the sole voice provider (TTS + STT) ─────────────────────────
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
EL_TTS_MODEL = "eleven_flash_v2_5"   # ~75 ms, low-latency; all live chatty turns
EL_TTS_MODEL_V3 = "eleven_v3"        # expressive + audio tags; reserved for set-pieces
EL_STT_MODEL = "scribe_v1"           # ElevenLabs Scribe — speech-to-text

# Nurse voice: Rachel — calm, clear American female. Override via env.
_EL_NURSE_DEFAULT = "21m00Tcm4TlvDq8ikWAM"
_EL_NURSE_STABILITY = 0.60    # consistent: the voice the resident hears most
_EL_NURSE_SIMILARITY = 0.80

NURSE_SYSTEM_PROMPT = (
    "You are the bedside nurse in an emergency-department simulation, assisting a "
    "physician (the resident) who speaks to you out loud. You are calm, competent and "
    "concise. When the resident gives an order or asks a question, acknowledge it in one "
    "or two short spoken sentences, the way a real nurse would read an order back. Do not "
    "narrate actions in asterisks or use markdown — your reply will be spoken aloud. "
    "If asked for a value you don't have, say you'll get it. Keep replies under ~40 words."
)


def _elevenlabs_key() -> str | None:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip() or None


def _ext_for(content_type: str) -> str:
    if "mp4" in content_type or "m4a" in content_type:
        return "mp4"
    if "ogg" in content_type:
        return "ogg"
    if "wav" in content_type:
        return "wav"
    return "webm"


def transcribe(audio: bytes, content_type: str = "audio/webm",
               prompt: str | None = None) -> str:
    """STT: audio bytes -> transcript text via ElevenLabs Scribe (``scribe_v1``).

    ElevenLabs is the only STT provider. Retries once on a transient 5xx /
    network error; a 4xx (bad request, auth) is raised immediately.

    ``prompt`` is accepted for call-site compatibility but unused — Scribe needs
    no vocabulary seeding (it is strong on medical terms natively)."""
    key = _elevenlabs_key()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set — STT requires ElevenLabs")
    ext = _ext_for(content_type)
    last_exc: Exception | None = None
    for attempt in range(2):
        if attempt:
            time.sleep(1.0)
        try:
            resp = httpx.post(
                f"{ELEVENLABS_BASE_URL}/speech-to-text",
                headers={"xi-api-key": key},
                files={"file": (f"audio.{ext}", audio, content_type)},
                data={"model_id": EL_STT_MODEL, "language_code": "en", "tag_audio_events": "false"},
                timeout=60.0,
            )
            resp.raise_for_status()
            return (resp.json().get("text") or "").strip()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status < 500:
                raise  # 4xx is non-transient — surface it
            logger.warning("Scribe STT attempt %d/2 failed (%s): %s", attempt + 1, status, exc)
        except httpx.RequestError as exc:
            last_exc = exc
            logger.warning("Scribe STT attempt %d/2 request error: %s", attempt + 1, exc)
    raise last_exc  # type: ignore[misc]


def _el_tts_request(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
    model_id: str,
    fmt: str,
    style: float = 0.0,
) -> bytes:
    """ElevenLabs TTS request.

    ``fmt='mp3'`` maps to ``mp3_44100_128`` (128 kbps, 44.1 kHz). Pass a raw
    ElevenLabs format string (e.g. ``pcm_16000``) directly for other formats."""
    from backend.speech_normalize import normalize_for_speech
    from backend.pronunciation import tts_locators
    text = normalize_for_speech(text)
    output_format = "mp3_44100_128" if fmt == "mp3" else fmt
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
            "use_speaker_boost": True,
        },
    }
    locators = tts_locators()
    if locators:
        body["pronunciation_dictionary_locators"] = locators
    # NB: ElevenLabs reads output_format from the QUERY STRING. Putting it in the
    # JSON body is silently ignored and the API defaults to MP3 — which, on the
    # PCM path, the browser decodes as raw samples (loud static).
    resp = httpx.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
        params={"output_format": output_format},
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content


def _el_tts_stream(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
    model_id: str = EL_TTS_MODEL,
    style: float = 0.0,
):
    """Yield raw PCM-16000 chunks from ElevenLabs /stream endpoint.

    Format: 16 kHz, 16-bit signed little-endian, mono.  The caller converts
    these to Float32 AudioBuffers for scheduling on the Web Audio API timeline."""
    from backend.speech_normalize import normalize_for_speech
    from backend.pronunciation import tts_locators
    text = normalize_for_speech(text)
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
            "use_speaker_boost": True,
        },
    }
    locators = tts_locators()
    if locators:
        body["pronunciation_dictionary_locators"] = locators
    # output_format MUST be a query parameter (ignored in the JSON body → MP3).
    with httpx.stream(
        "POST",
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream",
        params={"output_format": "pcm_16000"},
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        yield from resp.iter_bytes(chunk_size=4096)


def _resolve_el_voice(
    voice: str | None, stability_override: float | None
) -> tuple[str, float, float]:
    """Resolve a speaker's ElevenLabs voice_id + voice_settings.

    ``voice`` is an ElevenLabs voice_id (from ``actors.speaker_voice``); ``None``
    resolves to the nurse default. ``stability_override`` lets the Director drive
    affect from engine state (e.g. a distressed patient → lower stability)."""
    nurse_voice_id = os.getenv("ELEVENLABS_NURSE_VOICE_ID", _EL_NURSE_DEFAULT)
    voice_id = voice or nurse_voice_id
    if voice_id == nurse_voice_id:
        stability, similarity = _EL_NURSE_STABILITY, _EL_NURSE_SIMILARITY
    else:
        from backend.actors import el_settings_for_voice
        stability, similarity = el_settings_for_voice(voice_id)
    if stability_override is not None:
        stability = max(0.0, min(1.0, stability_override))
    return voice_id, stability, similarity


def _tts_model_for_tier(tier: str | None) -> str:
    """Map a launch-chooser speech tier to an ElevenLabs TTS model.

    ``extreme`` → the frontier expressive model (``eleven_v3``); everything else →
    Flash v2.5, the low-latency default that carries every ordinary live turn."""
    return EL_TTS_MODEL_V3 if (tier or "").lower() == "extreme" else EL_TTS_MODEL


# Style boost applied when the resident opts into Expressiveness (Extreme tier only).
# Higher `style` = more emotional variation/emphasis on the v3 frontier model.
EL_EXPRESSIVE_STYLE = 0.6


def _expressive_style(expressive: bool) -> float:
    return EL_EXPRESSIVE_STYLE if expressive else 0.0


def synthesize_stream(
    text: str, voice: str | None = None, stability_override: float | None = None,
    tier: str | None = None, expressive: bool = False,
) -> tuple[str, Any]:
    """Return ``('audio/pcm', PCM-16000 generator)`` for streaming TTS.

    Voice is ElevenLabs only. ``tier='extreme'`` selects the frontier model
    (``eleven_v3``); otherwise Flash v2.5, streamed PCM-16000 (the browser plays
    after the first ~75 ms chunk). ``expressive`` (Extreme only) raises the style so
    speech carries more emotion. Raises if ElevenLabs is unavailable; the room
    surfaces the error rather than silently degrading to another provider."""
    if not _elevenlabs_key():
        raise RuntimeError("ELEVENLABS_API_KEY is not set — voice requires ElevenLabs")
    voice_id, stability, similarity = _resolve_el_voice(voice, stability_override)
    model_id = _tts_model_for_tier(tier)
    style = _expressive_style(expressive)
    return "audio/pcm", _el_tts_stream(text, voice_id, stability, similarity, model_id, style)


def synthesize(text: str, voice: str | None = None, fmt: str = "mp3",
               stability_override: float | None = None, tier: str | None = None,
               expressive: bool = False) -> bytes:
    """TTS: text -> audio bytes via ElevenLabs (~75 ms on Flash v2.5).

    ``voice`` is an ElevenLabs voice_id; ``None`` resolves to the nurse default.
    ``tier='extreme'`` uses the frontier model (``eleven_v3``); ``expressive`` raises
    the style boost. Raises on ElevenLabs failure — there is no second provider."""
    if not _elevenlabs_key():
        raise RuntimeError("ELEVENLABS_API_KEY is not set — voice requires ElevenLabs")
    voice_id, stability, similarity = _resolve_el_voice(voice, stability_override)
    model_id = _tts_model_for_tier(tier)
    style = _expressive_style(expressive)
    return _el_tts_request(text, voice_id, stability, similarity, model_id, fmt, style)


def nurse_reply(transcript: str, history: list[dict] | None = None) -> str:
    """LLM: a resident utterance -> the nurse's spoken reply (via Cerebras).

    Rides the shared interactive runtime so the stateless nurse uses the same
    provider + timeout/retry the rest of the room does. No case context yet —
    that's the orchestration phase."""
    from backend.agent_runtime import complete

    context = ""
    if history:
        lines = []
        for turn in history[-6:]:
            who = "Resident" if turn.get("role") == "resident" else "Nurse"
            line = (turn.get("text") or "").strip()
            if line:
                lines.append(f"{who}: {line}")
        if lines:
            context = "Recent exchange:\n" + "\n".join(lines) + "\n\n"

    user_prompt = f"{context}Resident: {transcript}\nNurse:"
    return complete(NURSE_SYSTEM_PROMPT, user_prompt, kind="nurse")
