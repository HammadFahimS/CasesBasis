"""Thin model-I/O plumbing for the voice room.

Three independent legs, deliberately minimal (request/response, no streaming or
orchestration yet — that comes with the agent phase):

* STT  — Together AI Whisper (``/v1/audio/transcriptions``)
* LLM  — the app's existing provider (Cerebras) via ``agent_runtime.complete``
* TTS  — ElevenLabs Flash v2.5 (~75 ms, primary) / Together Cartesia Sonic-2 (fallback)

ElevenLabs is active whenever ``ELEVENLABS_API_KEY`` is set. Flash v2.5
(``eleven_flash_v2_5``) is used for all live chatty turns — it targets ~75 ms
latency and is about 50 % cheaper per character than the expressive models.
Together/Cartesia remains the fallback path so the room never goes silent if
ElevenLabs is unreachable.

API keys stay server-side; the browser only ever talks to our endpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Together / Cartesia (fallback / primary when EL key absent) ───────────────
TOGETHER_BASE_URL = "https://api.together.ai/v1"
STT_MODEL = os.getenv("TOGETHER_STT_MODEL", "openai/whisper-large-v3")
TTS_MODEL = os.getenv("TOGETHER_TTS_MODEL", "cartesia/sonic-2")
TTS_VOICE = os.getenv("TOGETHER_TTS_VOICE", "british reading lady")

# ── ElevenLabs ────────────────────────────────────────────────────────────────
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
EL_TTS_MODEL = "eleven_flash_v2_5"   # ~75 ms, low-latency; all live chatty turns
EL_TTS_MODEL_V3 = "eleven_v3"        # expressive + audio tags; reserved for set-pieces

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


def _together_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TOGETHER_API_KEY is not set")
    return key


def _elevenlabs_key() -> str | None:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip() or None


def transcribe(audio: bytes, content_type: str = "audio/webm",
               prompt: str | None = None) -> str:
    """STT: audio bytes -> transcript text (Together Whisper).

    ``prompt`` seeds Whisper's vocabulary — pass clinical terms (drug names,
    test names, actor roles) to dramatically improve recognition accuracy."""
    ext = "webm"
    if "mp4" in content_type or "m4a" in content_type:
        ext = "mp4"
    elif "ogg" in content_type:
        ext = "ogg"
    elif "wav" in content_type:
        ext = "wav"
    data: dict[str, str] = {"model": STT_MODEL, "language": "en", "response_format": "json"}
    if prompt:
        data["prompt"] = prompt
    resp = httpx.post(
        f"{TOGETHER_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {_together_key()}"},
        files={"file": (f"audio.{ext}", audio, content_type)},
        data=data,
        timeout=60.0,
    )
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()


def _tts_request(text: str, voice: str, fmt: str) -> bytes:
    """Together/Cartesia TTS request."""
    resp = httpx.post(
        f"{TOGETHER_BASE_URL}/audio/speech",
        headers={
            "Authorization": f"Bearer {_together_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": TTS_MODEL,
            "input": text,
            "voice": voice,
            "response_format": fmt,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content


def _el_tts_request(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
    model_id: str,
    fmt: str,
) -> bytes:
    """ElevenLabs TTS request.

    ``fmt='mp3'`` maps to ``mp3_44100_128`` (128 kbps, 44.1 kHz). Pass a raw
    ElevenLabs format string (e.g. ``pcm_16000``) directly for other formats."""
    output_format = "mp3_44100_128" if fmt == "mp3" else fmt
    resp = httpx.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "output_format": output_format,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content


def _el_tts_stream(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
):
    """Yield raw PCM-16000 chunks from ElevenLabs /stream endpoint.

    Format: 16 kHz, 16-bit signed little-endian, mono.  The caller converts
    these to Float32 AudioBuffers for scheduling on the Web Audio API timeline."""
    with httpx.stream(
        "POST",
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream",
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": EL_TTS_MODEL,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "output_format": "pcm_16000",
        },
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        yield from resp.iter_bytes(chunk_size=4096)


def synthesize_stream(
    text: str, voice: str | None = None, stability_override: float | None = None
) -> tuple[str, Any]:
    """Return (media_type, byte_generator) for streaming TTS.

    ElevenLabs (when key present): ``('audio/pcm', PCM-16000 generator)``
    Together fallback:              ``('audio/mpeg', single-chunk generator)``
    """
    el_key = _elevenlabs_key()
    if el_key:
        nurse_voice_id = os.getenv("ELEVENLABS_NURSE_VOICE_ID", _EL_NURSE_DEFAULT)
        voice_id = voice or nurse_voice_id
        if voice_id == nurse_voice_id:
            stability, similarity = _EL_NURSE_STABILITY, _EL_NURSE_SIMILARITY
        else:
            from backend.actors import el_settings_for_voice
            stability, similarity = el_settings_for_voice(voice_id)
        if stability_override is not None:
            stability = max(0.0, min(1.0, stability_override))
        return "audio/pcm", _el_tts_stream(text, voice_id, stability, similarity)

    def _together() -> Any:
        yield synthesize(text, voice, stability_override=stability_override)

    return "audio/mpeg", _together()


def synthesize(text: str, voice: str | None = None, fmt: str = "mp3",
               stability_override: float | None = None) -> bytes:
    """TTS: text -> audio bytes.

    When ``ELEVENLABS_API_KEY`` is set, uses ElevenLabs Flash v2.5 (~75 ms).
    Falls back to Together/Cartesia on ElevenLabs failure or when the key is absent.

    ``voice`` is an ElevenLabs voice_id when ElevenLabs is active, or a
    Cartesia voice name otherwise. ``None`` resolves to the nurse default for
    the active provider. A bad voice must never silence the room — errors fall
    through to the next provider automatically."""
    el_key = _elevenlabs_key()
    if el_key:
        nurse_voice_id = os.getenv("ELEVENLABS_NURSE_VOICE_ID", _EL_NURSE_DEFAULT)
        voice_id = voice or nurse_voice_id
        if voice_id == nurse_voice_id:
            stability, similarity = _EL_NURSE_STABILITY, _EL_NURSE_SIMILARITY
        else:
            from backend.actors import el_settings_for_voice
            stability, similarity = el_settings_for_voice(voice_id)
        if stability_override is not None:
            stability = max(0.0, min(1.0, stability_override))
        try:
            return _el_tts_request(text, voice_id, stability, similarity, EL_TTS_MODEL, fmt)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            logger.warning(
                "ElevenLabs TTS voice %r failed (%s); falling back to Together",
                voice_id, status or exc,
            )

    # Together / Cartesia — primary when EL key absent, fallback otherwise.
    # When falling back from ElevenLabs, ``voice`` holds an EL voice_id (not a
    # Cartesia name), so always use the Together default in that case.
    default = os.getenv("TOGETHER_TTS_VOICE", TTS_VOICE)
    together_voice = default if el_key else (voice or default)
    try:
        return _tts_request(text, together_voice, fmt)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if together_voice != default and 400 <= status < 500:
            logger.warning(
                "TTS voice %r rejected (%s); falling back to default voice", together_voice, status
            )
            return _tts_request(text, default, fmt)
        raise


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
