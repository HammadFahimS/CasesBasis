"""Thin model-I/O plumbing for the voice room.

Three independent legs:

* STT  — ElevenLabs Scribe (``scribe_v1``, ``/v1/speech-to-text``)
* LLM  — the app's existing provider (Cerebras) via ``agent_runtime.complete``
* TTS  — ElevenLabs Eleven v4 Turbo (``eleven_v4_turbo``), streamed PCM-16000

ElevenLabs is the sole voice provider — it requires ``ELEVENLABS_API_KEY``.
v4 Turbo is used for every live turn: it is real-time-rated (~100 ms inference),
expressive through audio tags, billed at Flash rates, and streams PCM the browser
plays after the first chunk. ``ELEVENLABS_TTS_MODEL`` overrides it (e.g. back to
``eleven_flash_v2_5``) without a deploy. There is no second voice vendor; a
voice failure surfaces as an error rather than silently degrading. (The LLM
provider is separate and unaffected.)

API keys stay server-side; the browser only ever talks to our endpoints.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── ElevenLabs — the sole voice provider (TTS + STT) ─────────────────────────
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
ELEVENLABS_WS_BASE = "wss://api.elevenlabs.io/v1"

# A single, connection-pooled client shared by every STT and TTS call. Without
# this, each `httpx.post`/`httpx.stream` opened a *fresh* TLS connection to
# ElevenLabs — ~100–300 ms of handshake baked into every hop, every turn. The
# real killer is httpx's default ``keepalive_expiry`` of only 5 s: any gap while
# the resident is talking or listening dropped the connection, so the next turn
# paid the handshake again. A 5-minute keepalive keeps the connection warm
# across normal between-turn idle. The client is thread-safe (pool-locked), so
# it is safe to share across the worker's request threads.
_EL_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_EL_LIMITS = httpx.Limits(
    max_keepalive_connections=8, max_connections=16, keepalive_expiry=300.0
)
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    """The shared, warm-connection ElevenLabs client (built once, lazily)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(timeout=_EL_TIMEOUT, limits=_EL_LIMITS)
    return _client

EL_TTS_MODEL = "eleven_v4_turbo"     # real-time + audio tags, 0.5× credits; every live turn
EL_TTS_MODEL_FLASH = "eleven_flash_v2_5"  # rollback via ELEVENLABS_TTS_MODEL (no audio tags)
EL_STT_MODEL = "scribe_v1"           # ElevenLabs Scribe — blocking speech-to-text
EL_STT_REALTIME_MODEL = "scribe_v2_realtime"  # Scribe v2 — streaming STT over WebSocket

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
            resp = _http().post(
                f"{ELEVENLABS_BASE_URL}/speech-to-text",
                headers={"xi-api-key": key},
                files={"file": (f"audio.{ext}", audio, content_type)},
                data={"model_id": EL_STT_MODEL, "language_code": "en", "tag_audio_events": "false"},
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


# ElevenLabs rejects the *whole* realtime connection with `invalid_request` if any
# single keyterm exceeds this — not just the offending term. The case vocabulary
# (`routers.voice._stt_prompt`) was written as a Whisper prompt and happily contains
# phrases like "Negative inspiratory force" (26 chars), so it must be normalised
# before it goes on the wire.
_KEYTERM_MAX_CHARS = 20
_KEYTERM_STOPWORDS = {"with", "and", "the", "of", "for", "a", "an", "on", "to"}


def _normalize_keyterms(raw: list[str] | None, limit: int = 32) -> list[str]:
    """Coerce case vocabulary into keyterms ElevenLabs will accept.

    A term within the limit is kept whole. An over-long phrase is split into its
    words and the distinctive ones kept — "Negative inspiratory force" becomes
    "Negative", "inspiratory", "force", which still biases recognition toward the
    clinical vocabulary that matters. Stopwords and duplicates are dropped so the
    budget goes on real terms.
    """
    out: list[str] = []
    seen: set[str] = set()
    for term in raw or []:
        term = (term or "").strip()
        if not term:
            continue
        parts = [term] if len(term) <= _KEYTERM_MAX_CHARS else term.split()
        for part in parts:
            part = part.strip(" ()[]{},.;:").strip()
            key = part.lower()
            if not part or len(part) > _KEYTERM_MAX_CHARS:
                continue
            if key in _KEYTERM_STOPWORDS or key in seen:
                continue
            seen.add(key)
            out.append(part)
            if len(out) >= limit:
                return out
    return out


async def open_realtime_stt(keyterms: list[str] | None = None) -> Any:
    """Open a Scribe v2 Realtime STT WebSocket to ElevenLabs and return the
    connected client. The caller relays audio and reads transcripts.

    Contract we set up here:
      * ``pcm_16000`` mono 16-bit input (the browser resamples to 16 kHz)
      * ``commit_strategy=manual`` — *we* decide when an utterance ends (our
        client-side VAD already does turn-taking), so ElevenLabs only finalises
        a transcript when we send a chunk with ``commit: true``.
      * ``keyterms`` — the case's clinical vocabulary, biasing recognition of
        drug/test/actor names (the old blocking path could not do this). Passed
        through :func:`_normalize_keyterms`, because one over-long term rejects
        the whole connection.

    Raises if the API key is missing or the socket cannot be opened; the caller
    falls back to the blocking :func:`transcribe` path in that case."""
    import urllib.parse

    import websockets  # declared dep (also pulled by uvicorn[standard])

    key = _elevenlabs_key()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set — realtime STT requires ElevenLabs")
    params: list[tuple[str, str]] = [
        ("model_id", EL_STT_REALTIME_MODEL),
        ("audio_format", "pcm_16000"),
        ("language_code", "en"),
        ("commit_strategy", "manual"),
    ]
    for kt in _normalize_keyterms(keyterms):
        params.append(("keyterms", kt))
    url = f"{ELEVENLABS_WS_BASE}/speech-to-text/realtime?{urllib.parse.urlencode(params)}"
    return await websockets.connect(
        url, additional_headers={"xi-api-key": key}, max_size=None, open_timeout=10
    )


def _el_tts_request(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
    model_id: str,
    fmt: str,
    tag: str | None = None,
) -> bytes:
    """ElevenLabs TTS request.

    ``fmt='mp3'`` maps to ``mp3_44100_128`` (128 kbps, 44.1 kHz). Pass a raw
    ElevenLabs format string (e.g. ``pcm_16000``) directly for other formats.
    ``tag`` is a delivery audio tag (``breathless``) prefixed as ``[breathless]``."""
    from backend.speech_normalize import normalize_for_speech
    from backend.pronunciation import tts_locators
    text = _tagged(normalize_for_speech(text), tag)
    output_format = "mp3_44100_128" if fmt == "mp3" else fmt
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "use_speaker_boost": True,
        },
    }
    locators = tts_locators()
    if locators:
        body["pronunciation_dictionary_locators"] = locators
    # NB: ElevenLabs reads output_format from the QUERY STRING. Putting it in the
    # JSON body is silently ignored and the API defaults to MP3 — which, on the
    # PCM path, the browser decodes as raw samples (loud static).
    resp = _http().post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
        params={"output_format": output_format},
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json=body,
    )
    resp.raise_for_status()
    return resp.content


def _el_tts_stream(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
    model_id: str = EL_TTS_MODEL,
    tag: str | None = None,
):
    """Yield raw PCM-16000 chunks from ElevenLabs /stream endpoint.

    Format: 16 kHz, 16-bit signed little-endian, mono.  The caller converts
    these to Float32 AudioBuffers for scheduling on the Web Audio API timeline."""
    from backend.speech_normalize import normalize_for_speech
    from backend.pronunciation import tts_locators
    text = _tagged(normalize_for_speech(text), tag)
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "use_speaker_boost": True,
        },
    }
    locators = tts_locators()
    if locators:
        body["pronunciation_dictionary_locators"] = locators
    # output_format MUST be a query parameter (ignored in the JSON body → MP3).
    with _http().stream(
        "POST",
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream",
        params={"output_format": "pcm_16000"},
        headers={
            "xi-api-key": _elevenlabs_key(),
            "Content-Type": "application/json",
        },
        json=body,
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


def tts_model() -> str:
    """The ElevenLabs TTS model for every live line: ``ELEVENLABS_TTS_MODEL`` if set
    (the no-deploy rollback, e.g. to ``eleven_flash_v2_5``), else v4 Turbo."""
    return os.environ.get("ELEVENLABS_TTS_MODEL", "").strip() or EL_TTS_MODEL


def supports_audio_tags(model_id: str) -> bool:
    """Only the v3/v4 families interpret ``[tags]``; Flash would read them aloud."""
    return model_id.startswith(("eleven_v3", "eleven_v4"))


def _tagged(text: str, tag: str | None) -> str:
    return f"[{tag}] {text}" if tag and text else text


def _affect_tag(affect: str | None, model_id: str) -> str | None:
    """The audio tag for a speaker's EmotionalState descriptor, on a model that
    honours tags. Looked up from a fixed table, so a client can only name a
    descriptor — never inject tag text."""
    if not affect or not supports_audio_tags(model_id):
        return None
    from backend.emotional_state import audio_tag
    return audio_tag(affect)


def synthesize_stream(
    text: str, voice: str | None = None, stability_override: float | None = None,
    affect: str | None = None,
) -> tuple[str, Any]:
    """Return ``('audio/pcm', PCM-16000 generator)`` for streaming TTS.

    Voice is ElevenLabs only, on :func:`tts_model`, streamed PCM-16000 (the browser
    plays after the first chunk). ``affect`` is the speaker's EmotionalState
    descriptor; on a tag-capable model it becomes a delivery audio tag. Raises if
    ElevenLabs is unavailable; the room surfaces the error rather than silently
    degrading to another provider."""
    if not _elevenlabs_key():
        raise RuntimeError("ELEVENLABS_API_KEY is not set — voice requires ElevenLabs")
    voice_id, stability, similarity = _resolve_el_voice(voice, stability_override)
    model_id = tts_model()
    tag = _affect_tag(affect, model_id)
    return "audio/pcm", _el_tts_stream(text, voice_id, stability, similarity, model_id, tag)


def synthesize(text: str, voice: str | None = None, fmt: str = "mp3",
               stability_override: float | None = None,
               affect: str | None = None) -> bytes:
    """TTS: text -> audio bytes via ElevenLabs on :func:`tts_model`.

    ``voice`` is an ElevenLabs voice_id; ``None`` resolves to the nurse default.
    ``affect`` is as for :func:`synthesize_stream`. Raises on ElevenLabs failure —
    there is no second provider."""
    if not _elevenlabs_key():
        raise RuntimeError("ELEVENLABS_API_KEY is not set — voice requires ElevenLabs")
    voice_id, stability, similarity = _resolve_el_voice(voice, stability_override)
    model_id = tts_model()
    tag = _affect_tag(affect, model_id)
    return _el_tts_request(text, voice_id, stability, similarity, model_id, fmt, tag)


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
