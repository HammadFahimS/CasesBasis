"""Thin model-I/O plumbing for the voice room.

Three independent legs, deliberately minimal (request/response, no streaming or
orchestration yet — that comes with the agent phase):

* STT  — Together AI Whisper (``/v1/audio/transcriptions``)
* LLM  — the app's existing provider (Cerebras) via ``build_llm_client``
* TTS  — Together AI Cartesia Sonic (``/v1/audio/speech``)

API keys stay server-side; the browser only ever talks to our endpoints.
Audio provider is Together for now; the STT/TTS functions are the seam to swap
to Deepgram later without touching the endpoints or the front-end.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

TOGETHER_BASE_URL = "https://api.together.ai/v1"
STT_MODEL = os.getenv("TOGETHER_STT_MODEL", "openai/whisper-large-v3")
TTS_MODEL = os.getenv("TOGETHER_TTS_MODEL", "cartesia/sonic-2")
TTS_VOICE = os.getenv("TOGETHER_TTS_VOICE", "british reading lady")

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


def transcribe(audio: bytes, content_type: str = "audio/webm") -> str:
    """STT: audio bytes -> transcript text (Together Whisper)."""
    ext = "webm"
    if "mp4" in content_type or "m4a" in content_type:
        ext = "mp4"
    elif "ogg" in content_type:
        ext = "ogg"
    elif "wav" in content_type:
        ext = "wav"
    resp = httpx.post(
        f"{TOGETHER_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {_together_key()}"},
        files={"file": (f"audio.{ext}", audio, content_type)},
        data={"model": STT_MODEL, "language": "en", "response_format": "json"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()


def synthesize(text: str, voice: str | None = None, fmt: str = "mp3") -> bytes:
    """TTS: text -> audio bytes (Together Cartesia Sonic).

    Voice resolves at call time (request override -> env -> default) so the
    nurse voice can be changed in .env without a code edit."""
    voice = voice or os.getenv("TOGETHER_TTS_VOICE", TTS_VOICE)
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


def nurse_reply(transcript: str, history: list[dict] | None = None) -> str:
    """LLM: a resident utterance -> the nurse's spoken reply (via Cerebras).

    Reuses the app's existing provider machinery so the nurse rides the same
    model the rest of CasesBasis uses. No case context yet — that's the
    orchestration phase."""
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    provider = get_llm_provider()
    client = build_llm_client(provider)
    model = resolve_model(provider, None)

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
    return client.text_response(
        model=model, system_prompt=NURSE_SYSTEM_PROMPT, user_prompt=user_prompt
    ).strip()
