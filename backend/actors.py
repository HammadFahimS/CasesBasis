"""The room's cast — one registry of speaker *kinds*.

A "speaker kind" (``ems``/``family``/``patient``/``rt``/``consult``/``nurse``) is
the tag the front-end styles, the TTS keys a voice on, and the router decides
when the resident addresses someone. That knowledge used to be restated in four
places — the provider→speaker map and ``actor_speaker`` (actor_voice), the
vocative table (nurse_brain) and the TTS voice map (web_app). This module is the
single source of truth for it.

It governs only *who voices* a line and *with what voice*. The
:class:`~backend.simulation_engine.SimulationEngine` remains the single source of
truth for every clinical fact; nothing here invents or grounds content.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerKind:
    id: str                          # 'ems' — the styling/TTS/router tag
    synonyms: tuple[str, ...]        # words that name this person (provider tag or vocative)
    voice_env: str = ""              # Together/Cartesia env var for voice name
    default_voice: str = ""          # Cartesia fallback voice name
    holds_history: bool = False      # may ground a history beat when addressed
    el_voice_env: str = ""           # ElevenLabs voice ID env var
    el_default_voice: str = ""       # ElevenLabs voice ID fallback
    el_stability: float = 0.5        # ElevenLabs voice_settings stability (0=expressive, 1=flat)
    el_similarity: float = 0.75      # ElevenLabs voice_settings similarity_boost

    def el_voice(self) -> str | None:
        """The configured ElevenLabs voice ID, or None (nurse default in voice_service)."""
        if not self.el_voice_env:
            return None
        return os.getenv(self.el_voice_env, self.el_default_voice) or None

    def voice(self) -> str | None:
        """The configured TTS voice for the active provider.

        Returns an ElevenLabs voice ID when ELEVENLABS_API_KEY is set; otherwise
        returns the Together/Cartesia voice name. None always signals 'use the
        nurse default in voice_service'.
        """
        if os.environ.get("ELEVENLABS_API_KEY"):
            return self.el_voice()
        if not self.voice_env:
            return None
        return os.getenv(self.voice_env, self.default_voice)


# ElevenLabs voice IDs are from the pre-built voice library (stable across accounts).
# Override any per-actor ID via the ELEVENLABS_*_VOICE_ID env var without a code change.
# Together/Cartesia names are kept as the fallback path (TOGETHER_*_TTS_VOICE env vars).
SPEAKERS: tuple[SpeakerKind, ...] = (
    SpeakerKind("nurse", ("nurse",)),   # nurse default voice lives in voice_service
    SpeakerKind(
        "patient", ("patient", "maam", "ma'am", "sir"),
        "TOGETHER_PATIENT_TTS_VOICE", "australian woman",
        holds_history=True,
        el_voice_env="ELEVENLABS_PATIENT_VOICE_ID",
        el_default_voice="EXAVITQu4vr4xnSDxMaL",   # Bella — soft, warm female
        el_stability=0.30,   # low: emotionally variable, distressed
        el_similarity=0.75,
    ),
    SpeakerKind(
        "ems", ("ems", "paramedic", "paramedics", "medic", "medics", "ambulance", "prehospital"),
        "TOGETHER_EMS_TTS_VOICE", "newsman",
        holds_history=True,
        el_voice_env="ELEVENLABS_EMS_VOICE_ID",
        el_default_voice="TxGEqnHWrfWFTfGW9XjX",   # Josh — deep, brisk male
        el_stability=0.65,
        el_similarity=0.75,
    ),
    SpeakerKind(
        "family",
        ("family", "husband", "wife", "partner", "spouse", "relative", "mother",
         "father", "parent", "son", "daughter", "carer", "caregiver", "mr", "mrs"),
        "TOGETHER_FAMILY_TTS_VOICE", "polite man",
        holds_history=True,
        el_voice_env="ELEVENLABS_FAMILY_VOICE_ID",
        el_default_voice="GBv7mTt0atIp3Br8iCZE",   # Thomas — calm, concerned male
        el_stability=0.45,   # slightly variable: worried
        el_similarity=0.75,
    ),
    SpeakerKind(
        "rt", ("rt", "respiratory"), "TOGETHER_RT_TTS_VOICE", "wise man",
        el_voice_env="ELEVENLABS_RT_VOICE_ID",
        el_default_voice="IKne3meq5aSn9XLyUdCD",   # Charlie — Australian male, practical
        el_stability=0.60,
        el_similarity=0.75,
    ),
    SpeakerKind(
        "consult", ("consult", "consultant", "specialist"),
        "TOGETHER_CONSULT_TTS_VOICE", "professional man",
        el_voice_env="ELEVENLABS_CONSULT_VOICE_ID",
        el_default_voice="pNInz6obpgDQGcFmaJgB",   # Adam — deep, measured male; senior specialist
        el_stability=0.70,   # high: authoritative, consistent
        el_similarity=0.80,
    ),
)

_BY_ID = {s.id: s for s in SPEAKERS}
_BY_SYN = {syn: s for s in SPEAKERS for syn in s.synonyms}


def kind_for_word(word: str | None) -> SpeakerKind | None:
    """The speaker kind named by a single word (a provider tag or a vocative)."""
    return _BY_SYN.get((word or "").strip().lower())


def speaker_voice(speaker_id: str | None) -> str | None:
    """The configured TTS voice for a speaker tag (None → nurse default)."""
    kind = _BY_ID.get((speaker_id or "").strip().lower())
    return kind.voice() if kind else None


def el_settings_for_voice(voice_id: str) -> tuple[float, float]:
    """Return (stability, similarity_boost) for an ElevenLabs voice_id.

    Scans the SPEAKERS registry for a matching el_voice(). Returns the default
    tuple (0.5, 0.75) when the voice_id belongs to the nurse (handled in
    voice_service) or is otherwise not found."""
    for s in SPEAKERS:
        if s.el_voice() == voice_id:
            return s.el_stability, s.el_similarity
    return 0.5, 0.75
