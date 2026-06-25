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
    voice_env: str = ""              # env var holding the TTS voice (empty → nurse default)
    default_voice: str = ""          # fallback voice when the env var is unset
    holds_history: bool = False      # may ground a history beat when addressed

    def voice(self) -> str | None:
        """The configured TTS voice, or None to fall back to the nurse default."""
        if not self.voice_env:
            return None
        return os.getenv(self.voice_env, self.default_voice)


# Defaults mirror .env.example. Cartesia Sonic voice names — if a name is rejected
# the provider, voice_service.synthesize falls back to the nurse voice.
SPEAKERS: tuple[SpeakerKind, ...] = (
    SpeakerKind("nurse", ("nurse",)),  # the nurse default voice lives in voice_service
    SpeakerKind(
        "patient", ("patient", "maam", "ma'am", "sir"),
        "TOGETHER_PATIENT_TTS_VOICE", "australian woman", holds_history=True,
    ),
    SpeakerKind(
        "ems", ("ems", "paramedic", "paramedics", "medic", "medics", "ambulance", "prehospital"),
        "TOGETHER_EMS_TTS_VOICE", "newsman", holds_history=True,
    ),
    SpeakerKind(
        "family",
        ("family", "husband", "wife", "partner", "spouse", "relative", "mother",
         "father", "parent", "son", "daughter", "carer", "caregiver", "mr", "mrs"),
        "TOGETHER_FAMILY_TTS_VOICE", "polite man", holds_history=True,
    ),
    SpeakerKind("rt", ("rt", "respiratory"), "TOGETHER_RT_TTS_VOICE", "wise man"),
    SpeakerKind(
        "consult", ("consult", "consultant", "specialist"),
        "TOGETHER_CONSULT_TTS_VOICE", "professional man",
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
