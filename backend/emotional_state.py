"""Live EmotionalState for the room's characters (multi-agent-plan Phase 5).

A character's *emotional state* is the part that varies turn-to-turn with the
patient's acuity. A breathless patient gives shorter, more distressed lines and is
voiced with lower TTS stability (more affect); a frightened relative speaks with a
little more worry as their loved one deteriorates; the professionals (EMS, RT,
consultant, nurse) stay steady. This is the ``EmotionalState`` term of
``CharacterAgent = Constitution + CaseFlavor + EmotionalState + VoiceBinding``
(see :mod:`backend.agents`): computed live from the engine each turn and handed to
both the prompt (an `instruction`) and the voice (a `stability` override).

The engine remains the single source of clinical truth — nothing here invents a
value; it only reads vitals/state to colour *how* a line is said. The patient logic
is the same escalation that previously lived in :mod:`backend.patient_voice`, now
generalised so every addressed actor resolves an EmotionalState the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

# Non-perfusing rhythms: the patient is pulseless and cannot speak.
_ARRESTED_RHYTHMS = ("pea", "asystole", "vf", "fibrillation", "pulseless")


@dataclass(frozen=True)
class EmotionalState:
    descriptor: str           # short label: "steady" | "breathless" | "frightened" | …
    instruction: str          # prompt fragment for the character ("" = none)
    stability: float | None   # ElevenLabs stability override (None = use voice default)


def _vitals(engine):
    try:
        return engine.current_vitals()
    except Exception:
        return None


def _gcs_number(vitals) -> int | None:
    raw = str(getattr(vitals, "gcs", "") or "").strip()
    return int(raw) if raw.isdigit() else None


def is_silent(engine) -> bool:
    """The patient cannot speak: intubated/sedated, pulseless, or GCS ≤ 8. Read
    from the vitals the current state carries, so any case's arrest, intubation or
    death state is silent without naming it in code."""
    v = _vitals(engine)
    if v is None:
        return False
    gcs = str(v.gcs or "").lower()
    rhythm = str(v.rhythm or "").lower()
    if "intubat" in gcs or "sedat" in gcs or "ventilat" in gcs:
        return True
    if any(r in rhythm for r in _ARRESTED_RHYTHMS):
        return True
    n = _gcs_number(v)
    return n is not None and n <= 8


def severity(engine) -> float:
    """0 (stable) … 1 (peri-arrest) from the patient's current acuity — SpO₂ bands,
    a depressed GCS, a non-perfusing rhythm. Drives how much the
    *reactive* characters (patient, family) escalate. Fail-soft to 0."""
    sev = 0.0
    v = _vitals(engine)
    if v is not None:
        spo2 = v.oxygen_saturation
        if spo2 is not None:
            if spo2 < 85:
                sev = max(sev, 0.9)
            elif spo2 < 90:
                sev = max(sev, 0.7)
            elif spo2 < 94:
                sev = max(sev, 0.4)
        n = _gcs_number(v)
        if n is not None and n <= 8:
            sev = max(sev, 0.9)
        elif n is not None and n <= 12:
            sev = max(sev, 0.7)
        if any(r in str(v.rhythm or "").lower() for r in _ARRESTED_RHYTHMS):
            sev = 1.0
    return sev


def _patient_stability(engine) -> float:
    """Lower stability = more vocal affect: 0.30 when calm, 0.15 peri-arrest."""
    if is_silent(engine):
        return 0.30
    return round(0.30 - 0.15 * severity(engine), 2)


def _patient_instruction(engine) -> str:
    """How the patient speaks this turn, from acuity. The authored ``patient_status``
    (already in the patient prompt) carries the case-specific picture; this only
    sets the register — silent, drowsy, distressed, or as the status describes."""
    if is_silent(engine):
        return ("You are unconscious or sedated and cannot speak. Do not reply with "
                "words — respond with silence (output an empty string).")
    sev = severity(engine)
    if sev >= 0.7:
        return ("You are drowsy and barely able to breathe. Reply in at most four "
                "words, mumbled. Every breath is a visible effort — do not form full "
                "sentences. You may trail off mid-word or not answer at all.")
    if sev >= 0.4:
        return ("You are noticeably more breathless and frightened than when the "
                "doctor arrived. Limit replies to short phrases and convey effort "
                "with each word.")
    return ("Speak as your current condition describes: if you are short of breath, "
            "keep to short sentences and pause to breathe; if you are comfortable, "
            "speak normally. Answer only what you are asked.")


def _family_state(engine) -> EmotionalState:
    """A relative's worry tracks the patient's acuity — steadier than the patient,
    but their voice warms/tightens as their loved one deteriorates."""
    sev = severity(engine)
    stability = round(0.45 - 0.15 * sev, 2)   # 0.45 composed → 0.30 when peri-arrest
    if sev >= 0.7:
        return EmotionalState(
            "frightened",
            "You are frightened watching your loved one struggle to breathe. Keep it "
            "short and worried; answer the question, don't make a speech.",
            stability,
        )
    if sev >= 0.4:
        return EmotionalState(
            "worried",
            "You are worried — your loved one looks worse. Answer plainly, with concern.",
            stability,
        )
    return EmotionalState("concerned", "You are concerned but composed.", stability)


# The professionals stay steady; their voice keeps its configured default stability.
_STEADY: dict[str, str] = {
    "ems": "You are a calm, matter-of-fact professional giving a clean handover.",
    "rt": "You are a calm, practical respiratory therapist.",
    "consult": "You are a measured, authoritative specialist.",
    "head": "You are the head physician — calm, authoritative and brief.",
    "nurse": "",
}


def for_speaker(speaker: str, engine) -> EmotionalState:
    """The live EmotionalState for who is speaking this turn."""
    if speaker == "patient":
        sev = severity(engine)
        label = "silent" if is_silent(engine) else \
            "drowsy" if sev >= 0.7 else "breathless" if sev >= 0.4 else "steady"
        return EmotionalState(label, _patient_instruction(engine), _patient_stability(engine))
    if speaker == "family":
        return _family_state(engine)
    return EmotionalState("steady", _STEADY.get(speaker, ""), None)
