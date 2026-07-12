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

# Patient states where the patient is sedated / arrested and does not speak.
SILENT_STATES = {"post_intubation", "pea_arrest", "icu_admission"}


@dataclass(frozen=True)
class EmotionalState:
    descriptor: str           # short label: "steady" | "breathless" | "frightened" | …
    instruction: str          # prompt fragment for the character ("" = none)
    stability: float | None   # ElevenLabs stability override (None = use voice default)


def severity(engine) -> float:
    """0 (stable) … 1 (peri-arrest) from the patient's current acuity (vitals + state).

    Drives how much the *reactive* characters (patient, family) escalate. Reads
    SpO₂ and the state id — fail-soft to 0 if a field is missing."""
    sid = engine.run.current_state_id
    sev = 0.0
    try:
        spo2 = getattr(engine.current_vitals(), "oxygen_saturation", None)
    except Exception:
        spo2 = None
    if spo2 is not None:
        if spo2 < 85:
            sev = max(sev, 0.9)
        elif spo2 < 90:
            sev = max(sev, 0.7)
        elif spo2 < 94:
            sev = max(sev, 0.4)
    if "intubation" in sid and "post" not in sid:
        sev = max(sev, 0.9)
    try:
        if engine.state.is_terminal:
            sev = max(sev, 0.5)
    except Exception:
        pass
    return sev


def _patient_stability(engine) -> float:
    """Lower stability = more vocal affect. Calm at baseline, near-zero at failure."""
    sid = engine.run.current_state_id
    if sid in SILENT_STATES:
        return 0.30
    if "intubation" in sid and "post" not in sid:
        return 0.15
    elapsed = engine.elapsed_in_state
    if elapsed > 180:
        return 0.20
    if elapsed > 60:
        return 0.25
    return 0.30


def _patient_instruction(engine) -> str:
    sid = engine.run.current_state_id
    if sid in SILENT_STATES:
        return ("You are sedated and cannot speak. Do not reply with words — "
                "respond with silence (output an empty string).")
    elapsed = engine.elapsed_in_state
    if "intubation" in sid and "post" not in sid:
        return ("You are exhausted and barely able to breathe. Reply in at most "
                "four words. Every breath is a visible effort — do not form full "
                "sentences. You may trail off mid-word.")
    if elapsed > 180:
        return ("Your breathing has worsened over the last few minutes. Keep each "
                "reply to one very short phrase (≤6 words). Show increasing "
                "exhaustion and breathlessness.")
    if elapsed > 60:
        return ("You are noticeably more breathless than when the doctor arrived. "
                "Limit replies to short sentences and convey effort with each word.")
    return ("You are breathless and anxious but can still speak in short sentences. "
            "You cannot complete full sentences without pausing to breathe.")


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
        return EmotionalState("breathless", _patient_instruction(engine), _patient_stability(engine))
    if speaker == "family":
        return _family_state(engine)
    return EmotionalState("steady", _STEADY.get(speaker, ""), None)
