"""Engine-aware nurse: a spoken resident utterance -> a grounded reply.

The engine is the single source of truth. The LLM only *interprets* what the
resident said and *phrases* the result; it never invents a value.

Per turn (:func:`run_turn`):

1. ``advance_clock`` — the world moves forward (time/inaction transitions fire).
2. :func:`interpret` — **one LLM call** maps the utterance to a structured
   :class:`NurseIntent` (which engine op + target), chosen from the case's real
   catalogue of history beats / investigations / expected actions.
3. :func:`dispatch` — **pure, deterministic** — runs the op on the engine. No
   LLM here, so the core is unit-tested offline.
4. :func:`speak` — **one LLM call** phrases the nurse's reply from the engine's
   real outcome.

All LLM access goes through the single :func:`_complete` seam, which tests
monkeypatch to exercise the deterministic dispatch without network calls.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from backend.medication import needs_dose_confirm, parse_med
from backend.sim_diagnostics import (
    build_exam_card,
    build_investigation_card,
    build_order_card,
)
from backend.simulation_engine import (
    IntentMatcher,
    LearnerAction,
    SimulationEngine,
    StepResult,
)
from backend.voice_session import VoiceSession, advance_clock

logger = logging.getLogger(__name__)

# Ops the nurse can route an utterance to. Kept small and engine-aligned.
OPS = {
    "read_vitals",
    "ask_history",
    "order_investigation",
    "perform_exam",
    "order_medication",
    "perform_action",
    "request_consult",
    "diagnose",
    "chitchat",
}

INTERPRET_SYSTEM = (
    "You are the routing brain of a bedside nurse in an emergency-department "
    "simulation. The physician (resident) speaks to you; classify their single "
    "utterance into ONE action the nurse would take. Reply with STRICT JSON only "
    "— no prose, no markdown — using exactly these keys:\n"
    '  {"op": <one of read_vitals|ask_history|order_investigation|perform_exam|order_medication|perform_action|request_consult|diagnose|chitchat>,\n'
    '   "target": <id/name from the lists below, or null>,\n'
    '   "action_text": <short paraphrase of the order, or "">}\n'
    "Rules: read_vitals for any request about current vital signs / how the "
    "patient looks on the monitor. ask_history to elicit history from the patient/"
    "EMS/family (pick the closest history id). order_investigation for tests/labs/"
    "imaging (pick the closest investigation id). perform_exam to physically "
    "examine the patient (pick the closest body-system name as target). "
    "request_consult to call, page or refer to a specialist (pick the closest "
    "consultant id as target). order_medication for giving a drug or medication — "
    "set target to the drug name and action_text to the dose/route if stated. "
    "perform_action for any other treatment or intervention that is not a drug "
    "(e.g. intubate, CPR, defibrillate, position the patient) — set action_text. "
    "diagnose when the resident commits a working or differential diagnosis "
    "('I think this is …', 'my working diagnosis is …') — set action_text to the "
    "diagnosis. chitchat for anything social or unclear. "
    "Only use an id/name that appears in the lists."
)

SPEAK_SYSTEM = (
    "You are the bedside nurse in an emergency-department simulation, speaking "
    "aloud to the resident. You are calm, competent and concise. Read back orders "
    "the way a real nurse would and report findings plainly. You are given the "
    "REAL result of what just happened — report it faithfully and never invent "
    "numbers or findings beyond it. If a result is pending or unavailable, say so. "
    "No markdown, no asterisks, no stage directions — your words are spoken. Keep "
    "it under ~40 words."
)


@dataclass
class NurseIntent:
    """The classified action behind one resident utterance."""

    op: str = "chitchat"
    target: str | None = None
    action_text: str = ""


@dataclass
class NurseTurn:
    """The full grounded outcome of one turn."""

    reply: str
    intent: NurseIntent
    vitals: dict
    state_id: str
    state_name: str
    revealed: str | None = None
    transitioned: bool = False
    critical_error: bool = False
    ended: bool = False
    clock_events: list[str] = field(default_factory=list)
    diagnostic: dict | None = None  # a drawer card (ECG/CXR/report/exam), if any
    speaker: str = "nurse"          # who voices the reply — nurse, or "consult" for a specialist
    working_dx: str | None = None   # the resident's latest committed working diagnosis
    cues: list[dict] = field(default_factory=list)  # proactive prompts that fired this turn


def _complete(system_prompt: str, user_prompt: str) -> str:
    """The single LLM seam (Cerebras via the app's provider). Monkeypatched in tests."""
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    provider = get_llm_provider()
    client = build_llm_client(provider)
    model = resolve_model(provider, None)
    return client.text_response(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt
    ).strip()


def _catalogue(engine: SimulationEngine) -> str:
    """The real, current menu of ids the interpreter is allowed to pick from."""
    case = engine.case
    state = engine.state
    lines = ["History beats (id — who: label):"]
    lines += [f"  {h.id} — {h.provider}: {h.label}" for h in case.history] or ["  (none)"]
    lines.append("Investigations (id — name [category]):")
    lines += [f"  {i.id} — {i.name} [{i.category}]" for i in case.investigations] or ["  (none)"]
    lines.append("Examinable body systems (use the name as target):")
    lines += [f"  {e.system}" for e in case.exam] or ["  (none)"]
    from backend.consult_voice import consultants
    lines.append("Consultants you can call (id — role):")
    lines += [f"  {a.id} — {a.role}" for a in consultants(case)] or ["  (none)"]
    lines.append(f"Current state: {state.name}. Patient: {state.patient_status or 'see vitals'}.")
    if state.expected_actions:
        lines.append("Reasonable next actions here: " + "; ".join(state.expected_actions))
    return "\n".join(lines)


def _history_lines(history: list[dict], limit: int = 6) -> str:
    out = []
    for turn in history[-limit:]:
        who = "Resident" if turn.get("role") == "resident" else "Nurse"
        line = (turn.get("text") or "").strip()
        if line:
            out.append(f"{who}: {line}")
    return "\n".join(out)


def _parse_intent(raw: str) -> NurseIntent:
    """Defensively pull a NurseIntent out of the model's JSON-ish reply."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return NurseIntent(op="chitchat")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return NurseIntent(op="chitchat")
    op = data.get("op")
    if op not in OPS:
        op = "chitchat"
    target = data.get("target")
    if target in ("", "null", None):
        target = None
    return NurseIntent(op=op, target=str(target) if target else None,
                       action_text=(data.get("action_text") or "").strip())


def interpret(transcript: str, engine: SimulationEngine, history: list[dict]) -> NurseIntent:
    """Map one utterance to a structured engine intent (one LLM call)."""
    user = (
        f"{_catalogue(engine)}\n\n"
        f"Recent exchange:\n{_history_lines(history) or '(none)'}\n\n"
        f'Resident just said: "{transcript}"\n\nJSON:'
    )
    try:
        return _parse_intent(_complete(INTERPRET_SYSTEM, user))
    except Exception as exc:  # never let interpretation break the turn
        logger.warning("nurse interpret failed: %s", exc)
        return NurseIntent(op="chitchat")


def dispatch(intent: NurseIntent, engine: SimulationEngine, transcript: str) -> StepResult | None:
    """Run the intent on the engine. Pure/deterministic — no LLM.

    Returns the engine's :class:`StepResult` (None for read_vitals/chitchat,
    which don't mutate state)."""
    if intent.op == "ask_history" and intent.target:
        return engine.ask(intent.target)
    if intent.op == "order_investigation" and intent.target:
        return engine.order_investigation(intent.target)
    if intent.op == "perform_exam" and intent.target:
        return engine.examine(intent.target)
    if intent.op == "order_medication":
        med = parse_med(intent.target, intent.action_text or transcript)
        text = med.drug + (f" {med.dose}" if med.dose else "")
        return engine.act(LearnerAction(kind="medicate", text=text))
    if intent.op == "diagnose":
        return engine.diagnose(intent.action_text or transcript)
    if intent.op == "perform_action":
        return engine.act(LearnerAction(kind="intervene", text=intent.action_text or transcript))
    return None  # read_vitals / chitchat / malformed target


def _outcome_brief(intent: NurseIntent, result: StepResult | None, vitals: dict) -> str:
    """The factual nugget speak() must report — the engine's real output."""
    if intent.op == "read_vitals":
        keep = {k: v for k, v in vitals.items() if v not in (None, {}, "")}
        return "Current vitals: " + json.dumps(keep)
    if intent.op == "order_medication":
        med = parse_med(intent.target, intent.action_text)
        if needs_dose_confirm(med) and not med.dose:
            return (
                f"The resident is ordering {med.drug} but stated no dose. This is a "
                "high-alert drug — read the order back and ask them to confirm the "
                "dose before it is drawn up. Do not give it yet."
            )
        parts = ["Giving " + med.drug + (f" {med.dose}" if med.dose else "") + "; read the order back to confirm."]
        if result and result.transitioned:
            parts.append(f"The patient's condition changed: {result.message}")
        if result and result.critical_error:
            parts.append("This was a serious misstep with a bad outcome — convey concern.")
        return " ".join(parts)
    if intent.op == "diagnose":
        # The nurse acknowledges the working diagnosis neutrally — she records it,
        # she does not grade it (the debrief scores recognition).
        dx = (intent.action_text or "").strip()
        return (
            f"The resident's working diagnosis is '{dx}'. Acknowledge that you've "
            "noted it, neutrally and without confirming or judging whether it's "
            "right. Do not reveal the correct answer."
        )
    if result is None:
        return "No specific result; respond conversationally and stay in role."
    parts = []
    if result.revealed:
        parts.append(f"Result/finding: {result.revealed}")
    if result.transitioned:
        parts.append(f"The patient's condition changed: {result.message}")
    if result.critical_error:
        parts.append("This was a serious misstep with a bad outcome — convey concern.")
    if not parts:
        parts.append("Order acknowledged; nothing new to report yet.")
    return " ".join(parts)


def speak(
    transcript: str, intent: NurseIntent, result: StepResult | None,
    vitals: dict, history: list[dict],
) -> str:
    """Phrase the nurse's spoken reply from the real engine outcome (one LLM call)."""
    user = (
        f"Recent exchange:\n{_history_lines(history) or '(none)'}\n\n"
        f'Resident said: "{transcript}"\n'
        f"{_outcome_brief(intent, result, vitals)}\n\n"
        "Nurse (spoken reply):"
    )
    try:
        return _complete(SPEAK_SYSTEM, user)
    except Exception as exc:
        logger.warning("nurse speak failed: %s", exc)
        # Fall back to a plain read-back so the turn still produces audio.
        if intent.op == "read_vitals":
            return "Let me read you the vitals on the monitor now."
        return "Got it — I'm on it."


INTRO_SYSTEM = (
    "You are the bedside nurse welcoming the resident to the bedside in an "
    "emergency department. Give a short spoken handoff so they can start: who the "
    "patient is, what brought them in (the chief complaint), and the most relevant "
    "background. Speak calmly and naturally, the way a nurse hands off at the "
    "bedside — not as a written report. No markdown, no lists, no stage directions; "
    "your words are read aloud. Keep it under about 60 words and finish by inviting "
    "the resident to begin."
)


def _intro_facts(case) -> str:
    """The raw handoff material the intro is phrased from."""
    p = case.patient
    pres = case.presentation
    ident = p.name or "the patient"
    if p.age:
        ident += f", a {p.age}-year-old" + (f" {p.gender}" if p.gender else "")
    lines = [f"Patient: {ident}.", f"Chief complaint: {pres.chief_complaint}."]
    if pres.opening_vignette:
        lines.append(f"Scenario: {pres.opening_vignette}")
    if p.past_medical_history:
        lines.append("Past medical history: " + ", ".join(p.past_medical_history) + ".")
    if p.medications:
        lines.append("Medications: " + ", ".join(p.medications) + ".")
    if p.allergies:
        lines.append("Allergies: " + ", ".join(p.allergies) + ".")
    return "\n".join(lines)


def _fallback_intro(case) -> str:
    """A deterministic handoff used when the LLM is unavailable."""
    p = case.patient
    pres = case.presentation
    desc = p.name or "the patient"
    if p.age:
        desc += f", a {p.age}-year-old" + (f" {p.gender}" if p.gender else "")
    out = f"This is {desc}, here with {pres.chief_complaint}."
    if pres.opening_vignette:
        out += f" {pres.opening_vignette}"
    return out + " Let me know how you'd like to start."


def opening_narration(session: VoiceSession) -> str:
    """The nurse's spoken bedside handoff at the start of the session.

    Narrates who the patient is, the chief complaint and the key background so the
    resident can begin. Phrased by the LLM with a deterministic fallback, and
    recorded on the nurse channel so it's part of the conversation context."""
    case = session.engine.case
    status = session.engine.state.patient_status
    user = (
        f"{_intro_facts(case)}\n"
        + (f"Right now: {status}\n" if status else "")
        + "\nNurse (spoken handoff):"
    )
    try:
        reply = _complete(INTRO_SYSTEM, user)
    except Exception as exc:  # never let the handoff break session creation
        logger.warning("nurse intro failed: %s", exc)
        reply = _fallback_intro(case)
    session.history.append({"role": "nurse", "text": reply})
    return reply


MATCH_SYSTEM = (
    "You judge whether a physician's action satisfies a described clinical "
    "trigger in an emergency simulation. Answer with a single word: yes or no."
)


def make_intent_matcher() -> IntentMatcher:
    """An LLM-backed matcher for the engine's soft action transitions.

    The engine calls this only when an action's keywords miss, so a phrasing the
    author didn't enumerate ('she's tiring, let's secure the airway') can still
    trip the intended transition. Fails closed (no match) on any error."""

    def matcher(action_text: str, transition_description: str) -> bool:
        user = (
            f'Physician action: "{action_text}"\n'
            f'Trigger: "{transition_description}"\n'
            "Does the action satisfy the trigger?"
        )
        try:
            answer = _complete(MATCH_SYSTEM, user).strip().lower()
        except Exception as exc:
            logger.warning("intent match failed: %s", exc)
            return False
        return answer.startswith("y")

    return matcher


def _build_card(intent: NurseIntent, engine: SimulationEngine, result: StepResult | None) -> dict | None:
    """Build a diagnostics-drawer card from an order/exam outcome (if any)."""
    if intent.op == "order_medication":
        med = parse_med(intent.target, intent.action_text)
        confirm = needs_dose_confirm(med)
        status = "confirm" if (confirm and not med.dose) else "given"
        return build_order_card(
            med.drug, med.dose, med.detail,
            status=status, confirm_required=confirm, card_id=med.slug,
        )
    if result is None or not result.revealed:
        return None
    if intent.op == "order_investigation":
        inv = next((i for i in engine.case.investigations if i.id == intent.target), None)
        if inv is not None:
            return build_investigation_card(
                inv, engine.current_vitals(), engine.case.patient, result.revealed
            )
    if intent.op == "perform_exam" and intent.target:
        return build_exam_card(intent.target, result.revealed)
    return None


def _voice_reply(
    session: VoiceSession, intent: NurseIntent, result: StepResult | None, vitals: dict, transcript: str,
) -> tuple[str, str]:
    """Phrase the reply and decide who voices it.

    Most outcomes are the nurse reading back an order. Two are voiced by the
    person who actually holds the information: a history beat is spoken by its
    provider (EMS/family/patient), and a bedside measurement (VC/NIF) is reported
    by the Respiratory Therapist. Returns ``(reply, speaker)``."""
    engine = session.engine
    # History from EMS / family / the patient is spoken by that person, grounded.
    if intent.op == "ask_history" and result is not None and result.revealed:
        beat = next((h for h in engine.case.history if h.id == intent.target), None)
        provider = beat.provider if beat else None
        from backend.actor_voice import provider_speaker, voice_history
        if provider_speaker(provider) != "nurse":
            return voice_history(session, provider, result.revealed, transcript)
    # Bedside respiratory measures are reported by the RT, if the case has one.
    if intent.op == "order_investigation" and result is not None and result.revealed:
        inv = next((i for i in engine.case.investigations if i.id == intent.target), None)
        if inv is not None and inv.category == "bedside":
            from backend.actor_voice import voice_bedside
            reply, speaker = voice_bedside(session, result.revealed, transcript)
            if reply is not None:
                return reply, speaker
    return speak(transcript, intent, result, vitals, session.history), "nurse"


def _fire_cues(session: VoiceSession) -> list[dict]:
    """Voice any authored cues that came due — the room proactively prompting a
    lingering team. Each is role-played by its actor and recorded for context."""
    raw = session.engine.due_cues()
    if not raw:
        return []
    from backend.actor_voice import actor_speaker, voice_cue
    case = session.engine.case
    out: list[dict] = []
    for cue in raw:
        actor = next((a for a in case.actors if a.id == cue.get("actor_id")), None)
        speaker = actor_speaker(actor)
        line = voice_cue(session, actor, speaker, cue["text"])
        if not line:
            continue
        session.history.append({"role": "nurse", "text": line})  # context for later turns
        out.append({"speaker": speaker, "text": line})
    return out


def _resolve_turn(
    session: VoiceSession, intent: NurseIntent, transcript: str, clock: list,
) -> NurseTurn:
    """Ground an already-classified intent: dispatch/consult, speak, record, package.

    Shared by :func:`run_turn` (voice → interpreted intent) and :func:`run_action`
    (a panel click → explicit intent), so both produce an identical NurseTurn."""
    engine = session.engine
    vitals = engine.current_vitals().model_dump(exclude_none=True)

    # A consult is role-played (like the patient), not engine-dispatched: the
    # specialist's own words are the reply, and they also land as a Consults card.
    if intent.op == "request_consult":
        from backend.consult_voice import consult_reply
        reply, card = consult_reply(session, intent.target, transcript)
        result = None
        speaker = "consult"
    else:
        result = dispatch(intent, engine, transcript)
        card = _build_card(intent, engine, result)
        reply, speaker = _voice_reply(session, intent, result, vitals, transcript)

    session.history.append({"role": "resident", "text": transcript})
    session.history.append({"role": "nurse", "text": reply})
    if card:
        session.cards.append(card)  # kept so a parked run rebuilds its drawer on resume

    cues = _fire_cues(session)
    working_dx = engine.run.working_diagnoses[-1] if engine.run.working_diagnoses else None

    transitioned = bool(result and result.transitioned) or bool(clock)
    critical = bool(result and result.critical_error) or any(c.critical_error for c in clock)
    return NurseTurn(
        reply=reply,
        intent=intent,
        vitals=vitals,
        state_id=engine.run.current_state_id,
        state_name=engine.state.name,
        revealed=result.revealed if result else (card["summary"] if card else None),
        transitioned=transitioned,
        critical_error=critical,
        ended=engine.run.ended,
        clock_events=[c.message for c in clock if c.message],
        diagnostic=card,
        speaker=speaker,
        working_dx=working_dx,
        cues=cues,
    )


def run_turn(session: VoiceSession, transcript: str) -> NurseTurn:
    """Full grounded turn: advance clock, interpret, dispatch, speak."""
    clock = advance_clock(session)
    intent = interpret(transcript, session.engine, session.history)
    return _resolve_turn(session, intent, transcript, clock)


def _action_label(session: VoiceSession, op: str, target: str | None) -> str:
    """A resident-facing phrasing of a panel click, so the transcript reads naturally."""
    case = session.engine.case
    if op == "perform_exam" and target:
        return f"Examine the {target} system."
    if op == "order_investigation" and target:
        inv = next((i for i in case.investigations if i.id == target), None)
        return f"Order {inv.name if inv else target}."
    if op == "order_medication" and target:
        return f"Give {target}."
    if op == "request_consult" and target:
        from backend.consult_voice import find_consultant
        actor = find_consultant(case, target)
        return f"Call {actor.role if actor else target}."
    return target or op


def run_action(
    session: VoiceSession, op: str, target: str | None = None, action_text: str = "",
) -> NurseTurn:
    """A grounded turn from an explicit UI action (panel click) — no interpret LLM call.

    The op/target come straight from the panel, so we skip interpretation and go
    directly to the deterministic dispatch. Everything else matches a voice turn."""
    if op not in OPS:
        raise ValueError(f"unknown op: {op}")
    clock = advance_clock(session)
    label = action_text.strip() or _action_label(session, op, target)
    intent = NurseIntent(op=op, target=target, action_text=label)
    return _resolve_turn(session, intent, label, clock)
