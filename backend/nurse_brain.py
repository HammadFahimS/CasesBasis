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

from backend.actors import kind_for_word
from backend.constitution import constitution
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
    "confirm_pending",   # "yes / proceed / go ahead" — execute the parked action
    "cancel_pending",    # "no / hold it / cancel" — drop the parked action
    "chitchat",
}

# Ops that represent a fresh actionable order — one of these supersedes a parked
# pending action rather than being read as a reply to the nurse's confirm question.
_PARKABLE_OPS = {
    "order_medication", "perform_action", "order_investigation",
    "perform_exam", "request_consult", "diagnose",
}

# Ops costly/irreversible enough that a low-confidence ("clarify") match should
# pause and ask the resident to confirm rather than execute the best guess. Cheap,
# reversible ops (perform_exam, ask_history, diagnose, read_vitals) instead run the
# most-likely interpretation the classifier already picked — blocking them on a
# clarify flag only produces a misleading fallback (e.g. a full physical exam
# collapsing to "nothing to note"). dispatch() and _outcome_brief() must agree on
# this set, or a clarify'd op with no reply handler falls through to a wrong line.
_CLARIFY_ASK_OPS = {"order_investigation", "order_medication", "perform_action"}

INTERPRET_SYSTEM = (
    "You are the routing brain of a bedside nurse in an emergency-department "
    "simulation. The physician (resident) speaks to you; classify their single "
    "utterance into ONE action the nurse would take. Reply with STRICT JSON only "
    "— no prose, no markdown — using exactly these keys:\n"
    '  {"op": <one of read_vitals|ask_history|order_investigation|perform_exam|order_medication|perform_action|request_consult|diagnose|chitchat>,\n'
    '   "target": <id/name from the lists below, or null>,\n'
    '   "action_text": <short paraphrase of the order, or "">,\n'
    '   "clarify": <true if the utterance is ambiguous or your target match is uncertain; false otherwise>,\n'
    '   "addressee": <who the resident is speaking to: patient|nurse|ems|family|rt|consult, or "none">}\n'
    "Rules: read_vitals ONLY when the resident explicitly asks for vital signs or "
    "a specific one — e.g. 'what are her vitals/obs?', 'what's her sat / blood "
    "pressure / heart rate / temp?', 'put her on the monitor'. A vague or open "
    "prompt such as 'what do we have here?', 'what's going on?', 'catch me up', "
    "'tell me about the patient', 'give me a rundown' is NOT a vitals request and "
    "is NOT an order — classify it chitchat so the nurse invites the resident to "
    "lead rather than reciting the vitals or the story unprompted. "
    "ask_history to elicit history from the patient/"
    "EMS/family (pick the closest history id). order_investigation for tests/labs/"
    "imaging (pick the closest investigation id). perform_exam to physically "
    "examine the patient — set target to the closest body-system name for a "
    "FOCUSED exam of one region (e.g. 'examine her chest', 'check her reflexes'), "
    "or to \"all\" for a WHOLE-BODY / systematic / head-to-toe physical exam or "
    "whenever no single region is named (e.g. 'do a physical exam', 'examine the "
    "patient', 'give her a once-over'). "
    "request_consult to call, page or refer to a specialist (pick the closest "
    "consultant id as target). order_medication for giving a drug or medication — "
    "set target to the drug name and action_text to the dose/route if stated. "
    "perform_action for any other treatment or intervention that is not a drug "
    "(e.g. intubate, CPR, defibrillate, position the patient) — set action_text. "
    "diagnose when the resident commits a working or differential diagnosis "
    "('I think this is …', 'my working diagnosis is …') — set action_text to the "
    "diagnosis. confirm_pending when the resident is AFFIRMING a pending order the "
    "nurse just read back ('yes', 'proceed', 'go ahead', 'do it', 'confirm', "
    "'that's right') — NOT a new order. cancel_pending when they are CALLING IT OFF "
    "('no', 'hold it', 'cancel that', 'stop', 'don't'). chitchat for anything "
    "social or unclear. "
    "Only use an id/name that appears in the lists. "
    "Set clarify:true when: the stated test or procedure name is unclear or a "
    "plausible mishearing of multiple catalogue entries (e.g. 'ACG' could be ABG "
    "or ECG); your target match feels like a stretch; or the utterance is genuinely "
    "ambiguous between two different ops. When clarify:true, still pick the most "
    "likely op and target so the nurse can name them in the clarification question.\n"
    "ADDRESSEE — reason about who the resident is talking to, choosing only from the "
    "people listed as in the room:\n"
    "- They name or clearly aim at a person ('nurse, …', 'husband, …', 'paramedic, "
    "…'), or it's a natural follow-up to whoever just spoke → that person.\n"
    "- A direct question to the PATIENT about how THEY feel or their own symptoms/story "
    "('how are you feeling?', 'when did this start?', 'any chest pain?') → patient.\n"
    "- An order, a request for data/results, general case-talk, or the resident merely "
    "stating/narrating something or thinking aloud that is NOT aimed at anyone → nurse "
    "(she is the room's point of contact). NEVER send an undirected statement to the "
    "patient just because it mentions a body part or a value.\n"
    "- Use 'none' only if you genuinely cannot tell; when unsure and it is not a bedside "
    "symptom question, prefer nurse over patient. The addressee is independent of op — "
    "an order is still addressed to the nurse; a symptom question to the patient."
)

SPEAK_SYSTEM = constitution("nurse")


@dataclass
class NurseIntent:
    """The classified action behind one resident utterance."""

    op: str = "chitchat"
    target: str | None = None
    action_text: str = ""
    clarify: bool = False  # interpreter flagged uncertain/ambiguous match
    addressee: str | None = None  # who the resident is speaking to (reasoned): a speaker kind or None


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
    needs_addressee: bool = False   # Director uncertain — resident must pick who to address
    summon: str | None = None       # actor id brought into the room this turn (client marks present)


def _complete(system_prompt: str, user_prompt: str) -> str:
    """The single nurse LLM seam (via the shared interactive runtime, which keeps
    the app's provider + Cerebras fallback). Monkeypatched in tests."""
    from backend.agent_runtime import complete

    return complete(system_prompt, user_prompt, kind="nurse")


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


def _room_roster(engine: SimulationEngine) -> str:
    """Who the resident can be addressing right now — present vs not-yet-here — so the
    interpreter can reason about the addressee rather than keyword-match a vocative."""
    from backend.actor_voice import actor_speaker
    present, absent = ["patient", "nurse"], []
    for a in engine.case.actors:
        kind = actor_speaker(a)
        if kind in ("patient", "nurse"):
            continue
        label = f"{kind} ({a.role})" if getattr(a, "role", None) else kind
        (present if engine.is_present(a.id) else absent).append(label)
    lines = ["People in the room now (possible addressees): " + ", ".join(present)]
    if absent:
        lines.append("Not in the room yet (cannot be addressed): " + ", ".join(absent))
    return "\n".join(lines)


_ROLE_LABELS = {
    "resident": "Resident", "nurse": "Nurse", "patient": "Patient",
    "ems": "EMS", "family": "Family", "rt": "RT", "consult": "Consultant",
}


def _history_lines(history: list[dict], limit: int = 6) -> str:
    out = []
    for turn in history[-limit:]:
        role = turn.get("role")
        who = _ROLE_LABELS.get(role, role.title() if role else "Nurse")
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
    addressee = (data.get("addressee") or "").strip().lower() or None
    if addressee in ("none", "null", "unclear", "unknown", ""):
        addressee = None
    return NurseIntent(
        op=op,
        target=str(target) if target else None,
        action_text=(data.get("action_text") or "").strip(),
        clarify=bool(data.get("clarify", False)),
        addressee=addressee,
    )


def _addressed_history_speaker(transcript: str) -> str | None:
    """If the utterance opens by naming a bystander who *holds history* ('EMS, …'
    / 'Husband, …'), the speaker class they addressed — else None. Staff (nurse,
    RT) and consultants don't hold history beats, so addressing them routes
    normally. Used to keep a beat with the person actually spoken to, rather than
    whoever's beat reads most relevant."""
    head = (transcript or "").strip().lower()
    m = re.match(r"[a-z']+", head)
    kind = kind_for_word(m.group(0)) if m else None
    return kind.id if (kind and kind.holds_history) else None


def _beat_for_speaker(engine: SimulationEngine, speaker_class: str) -> str | None:
    """A history-beat id held by ``speaker_class`` — preferring one not yet asked."""
    from backend.actor_voice import provider_speaker
    beats = [h for h in engine.case.history if provider_speaker(h.provider) == speaker_class]
    if not beats:
        return None
    unrevealed = [h for h in beats if h.id not in engine.run.revealed_history]
    return (unrevealed or beats)[0].id


def _apply_addressee(intent: NurseIntent, transcript: str, engine: SimulationEngine) -> NurseIntent:
    """Honour an explicit vocative: if the resident addressed EMS/family/the patient,
    keep the history beat with that person. Overrides only when the interpreter's
    pick lands on the wrong person (or it under-classified as chitchat)."""
    addressed = _addressed_history_speaker(transcript)
    if not addressed or intent.op not in ("ask_history", "chitchat"):
        return intent
    from backend.actor_voice import provider_speaker
    current = next((h for h in engine.case.history if h.id == intent.target), None)
    if current is not None and provider_speaker(current.provider) == addressed:
        return intent  # the interpreter already picked this person's beat — keep its specificity
    beat_id = _beat_for_speaker(engine, addressed)
    if not beat_id:
        return intent
    return NurseIntent(op="ask_history", target=beat_id, action_text=intent.action_text,
                       addressee=intent.addressee or addressed)


def interpret(transcript: str, engine: SimulationEngine, history: list[dict]) -> NurseIntent:
    """Map one utterance to a structured engine intent (one LLM call)."""
    user = (
        f"{_catalogue(engine)}\n\n"
        f"{_room_roster(engine)}\n\n"
        f"Recent exchange:\n{_history_lines(history) or '(none)'}\n\n"
        f'Resident just said: "{transcript}"\n\nJSON:'
    )
    try:
        intent = _parse_intent(_complete(INTERPRET_SYSTEM, user))
    except Exception as exc:  # never let interpretation break the turn
        logger.warning("nurse interpret failed: %s", exc)
        intent = NurseIntent(op="chitchat")
    return _apply_addressee(intent, transcript, engine)


# Bare affirmations / negations that resolve a parked pending action even when the
# interpreter under-classified them as chitchat (the anti-loop safety net for S3).
_AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|confirm(?:ed)?|proceed|go ahead|do it|"
    r"go for it|that's right|affirmative|please do|sounds good|ok(?:ay)?)\b",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"\b(no|nope|hold (?:it|off|on)|cancel|stop|don't|do not|wait|never mind|abort)\b",
    re.IGNORECASE,
)
# perform_action utterances that are a disposition/admission decision — confirmable
# (read back before committing), unlike an immediate intervention (intubate, CPR).
_DISPOSITION_RE = re.compile(
    r"\b(admit|admission|icu|intensive care|transfer (?:to|her|him|them)|"
    r"disposition|plasma exchange|send (?:her|him|them|up)|up to the unit)\b",
    re.IGNORECASE,
)


def _is_affirmation(text: str) -> bool:
    return bool(_AFFIRM_RE.search(text or ""))


def _is_cancellation(text: str) -> bool:
    return bool(_CANCEL_RE.search(text or ""))


def _is_disposition(text: str) -> bool:
    return bool(_DISPOSITION_RE.search(text or ""))


def _execute_pending(engine: SimulationEngine, pending: dict, transcript: str,
                     intent: NurseIntent) -> StepResult | None:
    """Run a parked action now that it's confirmed (clearing it first)."""
    from backend.medication import _extract_dose
    engine.run.pending_action = None
    if pending["op"] == "order_medication":
        dose = pending.get("dose") or _extract_dose(transcript, intent.action_text)
        text = pending["target"] + (f" {dose}" if dose else "")
        return engine.act(LearnerAction(kind="medicate", text=text))
    if pending["op"] == "perform_action":
        return engine.act(LearnerAction(kind="intervene", text=pending.get("action_text") or ""))
    return None


def dispatch(intent: NurseIntent, engine: SimulationEngine, transcript: str) -> StepResult | None:
    """Run the intent on the engine. Pure/deterministic — no LLM.

    Returns the engine's :class:`StepResult` (None for read_vitals/chitchat, and
    for a freshly *parked* confirmable action, which mutate no clinical state yet).

    Turn-spanning protocol (S3): a high-alert medication ordered without a dose, or
    a disposition/admission, is **not executed** — it's parked on
    ``engine.run.pending_action`` and the nurse reads it back. A following
    affirmation ("proceed"), or the missing dose, executes it; a negation cancels
    it; a brand-new order supersedes it."""
    if intent.clarify and intent.op in _CLARIFY_ASK_OPS:
        return None  # hold — let _outcome_brief ask "did you mean X?" (costly ops only)

    # 1 — resolve a parked action against this utterance, before normal routing.
    pending = engine.run.pending_action
    if pending:
        if intent.op == "cancel_pending" or _is_cancellation(transcript):
            engine.run.pending_action = None
            return None
        supplies_dose = (
            pending["op"] == "order_medication"
            and pending.get("status") == "awaiting_info"
            and _med_dose(transcript, intent.action_text) is not None
        )
        if intent.op == "confirm_pending" or _is_affirmation(transcript) or supplies_dose:
            return _execute_pending(engine, pending, transcript, intent)
        if intent.op in _PARKABLE_OPS:
            engine.run.pending_action = None  # a new order supersedes — fall through
        else:
            return None  # unrelated chatter while a confirm is pending — keep it

    # 2 — normal routing (confirmables park instead of executing).
    if intent.op == "ask_history" and intent.target:
        return engine.ask(intent.target)
    if intent.op == "order_investigation" and intent.target:
        return engine.order_investigation(intent.target)
    if intent.op == "perform_exam":
        # The classifier decides scope (no keyword matching — it understands
        # paraphrase): a body-system target is a focused exam; the sentinel "all"
        # (or an empty target) is a whole-body physical exam. A stray/unknown
        # system still degrades gracefully in engine.examine().
        target = (intent.target or "").strip()
        if target and target.lower() != "all":
            return engine.examine(intent.target)
        intent.target = None   # a full exam has no single system → titles as "Physical exam"
        return engine.examine_all()
    if intent.op == "order_medication":
        med = parse_med(intent.target, intent.action_text or transcript)
        if needs_dose_confirm(med) and not med.dose:
            # High-alert with no dose stated → park and read back for the dose.
            engine.run.pending_action = {
                "op": "order_medication", "target": med.drug, "dose": None,
                "action_text": med.detail, "label": med.drug, "status": "awaiting_info",
            }
            return None
        text = med.drug + (f" {med.dose}" if med.dose else "")
        return engine.act(LearnerAction(kind="medicate", text=text))
    if intent.op == "diagnose":
        return engine.diagnose(intent.action_text or transcript)
    if intent.op == "perform_action":
        action_text = intent.action_text or transcript
        if _is_disposition(action_text):
            # A disposition decision → read it back and confirm before committing.
            engine.run.pending_action = {
                "op": "perform_action", "target": None, "dose": None,
                "action_text": action_text, "label": "the admission/disposition",
                "status": "awaiting_confirm",
            }
            return None
        return engine.act(LearnerAction(kind="intervene", text=action_text))
    return None  # read_vitals / chitchat / confirm/cancel with nothing pending


def _med_dose(transcript: str, action_text: str) -> str | None:
    from backend.medication import _extract_dose
    return _extract_dose(transcript, action_text)


# Which vital(s) a read_vitals utterance actually named. Deliberately mirrors the
# cockpit's client-side reveal vocabulary (voice_room_functional.html VITAL_WORDS)
# so the nurse's *spoken* report and the on-screen panel reveal the same vitals —
# asking for "BP and HR" should not make her recite the whole monitor.
_VITAL_ASK_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\b(spo2|sp02|o2 ?sats?|oxygen|saturations?|sats?)\b", re.I), ["oxygen_saturation"]),
    (re.compile(r"\b(blood pressure|bp)\b", re.I), ["blood_pressure"]),
    (re.compile(r"\b(heart rate|hr|pulse)\b", re.I), ["heart_rate", "rhythm"]),
    (re.compile(r"\b(respiratory rate|resp rate|breathing rate|rr)\b", re.I), ["respiratory_rate"]),
    (re.compile(r"\b(temperature|temp|fever)\b", re.I), ["temperature_c"]),
    (re.compile(r"\b(glucose|blood sugar|sugar|bsl|cbg)\b", re.I), ["glucose"]),
    (re.compile(r"\bpain\b", re.I), ["pain"]),
]


def _requested_vital_keys(transcript: str, available: dict) -> list[str] | None:
    """The specific vital keys the resident named, filtered to those we actually
    have. Returns ``None`` for a generic ask ("what are her vitals/obs?", "put her
    on the monitor") — the caller then reports the full set. Rhythm rides along
    with heart rate, the way a monitor read-back naturally states it."""
    named: list[str] = []
    for pat, keys in _VITAL_ASK_PATTERNS:
        if pat.search(transcript or ""):
            for k in keys:
                if k not in named:
                    named.append(k)
    if not named:
        return None
    return [k for k in named if k in available]


def _outcome_brief(intent: NurseIntent, result: StepResult | None, vitals: dict,
                   already_asked: bool = False, engine: SimulationEngine | None = None,
                   transcript: str = "") -> str:
    """The factual nugget speak() must report — the engine's real output."""
    # Clarification guard: interpreter flagged ambiguous/uncertain match
    if intent.clarify and intent.op in _CLARIFY_ASK_OPS:
        label = intent.target or intent.action_text or "that"
        return (
            f"You were uncertain what the resident meant. Ask them to clarify: "
            f"'Did you mean {label}? Please say it again so I can confirm.'"
        )

    # ── turn-spanning confirmation protocol (S3) ──────────────────────────────
    pending = engine.run.pending_action if engine is not None else None
    if intent.op == "cancel_pending":
        return ("The resident called off the pending order. Acknowledge you're holding "
                "it and await further instruction.")
    if intent.op == "confirm_pending":
        if result is None:
            return "There was nothing pending to confirm — respond briefly and stay in role."
        parts = ["The resident confirmed the pending order; it's being carried out now."]
        if result.transitioned:
            parts.append(f"The patient's condition changed: {result.message}")
        if result.critical_error:
            parts.append("This was a serious misstep with a bad outcome — convey concern.")
        return " ".join(parts)
    if pending is not None and result is None and intent.op == "perform_action" \
            and pending.get("status") == "awaiting_confirm":
        return (
            f"The resident is directing a disposition: '{pending['action_text']}'. Read "
            "it back and ask them to confirm before you set it in motion. Do not act yet."
        )

    if intent.op == "read_vitals":
        keep = {k: v for k, v in vitals.items() if v not in (None, {}, "")}
        requested = _requested_vital_keys(transcript, keep)
        if requested is not None:                    # named specific vitals → report only those
            keep = {k: keep[k] for k in requested}
            if not keep:
                return ("The resident asked for a vital that isn't being tracked for this "
                        "patient. Tell them briefly it's not available on the monitor.")
            return ("The resident asked for specific vitals — report ONLY these and nothing "
                    "else: " + json.dumps(keep))
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
    if intent.op == "order_investigation":
        if result and result.revealed:
            return (
                "Report this test result to the resident as RAW findings only, "
                "exactly the signal given — do NOT add any interpretation, "
                "diagnosis, or what it means; the resident interprets it: "
                f"{result.revealed}"
            )
        if result and result.message:
            return f"The test isn't back yet — relay this plainly: {result.message}"
        return "Order acknowledged; nothing back yet — say so briefly."
    if intent.op == "perform_exam":
        # The exam findings are charted as a card for the resident to read — never
        # read them aloud (like imaging). The nurse only says the exam is done and
        # the findings are in the notes to review.
        if result and result.revealed:
            system = (intent.target or "").strip()
            return (
                f"You performed the {system + ' ' if system else ''}exam. Say ONLY that "
                "it's done and the findings are charted in the notes for the resident to "
                "review — ONE short line. Do NOT read out, describe, summarise, or "
                "interpret any of the findings."
            )
        if result and result.message:
            return f"The exam couldn't be completed — relay this plainly: {result.message}"
        return "Exam acknowledged; nothing to note — say so briefly."
    if result is None:
        return (
            "The resident hasn't given a specific order or question yet. Respond in "
            "one short line and stay in role — invite them to lead (e.g. ask what "
            "they'd like to do first). Do NOT volunteer the vitals, the history, or "
            "the patient's story unprompted; make them ask."
        )
    parts = []
    if result.revealed:
        if already_asked:
            parts.append(
                f"The resident already heard this — acknowledge briefly that you "
                f"already shared it, then give a one-sentence reminder: {result.revealed}"
            )
        else:
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
    vitals: dict, history: list[dict], already_asked: bool = False,
    engine: SimulationEngine | None = None,
) -> str:
    """Phrase the nurse's spoken reply from the real engine outcome (one LLM call)."""
    user = (
        f"Recent exchange:\n{_history_lines(history) or '(none)'}\n\n"
        f'Resident said: "{transcript}"\n'
        f"{_outcome_brief(intent, result, vitals, already_asked=already_asked, engine=engine, transcript=transcript)}\n\n"
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
    "You are the bedside nurse greeting a resident who has just walked in, at the "
    "moment a new patient arrives. Give a short, warm, spoken GREETING ONLY — one "
    "sentence, maybe two.\n"
    "Do this and nothing more: greet the resident, tell them the patient (by first "
    "name) has just arrived for them to assess, and invite them to begin.\n"
    "FORBIDDEN — reveal NOTHING about the case. No chief complaint, no symptoms, no "
    "reason for the visit, no history, no medications, no allergies, no vital signs "
    "or numbers, no examination findings, no clinical detail of any kind. The "
    "resident must discover all of it by asking you and examining the patient.\n"
    "No markdown, no stage directions — your words are spoken aloud. Under 25 words.\n"
    "Always speak in English, even if the patient's name looks non-English — never "
    "switch languages.\n"
    "Example shape: \"Hi doctor — Emma's just arrived and she's all yours. "
    "Where would you like to start?\""
)


def _fallback_intro(case) -> str:
    """A deterministic minimal greeting used when the LLM is unavailable.

    Reveals nothing about the case beyond the patient's name — the resident
    discovers the complaint, history and vitals by asking."""
    name = case.patient.name or "your patient"
    return f"Hi doctor — {name} has just arrived for you to assess. Where would you like to start?"


def opening_narration(session: VoiceSession) -> str:
    """The nurse's spoken bedside handoff at the start of the session.

    Narrates who the patient is, the chief complaint and the key background so the
    resident can begin. Phrased by the LLM with a deterministic fallback, and
    recorded on the nurse channel so it's part of the conversation context."""
    case = session.engine.case
    # Feed ONLY the patient's first name — the greeting must reveal nothing else,
    # so we never even hand the model the complaint, history or vitals to leak.
    name = case.patient.name or "the patient"
    user = f"Patient's first name: {name}.\n\nNurse (spoken greeting):"
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
                inv, engine.current_vitals(), engine.case.patient, result.revealed,
                state_id=engine.run.current_state_id,
            )
    if intent.op == "perform_exam":
        # Focused exam → the named system; full exam (no target) → "Physical".
        return build_exam_card(intent.target or "Physical", result.revealed)
    return None


def _voice_reply(
    session: VoiceSession, intent: NurseIntent, result: StepResult | None, vitals: dict,
    transcript: str, already_asked: bool = False,
) -> tuple[str, str]:
    """Delegate to the Director, which owns speaker-voice routing."""
    from backend.director import voice_reply
    return voice_reply(session, intent, result, vitals, transcript, already_asked=already_asked)


def _fire_cues(session: VoiceSession) -> list[dict]:
    """Delegate to the Director, which owns proactive room events."""
    from backend.director import fire_cues
    return fire_cues(session)


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
        # Capture before dispatch mutates revealed_history so we can tell the
        # speaker "you already mentioned this" on a repeated history ask.
        was_already_asked = (
            intent.op == "ask_history"
            and bool(intent.target)
            and intent.target in engine.run.revealed_history
        )
        result = dispatch(intent, engine, transcript)
        card = _build_card(intent, engine, result)
        reply, speaker = _voice_reply(
            session, intent, result, vitals, transcript, already_asked=was_already_asked,
        )

    session.history.append({"role": "resident", "text": transcript})
    # Persist under the speaker who actually voiced the reply (S2), not "nurse".
    session.history.append({"role": speaker, "text": reply})
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


def run_turn(
    session: VoiceSession, transcript: str, forced_addressee: str | None = None
) -> NurseTurn:
    """Full grounded turn — the single unified pipeline the Director routes.

    advance clock → interpret → **Director decides the addressee** (staff, a named
    bystander, or the patient) → dispatch/voice. The patient is just another routed
    actor here: a grounded patient history beat is voiced through the engine, and a
    conversational utterance falls to the patient's free roleplay — no separate
    client endpoint, no ``else → patient`` default (S1).

    ``forced_addressee`` is set when the resident taps an addressee chip after a
    Director ask-back; it routes directly to the named actor (or the patient)."""
    with session.lock:
        engine = session.engine
        clock = advance_clock(session)
        intent = interpret(transcript, engine, session.history)

        from backend.routing import route
        decision = route(intent, engine, transcript, forced_addressee=forced_addressee)
        logger.info("route: addressee=%s present=%s summon=%s reason=%s",
                    decision.addressee, decision.present, decision.summon, decision.reason)

        if decision.ask_back:
            from backend.director import ask_back_prompt, head_clarify_prompt
            vitals = engine.current_vitals().model_dump(exclude_none=True)
            session.history.append({"role": "resident", "text": transcript})
            # The Head Physician voices the clarify: a generic "who's that for?" when the
            # addressee is unclear, or the provider-naming line for an ambiguous history ask.
            line = (head_clarify_prompt(engine)
                    if decision.reason == "unclear addressee"
                    else ask_back_prompt(engine))
            return NurseTurn(
                reply=line, intent=intent, vitals=vitals,
                state_id=engine.run.current_state_id, state_name=engine.state.name,
                clock_events=[c.message for c in clock if c.message],
                speaker="head", needs_addressee=True,
            )

        # A pure "call the husband" — the nurse fetches them; she acknowledges and
        # the actor enters the room (addressable next turn).
        if decision.summon and decision.addressee == "nurse":
            return _summon_turn(session, intent, transcript, clock, decision)
        # Calling a consultant summons them inline and they still answer this turn.
        if decision.summon:
            engine.summon(decision.summon)

        # The addressee is a real person not in the room → answer honestly (the
        # nurse states the room fact), never fabricate a reply for them (C1/C3).
        if not decision.present:
            return _absent_actor_turn(session, intent, transcript, clock, decision)

        if decision.addressee == "patient":
            return _patient_routed_turn(session, intent, transcript, clock)

        # The resident is now engaging this specific person — record it so authored
        # cues that only nudge "go and talk to them" retire once served (Issue-2).
        if decision.actor_id and decision.addressee not in ("nurse", "patient"):
            engine.mark_engaged(decision.actor_id)

        # A present bystander (EMS / family / RT) answers from their OWN knowledge —
        # positive routing (C3): a history/conversational turn reaches them, never the
        # nurse covering. If they hold a relevant beat we ground to it; if they hold
        # none (e.g. the RT has no history), they decline in character rather than the
        # nurse fabricating a cover. Orders to them fall through to normal dispatch.
        if decision.addressee in ("ems", "family", "rt") and intent.op in ("ask_history", "chitchat"):
            beat_id = _beat_for_speaker(engine, decision.addressee)
            if beat_id:
                intent = NurseIntent(
                    op="ask_history", target=beat_id, action_text=intent.action_text
                )
            else:
                return _refusal_turn(session, intent, transcript, clock, decision)

        return _resolve_turn(session, intent, transcript, clock)


def _patient_routed_turn(
    session: VoiceSession, intent: NurseIntent, transcript: str, clock: list,
) -> NurseTurn:
    """The patient speaks this turn — grounded on an authored beat where the words
    map to one, else free roleplay.

    Symptom-directed conversation ("how are you feeling?", "when did this start?")
    is grounded against an unrevealed patient history beat so the reveal is tracked
    in ``run.revealed_history``; anything else is the patient's free roleplay,
    voiced through :func:`patient_voice.patient_reply`."""
    from backend.actor_voice import provider_speaker
    engine = session.engine
    patient_beats = [
        h for h in engine.case.history if provider_speaker(h.provider) == "patient"
    ]
    # Symptom-directed → ground to the next unrevealed patient beat (engine reveal).
    if patient_beats and (set(transcript.lower().split()) & _SYMPTOM_WORDS):
        unrevealed = [h for h in patient_beats
                      if h.id not in engine.run.revealed_history]
        if unrevealed:
            grounded = NurseIntent(op="ask_history", target=unrevealed[0].id)
            return _resolve_turn(session, grounded, transcript, clock)

    # Free roleplay — patient_reply records the exchange on the patient channel.
    from backend.patient_voice import patient_reply
    reply = patient_reply(session, transcript)
    vitals = engine.current_vitals().model_dump(exclude_none=True)
    cues = _fire_cues(session)
    transitioned = bool(clock)
    critical = any(c.critical_error for c in clock)
    return NurseTurn(
        reply=reply,
        intent=intent,
        vitals=vitals,
        state_id=engine.run.current_state_id,
        state_name=engine.state.name,
        transitioned=transitioned,
        critical_error=critical,
        ended=engine.run.ended,
        clock_events=[c.message for c in clock if c.message],
        speaker="patient",
        working_dx=engine.run.working_diagnoses[-1] if engine.run.working_diagnoses else None,
        cues=cues,
    )


def _absent_actor_turn(
    session: VoiceSession, intent: NurseIntent, transcript: str, clock: list, decision,
) -> NurseTurn:
    """The resident addressed someone who is not in the room. The nurse states the
    room fact honestly and offers to call them — no fabricated reply, no silent
    drop (C1). This is a *room-state* line, not invented clinical content."""
    engine = session.engine
    actor = next((a for a in engine.case.actors if a.id == decision.actor_id), None)
    role = (actor.role if actor else decision.addressee).strip()
    wait = engine.arrival_wait(decision.actor_id)
    if wait and wait > 0:
        mins = max(1, round(wait / 60))
        reply = (f"{role} isn't here yet — about {mins} minute{'s' if mins != 1 else ''} out. "
                 f"Want me to call ahead?")
    else:
        reply = f"{role} isn't at the bedside right now. Should I call them in?"

    session.history.append({"role": "resident", "text": transcript})
    session.history.append({"role": "nurse", "text": reply})
    vitals = engine.current_vitals().model_dump(exclude_none=True)
    cues = _fire_cues(session)
    return NurseTurn(
        reply=reply, intent=intent, vitals=vitals,
        state_id=engine.run.current_state_id, state_name=engine.state.name,
        transitioned=bool(clock), critical_error=any(c.critical_error for c in clock),
        ended=engine.run.ended, clock_events=[c.message for c in clock if c.message],
        speaker="nurse",
        working_dx=engine.run.working_diagnoses[-1] if engine.run.working_diagnoses else None,
        cues=cues,
    )


def _summon_turn(
    session: VoiceSession, intent: NurseIntent, transcript: str, clock: list, decision,
) -> NurseTurn:
    """The resident asked to bring someone in ("call the husband"). The nurse fetches
    them (engine.summon) and acknowledges; the actor is in the room from now on, so a
    direct address next turn reaches them (C1)."""
    engine = session.engine
    engine.summon(decision.summon)
    actor = next((a for a in engine.case.actors if a.id == decision.summon), None)
    role = (actor.role if actor else decision.summon).strip()
    reply = f"Bringing {role} in now — they're right here at the bedside for you."

    session.history.append({"role": "resident", "text": transcript})
    session.history.append({"role": "nurse", "text": reply})
    vitals = engine.current_vitals().model_dump(exclude_none=True)
    cues = _fire_cues(session)
    return NurseTurn(
        reply=reply, intent=intent, vitals=vitals,
        state_id=engine.run.current_state_id, state_name=engine.state.name,
        transitioned=bool(clock), critical_error=any(c.critical_error for c in clock),
        ended=engine.run.ended, clock_events=[c.message for c in clock if c.message],
        speaker="nurse",
        working_dx=engine.run.working_diagnoses[-1] if engine.run.working_diagnoses else None,
        cues=cues,
        summon=decision.summon,   # tell the client to mark them present now
    )


def _refusal_turn(
    session: VoiceSession, intent: NurseIntent, transcript: str, clock: list, decision,
) -> NurseTurn:
    """A present actor was asked something outside what they hold → they decline in
    character (C3), instead of the nurse fabricating a cover. No engine change."""
    from backend.agents import refuse
    engine = session.engine
    speaker = decision.addressee
    actor = next((a for a in engine.case.actors if a.id == decision.actor_id), None)
    reply = refuse(speaker, actor)

    session.history.append({"role": "resident", "text": transcript})
    session.history.append({"role": speaker, "text": reply})
    vitals = engine.current_vitals().model_dump(exclude_none=True)
    cues = _fire_cues(session)
    return NurseTurn(
        reply=reply, intent=intent, vitals=vitals,
        state_id=engine.run.current_state_id, state_name=engine.state.name,
        transitioned=bool(clock), critical_error=any(c.critical_error for c in clock),
        ended=engine.run.ended, clock_events=[c.message for c in clock if c.message],
        speaker=speaker,
        working_dx=engine.run.working_diagnoses[-1] if engine.run.working_diagnoses else None,
        cues=cues,
    )


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


_SYMPTOM_WORDS = frozenset({
    "feel", "feeling", "felt", "symptom", "symptoms", "tell", "happened",
    "wrong", "bothering", "complain", "complaint", "breath", "breathe",
    "breathing", "weak", "weakness", "trouble", "problem", "since", "long",
    "started", "begin", "began", "worse", "worsen", "history", "going on",
    "what brought", "why are", "why did",
})


def run_patient_turn(session: VoiceSession, transcript: str) -> NurseTurn | None:
    """Grounded patient-addressed turn for ask_history beats held by the patient.

    When the resident speaks directly to the patient and the utterance maps to a
    patient-held history beat, runs it through the engine (grounding the reveal in
    ``run.revealed_history``) and voices it through the patient character. Returns
    ``None`` to signal "this is conversational — fall through to patient_reply".

    Falls back to the first unrevealed patient beat when the interpreter classifies
    the utterance as chitchat but it uses symptom-directed language — catches "how
    are you feeling?" and similar direct-symptom questions the LLM interpreter
    may under-classify."""
    with session.lock:
        clock = advance_clock(session)
        intent = interpret(transcript, session.engine, session.history)
        from backend.actor_voice import provider_speaker
        patient_beats = [
            h for h in session.engine.case.history
            if provider_speaker(h.provider) == "patient"
        ]
        # Direct match: interpreter returned ask_history for a patient-held beat.
        if intent.op == "ask_history" and intent.target:
            beat = next((h for h in patient_beats if h.id == intent.target), None)
            if beat is not None:
                return _resolve_turn(session, intent, transcript, clock)
        # Fallback: interpreter said chitchat but transcript sounds like a symptom
        # question AND there's at least one unrevealed patient beat.
        if intent.op == "chitchat" and patient_beats:
            words = set(transcript.lower().split())
            if words & _SYMPTOM_WORDS:
                unrevealed = [h for h in patient_beats
                              if h.id not in session.engine.run.revealed_history]
                if unrevealed:
                    intent = NurseIntent(op="ask_history", target=unrevealed[0].id)
                    return _resolve_turn(session, intent, transcript, clock)
        return None


def run_action(
    session: VoiceSession, op: str, target: str | None = None, action_text: str = "",
) -> NurseTurn:
    """A grounded turn from an explicit UI action (panel click) — no interpret LLM call.

    The op/target come straight from the panel, so we skip interpretation and go
    directly to the deterministic dispatch. Everything else matches a voice turn."""
    if op not in OPS:
        raise ValueError(f"unknown op: {op}")
    with session.lock:  # serialise against a background runtime tick (see run_turn)
        clock = advance_clock(session)
        label = action_text.strip() or _action_label(session, op, target)
        intent = NurseIntent(op=op, target=target, action_text=label)
        return _resolve_turn(session, intent, label, clock)
