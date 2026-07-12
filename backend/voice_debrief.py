"""End-of-case debrief for a voice-driven simulation run.

Closes the learning loop: when a case ends (a terminal state, or the resident
concludes early), assemble a debrief from three sources —

1. **Run summary** — deterministic facts pulled from the engine: the state path
   taken, time on the clock, what was examined / ordered / consulted, and any
   critical errors. No LLM, fully testable offline.
2. **Teaching** — the case author's ``debrief`` (summary, teaching points, common
   errors, discussion, references) plus the learning ``objectives``.
3. **Coaching note** — one LLM call that turns the run summary + teaching points
   into a short, specific note for *this* resident. Fail-soft: an empty note if
   the model is unavailable, so the deterministic debrief always renders.
"""

from __future__ import annotations

import logging

from backend.voice_session import VoiceSession

logger = logging.getLogger(__name__)

COACH_SYSTEM = (
    "You are an attending physician giving a resident brief, constructive feedback "
    "right after a simulated emergency case. You are given what the resident did "
    "and the case's key teaching points. Write 2-4 sentences of specific, honest "
    "but encouraging coaching: name one or two things they did well and the most "
    "important thing to improve, grounded in what actually happened. Do not invent "
    "actions they didn't take. No markdown, no lists — just plain spoken prose."
)

# Used when the resident took essentially no clinical action. The generic prompt
# above will fabricate praise ("stayed calm", "avoided errors") for a do-nothing
# run — pedagogically wrong, since inaction is a failure in an emergency. This one
# forbids that and pushes an honest, corrective note instead.
COACH_SYSTEM_IDLE = (
    "You are a supportive attending physician debriefing a resident after a "
    "simulated emergency case in which they didn't really get started — no history, "
    "exam, investigations, treatment or diagnosis were recorded. Keep it warm, brief "
    "and encouraging, like a mentor who wants them to dive in next time. Do not "
    "invent praise for things they didn't do (don't credit 'staying calm' or "
    "'avoiding errors'), but don't scold or use harsh words like 'failed' either. In "
    "2-3 friendly sentences: gently note that the case is scored on what they "
    "actively do, point them to a good first move or two for this patient (drawn "
    "from the teaching points), and encourage them to jump in and work the case next "
    "time. No markdown, no lists — plain spoken prose."
)


def _complete(system_prompt: str, user_prompt: str) -> str:
    """The coaching-note LLM seam (monkeypatched in tests)."""
    from backend.agent_runtime import complete

    return complete(system_prompt, user_prompt, kind="coach")


def _cards_by(session: VoiceSession, *categories: str) -> list[str]:
    return [c.get("title", "?") for c in session.cards if c.get("category") in categories]


def _result_reads(session: VoiceSession) -> list[dict]:
    """The clinical read of each investigation the resident ordered — kept hidden
    during the case (the resident interprets live) and surfaced here so the debrief
    still teaches what each result meant."""
    inv_cats = {"lab", "ecg", "imaging", "bedside", "other"}
    ordered = {c.get("id") for c in session.cards if c.get("category") in inv_cats}
    out = []
    for inv in session.case.investigations:
        if inv.id in ordered and getattr(inv, "interpretation", ""):
            out.append({"test": inv.name, "read": inv.interpretation})
    return out


def run_summary(session: VoiceSession) -> dict:
    """Deterministic 'what the resident did', drawn from the engine + cards."""
    engine = session.engine
    run = engine.run
    case = session.case
    state_names = {s.id: s.name for s in case.states}
    terminal = case.state(run.current_state_id)
    elapsed = run.elapsed_seconds
    return {
        "ended": run.ended,
        "outcome": terminal.name if terminal else run.current_state_id,
        "outcome_terminal": bool(getattr(terminal, "is_terminal", False)),
        "elapsed_seconds": elapsed,
        "elapsed_label": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
        "states_path": [state_names.get(sid, sid) for sid in run.visited_states],
        "investigations": _cards_by(session, "lab", "ecg", "imaging", "bedside", "other"),
        "result_reads": _result_reads(session),
        "exams": _cards_by(session, "exam"),
        "consults": _cards_by(session, "consult"),
        "critical_errors": list(run.critical_errors),
        "working_diagnoses": list(run.working_diagnoses),
    }


def _teaching(case) -> dict:
    d = case.debrief
    obj = case.objectives
    return {
        "summary": d.summary,
        "teaching_points": list(d.teaching_points),
        "common_errors": list(d.common_errors),
        "discussion": list(d.discussion),
        "references": list(d.references),
        "objectives": {
            "overall_goal": obj.overall_goal,
            "medical": list(obj.medical),
        },
    }


def _coaching_note(summary: dict, teaching: dict, engaged: bool = True) -> str:
    did = (
        f"Outcome: {summary['outcome']} after {summary['elapsed_label']}.\n"
        f"Examined: {', '.join(summary['exams']) or 'nothing'}.\n"
        f"Investigations: {', '.join(summary['investigations']) or 'none'}.\n"
        + "".join(f"  - {r['test']} meant: {r['read']}\n" for r in summary.get("result_reads") or [])
        + f"Consults: {', '.join(summary['consults']) or 'none'}.\n"
        f"Working diagnosis: {', '.join(summary.get('working_diagnoses') or []) or 'none stated'}.\n"
        f"Critical errors: {', '.join(summary['critical_errors']) or 'none'}."
    )
    points = "\n".join(f"- {p}" for p in teaching["teaching_points"]) or "(none provided)"
    if engaged:
        system = COACH_SYSTEM
        user = (
            f"What the resident did:\n{did}\n\n"
            f"Key teaching points for this case:\n{points}\n\n"
            "Coaching note:"
        )
    else:
        system = COACH_SYSTEM_IDLE
        user = (
            "The resident took essentially no action in this case — the record below "
            "is empty.\n\n"
            f"What the resident did:\n{did}\n\n"
            f"Key teaching points for this case:\n{points}\n\n"
            "Honest coaching note (do not praise inaction):"
        )
    try:
        return _complete(system, user)
    except Exception as exc:  # the deterministic debrief must still render
        logger.warning("debrief coaching note failed: %s", exc)
        return ""


def build_debrief(session: VoiceSession) -> dict:
    """Assemble the full end-of-case debrief payload for the front-end overlay."""
    # Settle the clock first: on a manual "End & Debrief" the engine has only
    # ticked up to the last nurse turn, so elapsed would undercount any patient-
    # only or idle time. advance_clock catches it up to wall-time (and is a no-op
    # on an already-terminal run).
    from backend.voice_session import advance_clock
    advance_clock(session)
    summary = run_summary(session)
    teaching = _teaching(session.case)
    from backend.sim_scoring import score_run
    assessment = [c.to_dict() for c in score_run(session)]
    engaged = _did_engage(session)
    return {
        "case_title": session.case.meta.title,
        "summary": summary,
        "teaching": teaching,
        "assessment": assessment,
        "coaching_note": _coaching_note(summary, teaching, engaged),
    }


def _did_engage(session: VoiceSession) -> bool:
    """Did the resident actually work the case, or effectively do nothing?

    True if they took any recorded action (an exam, order, investigation or
    consult card), committed a working diagnosis, tripped a critical error, or
    moved the case past its opening state. A run with none of these is idle, and
    the coaching note must not congratulate it for 'avoiding errors'."""
    run = session.engine.run
    return bool(
        session.cards
        or run.working_diagnoses
        or run.critical_errors
        or len(run.visited_states) > 1
    )
