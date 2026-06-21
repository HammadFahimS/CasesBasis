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


def _complete(system_prompt: str, user_prompt: str) -> str:
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    provider = get_llm_provider()
    client = build_llm_client(provider)
    model = resolve_model(provider, None)
    return client.text_response(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt
    ).strip()


def _cards_by(session: VoiceSession, *categories: str) -> list[str]:
    return [c.get("title", "?") for c in session.cards if c.get("category") in categories]


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
        "exams": _cards_by(session, "exam"),
        "consults": _cards_by(session, "consult"),
        "critical_errors": list(run.critical_errors),
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


def _coaching_note(summary: dict, teaching: dict) -> str:
    did = (
        f"Outcome: {summary['outcome']} after {summary['elapsed_label']}.\n"
        f"Examined: {', '.join(summary['exams']) or 'nothing'}.\n"
        f"Investigations: {', '.join(summary['investigations']) or 'none'}.\n"
        f"Consults: {', '.join(summary['consults']) or 'none'}.\n"
        f"Critical errors: {', '.join(summary['critical_errors']) or 'none'}."
    )
    points = "\n".join(f"- {p}" for p in teaching["teaching_points"]) or "(none provided)"
    user = (
        f"What the resident did:\n{did}\n\n"
        f"Key teaching points for this case:\n{points}\n\n"
        "Coaching note:"
    )
    try:
        return _complete(COACH_SYSTEM, user)
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
    return {
        "case_title": session.case.meta.title,
        "summary": summary,
        "teaching": teaching,
        "coaching_note": _coaching_note(summary, teaching),
    }
