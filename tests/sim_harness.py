"""Headless live-sim conversation harness (Phase 1 of docs/multi-agent-plan.md §4).

Run a whole multi-turn resident↔room conversation through the *real* server
pipeline — Director routing, engine dispatch, voice routing, persistence — with
**no browser, no audio, no live LLM calls**, and assert on structured state
(speaker, op, transitions, score) rather than generated prose.

How it stays deterministic
--------------------------
Only the two genuinely non-deterministic seams are stubbed; everything else is
the production code path:

* ``interpret`` LLM call → returns the **scripted intent** the conversation
  declares for that turn (the rest of ``interpret`` — incl. ``_apply_addressee``
  — still runs for real).
* every phrasing seam (``speak`` / bystander / consult / patient voice) → a
  deterministic marker. We never assert on wording; the *speaker* comes from the
  ``NurseTurn`` the pipeline returns, not from the marker.

Client-routing fidelity
-----------------------
Today the browser decides *which endpoint* an utterance hits (the S1 defect).
:func:`client_route` is a faithful Python port of ``detectAddressee`` in
``voice_room.html`` so the harness reproduces that decision exactly — letting the
characterization tests prove the mis-route, and the later phases flip it green
once routing moves server-side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend import actor_voice, consult_voice, nurse_brain, patient_voice
from backend.actor_voice import actor_speaker
from backend.nurse_brain import NurseTurn, run_action, run_turn
from backend.voice_session import VoiceSession, start_session

SCRIPTS_DIR = Path(__file__).parent / "sim_scripts"


# ── LLM stubbing ──────────────────────────────────────────────────────────────

class StubLLM:
    """Replaces every ``_complete`` seam so a conversation runs fully offline.

    ``intent_json`` is set by the runner before each spoken turn; the interpret
    seam returns it verbatim so the real ``interpret`` (parse + addressee logic)
    runs on top. Phrasing seams return a fixed marker — prose is never asserted.
    """

    def __init__(self) -> None:
        self.intent_json = '{"op": "chitchat"}'

    def nurse_complete(self, system: str, user: str) -> str:
        if "routing brain" in system:
            return self.intent_json
        if "judge whether" in system:      # the soft action matcher — fail closed
            return "no"
        return "[spoken reply]"             # speak() / opening narration

    def voice_complete(self, system: str, user: str) -> str:
        return "[spoken reply]"             # bystander / patient / cue / consult


def install_stubs(monkeypatch, stub: StubLLM) -> None:
    """Point every LLM seam at the stub (nurse, patient/bystander, consult)."""
    monkeypatch.setattr(nurse_brain, "_complete", stub.nurse_complete)
    monkeypatch.setattr(patient_voice, "_complete", stub.voice_complete)
    monkeypatch.setattr(actor_voice, "_complete", stub.voice_complete)
    monkeypatch.setattr(consult_voice, "_complete", stub.voice_complete)


# ── client-side routing port (the S1 defect, reproduced faithfully) ───────────

_SPEAKER_SYNS = {
    "nurse": ["nurse"],
    "ems": ["ems", "paramedic", "medic", "prehospital"],
    "family": ["husband", "wife", "family", "partner", "mother", "father",
               "dad", "mom", "son", "daughter", "relative"],
    "rt": ["rt", "respiratory therapist", "respiratory", "therapist"],
    "patient": ["patient"],
}


def build_roster(session: VoiceSession) -> list[dict]:
    """The voice roster the browser would receive: one row per case actor."""
    return [
        {"id": a.id, "role": a.role or "", "speaker": actor_speaker(a)}
        for a in session.case.actors
    ]


def build_address_index(roster: list[dict]) -> list[dict]:
    """Port of ``buildAddressIndex`` — vocative words → (speaker, toStaff)."""
    out: list[dict] = []
    for r in roster:
        words = set(_SPEAKER_SYNS.get(r["speaker"], []))
        words.update(w for w in re.split(r"[^a-z]+", r["role"].lower()) if len(w) > 2)
        words.update(w for w in re.split(r"[^a-z]+", r["id"].lower()) if len(w) > 1)
        for w in words:
            out.append({"word": w, "speaker": r["speaker"], "to_staff": r["speaker"] != "patient"})
    if not any(a["speaker"] == "nurse" for a in out):
        out.append({"word": "nurse", "speaker": "nurse", "to_staff": True})
    out.sort(key=lambda a: -len(a["word"]))   # longest match wins
    return out


def client_route(said: str, address: list[dict]) -> dict:
    """Port of ``detectAddressee`` — the legacy ``else → patient`` default and all."""
    low = re.sub(r"^\s+", "", said or "").lower()
    for a in address:
        if re.match(r"^" + re.escape(a["word"]) + r"\b[\s,:.\-—]*", low, re.I):
            stripped = re.sub(
                r"^\s*" + re.escape(a["word"]) + r"\b[\s,:.\-—]*", "", said, flags=re.I
            ).strip()
            return {"to_staff": a["to_staff"], "speaker": a["speaker"], "text": stripped or said}
    return {"to_staff": False, "speaker": "patient", "text": said}   # the S1 default


# ── conversation run model ────────────────────────────────────────────────────

@dataclass
class TurnRun:
    """One played turn: what was said, where it routed, and the engine outcome."""

    say: str
    endpoint: str               # staff | patient | patient_conversational | clock | goto
    route_speaker: str          # who the *client* decided to address
    turn: NurseTurn | None      # the grounded outcome (None for conversational/setup)
    persisted_role: str | None  # the role written to session.history (S2 probe)


@dataclass
class ConversationRun:
    session: VoiceSession
    turns: list[TurnRun] = field(default_factory=list)


def play(session: VoiceSession, steps: list[dict], stub: StubLLM,
         address: list[dict]) -> list[TurnRun]:
    """Execute a list of script steps against a live session, returning each outcome."""
    out: list[TurnRun] = []
    for step in steps:
        out.append(_play_step(session, step, stub, address))
    return out


def _play_step(session: VoiceSession, step: dict, stub: StubLLM,
               address: list[dict]) -> TurnRun:
    # ── non-spoken setup steps ────────────────────────────────────────────────
    if "goto" in step:
        session.engine.goto(step["goto"])
        return TurnRun(f"(goto {step['goto']})", "goto", "", None, None)
    if "tick" in step:                              # advance the autonomous clock
        session.engine.tick(int(step["tick"]))
        return TurnRun(f"(tick {step['tick']}s)", "clock", "", None, None)
    if "action" in step:                            # a panel click (no interpret LLM)
        a = step["action"]
        turn = run_action(session, a["op"], a.get("target"), a.get("action_text", ""))
        return TurnRun(f"(click {a['op']})", "staff", "", turn,
                       _last_role(session))

    # ── a spoken turn ─────────────────────────────────────────────────────────
    # Post-Phase-2 the server Director is the single router: every utterance goes
    # through run_turn (raw transcript + optional chip), and the server decides who
    # answers. `endpoint` is derived from the resolved speaker, not chosen here.
    said = step["say"]
    stub.intent_json = _intent_json(step.get("intent", {"op": "chitchat"}))
    forced = step.get("forced_addressee")
    turn = run_turn(session, said, forced_addressee=forced)
    endpoint = "patient" if turn.speaker == "patient" else "staff"
    return TurnRun(said, endpoint, turn.speaker, turn, _last_role(session))


def _intent_json(intent: dict) -> str:
    import json
    return json.dumps({
        "op": intent.get("op", "chitchat"),
        "target": intent.get("target"),
        "action_text": intent.get("action_text", ""),
        "clarify": bool(intent.get("clarify", False)),
        "addressee": intent.get("addressee"),
    })


def _last_role(session: VoiceSession) -> str | None:
    return session.history[-1]["role"] if session.history else None


# ── script loading + assertion ────────────────────────────────────────────────

def load_script(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def list_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.yaml"))


def run_script(script: dict, monkeypatch) -> ConversationRun:
    """Build a session on the script's case and play every turn (stubs installed)."""
    stub = StubLLM()
    install_stubs(monkeypatch, stub)
    session = start_session(script["case"], script.get("resident", "resident-test"))
    address = build_address_index(build_roster(session))
    steps = list(script.get("setup", [])) + list(script["turns"])
    turns = play(session, steps, stub, address)
    return ConversationRun(session=session, turns=turns)


def check_turn(tr: TurnRun, expect: dict) -> list[str]:
    """Return a list of human-readable mismatches for one turn (empty == pass)."""
    fails: list[str] = []

    def cmp(key: str, actual, want):
        if actual != want:
            fails.append(f"[{tr.say!r}] {key}: got {actual!r}, want {want!r}")

    if "endpoint" in expect:
        cmp("endpoint", tr.endpoint, expect["endpoint"])
    if "route_speaker" in expect:
        cmp("route_speaker", tr.route_speaker, expect["route_speaker"])
    if "persisted_role" in expect:
        cmp("persisted_role", tr.persisted_role, expect["persisted_role"])

    t = tr.turn
    if t is None:
        if any(k in expect for k in ("speaker", "op", "target", "transitioned",
                                     "critical_error", "state", "reveals", "ended")):
            fails.append(f"[{tr.say!r}] expected a grounded turn but got none "
                         f"(endpoint={tr.endpoint})")
        return fails

    if "speaker" in expect:
        cmp("speaker", t.speaker, expect["speaker"])
    if "op" in expect:
        cmp("op", t.intent.op, expect["op"])
    if "target" in expect:
        cmp("target", t.intent.target, expect["target"])
    if "transitioned" in expect:
        cmp("transitioned", t.transitioned, expect["transitioned"])
    if "critical_error" in expect:
        cmp("critical_error", t.critical_error, expect["critical_error"])
    if "ended" in expect:
        cmp("ended", t.ended, expect["ended"])
    if "state" in expect:
        cmp("state", t.state_id, expect["state"])
    if "reveals" in expect:
        got = (t.revealed or "")
        if expect["reveals"] is None:              # `reveals: null` — nothing grounded
            if got:
                fails.append(f"[{tr.say!r}] reveals: expected nothing, got {got!r}")
        elif expect["reveals"].lower() not in got.lower():
            fails.append(f"[{tr.say!r}] reveals: {expect['reveals']!r} not in {got!r}")
    if "card_kind" in expect:
        cmp("card_kind", (t.diagnostic or {}).get("kind"), expect["card_kind"])
    if "reply_lacks" in expect:            # a phrase that must NOT be voiced (C4 leaks)
        for phrase in ([expect["reply_lacks"]] if isinstance(expect["reply_lacks"], str)
                       else expect["reply_lacks"]):
            if phrase.lower() in (t.reply or "").lower():
                fails.append(f"[{tr.say!r}] reply leaks {phrase!r}: {t.reply!r}")
    return fails


def check_final(run: ConversationRun, final: dict) -> list[str]:
    """Assert the terminal `final:` block (state, working dx, safety, score)."""
    from backend import sim_scoring
    fails: list[str] = []
    engine = run.session.engine

    if "reached_state" in final and engine.run.current_state_id != final["reached_state"]:
        fails.append(f"final state: got {engine.run.current_state_id!r}, "
                     f"want {final['reached_state']!r}")
    if "working_dx_contains" in final:
        dxs = " ".join(engine.run.working_diagnoses).lower()
        if final["working_dx_contains"].lower() not in dxs:
            fails.append(f"working dx {dxs!r} lacks {final['working_dx_contains']!r}")
    if "critical_error" in final:
        had = bool(engine.run.critical_errors)
        if had != final["critical_error"]:
            fails.append(f"final critical_error: got {had}, want {final['critical_error']}")
    if "ended" in final and engine.run.ended != final["ended"]:
        fails.append(f"final ended: got {engine.run.ended}, want {final['ended']}")

    criteria = sim_scoring.score_run(run.session)
    score = sim_scoring.overall_score(criteria)
    if "min_score" in final and score < final["min_score"]:
        fails.append(f"score {score} < min_score {final['min_score']}")
    if "max_score" in final and score > final["max_score"]:
        fails.append(f"score {score} > max_score {final['max_score']}")
    return fails
