from __future__ import annotations

import pytest

from backend import nurse_brain
from backend.nurse_brain import (
    NurseIntent,
    dispatch,
    run_turn,
    _build_card,
    _is_full_exam_request,
    _parse_intent,
)
from backend.simulated_case_store import load_seed_case
from backend.simulation_engine import SimulationEngine


def _engine():
    return SimulationEngine(load_seed_case("myasthenic_crisis"))


# ── pure dispatch (no LLM, no DB) ────────────────────────────────────────────
def test_dispatch_order_investigation_returns_real_result():
    engine = _engine()
    result = dispatch(NurseIntent(op="order_investigation", target="ecg"), engine, "get an ECG")
    assert result is not None
    # The raw result hands the tracing over for the resident to read; the clinical
    # interpretation ("sinus tachycardia / no ischaemia") is deliberately withheld.
    assert "monitor" in (result.revealed or "").lower()
    assert "ischaemic" not in (result.revealed or "").lower()


def test_dispatch_gated_investigation_reports_pending():
    # The venous gas is gated 120s; ordering it at t=0 must not leak a result — it
    # schedules a callback (the room talks back) instead of a dead pending.
    engine = _engine()
    result = dispatch(NurseIntent(op="order_investigation", target="vbg"), engine, "get a gas")
    assert result is not None
    assert result.revealed is None                       # nothing handed over yet
    assert "pH" not in (result.message or "")            # and the message can't leak it
    assert "come back" in result.message.lower()         # it's coming, not dropped
    assert any(ev["inv_id"] == "vbg" for ev in engine.run.deferred_events)


def test_dispatch_ask_history_reveals_real_beat():
    engine = _engine()
    result = dispatch(NurseIntent(op="ask_history", target="ems_handover"), engine, "ems story")
    assert result is not None
    assert (result.revealed or "").startswith("Called 20")


# ── addressee override (deterministic, no LLM) ───────────────────────────────
def test_addressee_keeps_history_with_the_person_addressed():
    # "EMS, what's the story?" must reach the paramedic even when the interpreter
    # picked the richer patient beat by content; "Husband, …" must reach family.
    engine = _engine()
    out = nurse_brain._apply_addressee(
        NurseIntent(op="ask_history", target="patient_symptoms"), "EMS, what's the story?", engine
    )
    assert out.op == "ask_history" and out.target == "ems_handover"

    out = nurse_brain._apply_addressee(
        NurseIntent(op="ask_history", target="ems_handover"), "Husband, what's her background?", engine
    )
    assert out.op == "ask_history" and out.target in ("husband_collateral", "husband_family_history")


def test_addressee_coerces_chitchat_when_a_bystander_is_named():
    engine = _engine()
    out = nurse_brain._apply_addressee(
        NurseIntent(op="chitchat"), "Paramedics, walk me through it.", engine
    )
    assert out.op == "ask_history" and out.target == "ems_handover"


def test_addressee_keeps_specific_beat_when_class_already_matches():
    # The interpreter already chose a family beat — don't clobber its specificity.
    engine = _engine()
    out = nurse_brain._apply_addressee(
        NurseIntent(op="ask_history", target="husband_family_history"),
        "Husband, any family history?", engine,
    )
    assert out.target == "husband_family_history"


def test_addressee_leaves_unaddressed_and_staff_utterances_alone():
    engine = _engine()
    unaddressed = NurseIntent(op="ask_history", target="patient_symptoms")
    assert nurse_brain._apply_addressee(unaddressed, "What's been happening?", engine).target == "patient_symptoms"
    # Addressing the nurse to give a drug must not be rerouted to a history beat.
    med = NurseIntent(op="order_medication", target="rocuronium", action_text="0.6 mg/kg")
    assert nurse_brain._apply_addressee(med, "Nurse, give rocuronium.", engine).op == "order_medication"


def test_dispatch_perform_action_trips_transition():
    engine = _engine()
    assert engine.run.current_state_id == "baseline"
    result = dispatch(
        NurseIntent(op="perform_action", action_text="let's get a blood gas"),
        engine, "let's get a blood gas",
    )
    assert result is not None and result.transitioned
    assert engine.run.current_state_id == "recognition"


def test_dispatch_order_medication_rocuronium_advances_to_post_intubation():
    # The case's crux: in MG, rocuronium (a non-depolariser) is the correct
    # paralytic — ordering it must still trip the intended transition through the
    # new order_medication path.
    engine = _engine()
    engine.goto("intubation_prep")
    intent = NurseIntent(op="order_medication", target="rocuronium", action_text="0.6 mg/kg")
    result = dispatch(intent, engine, "give rocuronium 0.6 mg/kg")
    assert result is not None and result.transitioned
    assert engine.run.current_state_id == "post_intubation"


def test_dispatch_standard_succinylcholine_trips_critical_pea_arrest():
    # A standard succinylcholine dose in MG is the authored critical error — the
    # new medication path must still surface it as a critical transition.
    engine = _engine()
    engine.goto("intubation_prep")
    intent = NurseIntent(op="order_medication", target="succinylcholine", action_text="standard dose 1.5 mg/kg")
    result = dispatch(intent, engine, "give the usual succinylcholine")
    assert result is not None and result.transitioned and result.critical_error
    assert engine.run.current_state_id == "pea_arrest"


def test_dispatch_diagnose_records_working_diagnosis():
    engine = _engine()
    intent = NurseIntent(op="diagnose", action_text="myasthenic crisis")
    result = dispatch(intent, engine, "I think this is myasthenic crisis")
    assert result is not None
    assert engine.run.working_diagnoses == ["myasthenic crisis"]


def test_dispatch_read_vitals_does_not_mutate():
    engine = _engine()
    result = dispatch(NurseIntent(op="read_vitals"), engine, "what's her heart rate")
    assert result is None
    assert engine.run.current_state_id == "baseline"


def test_read_vitals_reports_only_the_vitals_asked_for():
    """The nurse's spoken brief must name only the vitals the resident asked for —
    it used to dump the whole monitor, so 'BP and HR' recited RR/SpO2/Temp too."""
    from backend.nurse_brain import _outcome_brief, _requested_vital_keys
    vitals = {"heart_rate": 108, "blood_pressure": "126/85", "respiratory_rate": 18,
              "oxygen_saturation": 98, "temperature_c": 37.1, "rhythm": "sinus tachycardia"}
    # a specific ask (rhythm rides with heart rate); generic ask → None (full set)
    assert _requested_vital_keys("nurse, get me BP and HR", vitals) == \
        ["blood_pressure", "heart_rate", "rhythm"]
    assert _requested_vital_keys("what are her vitals?", vitals) is None
    # even when a generic word co-occurs, named specifics win
    assert _requested_vital_keys("get me her obs, BP and HR", vitals) == \
        ["blood_pressure", "heart_rate", "rhythm"]

    brief = _outcome_brief(NurseIntent(op="read_vitals"), None, vitals, transcript="get me BP and HR")
    assert "126/85" in brief and "108" in brief
    # unrelated vitals must not leak into the report the model speaks
    assert "respiratory_rate" not in brief and "oxygen_saturation" not in brief and "98" not in brief


def test_read_vitals_generic_ask_reports_the_full_set():
    from backend.nurse_brain import _outcome_brief
    vitals = {"heart_rate": 108, "respiratory_rate": 18, "oxygen_saturation": 98}
    brief = _outcome_brief(NurseIntent(op="read_vitals"), None, vitals, transcript="what are her obs?")
    assert "respiratory_rate" in brief and "oxygen_saturation" in brief


def test_dispatch_perform_exam_reveals_and_can_transition():
    engine = _engine()
    result = dispatch(NurseIntent(op="perform_exam", target="Neurological"), engine, "neuro exam")
    assert result is not None and result.revealed
    # A neuro exam in baseline is an expected action that advances the case.
    assert engine.run.current_state_id == "recognition"


def test_full_exam_matcher_distinguishes_generic_from_focused():
    for phrase in ("do a physical exam", "perform a full examination",
                   "examine the patient", "head to toe exam"):
        assert _is_full_exam_request(phrase) is True
    for phrase in ("examine her chest", "check her reflexes", "listen to the lungs"):
        assert _is_full_exam_request(phrase) is False


def test_generic_physical_exam_reveals_every_authored_system():
    engine = _engine()
    all_systems = {e.system for e in engine.case.exam}
    assert len(all_systems) > 1  # guard: the case must have a multi-system exam
    intent = NurseIntent(op="perform_exam", target=None)
    result = dispatch(intent, engine, "do a full physical exam")
    assert result is not None and result.revealed
    # Every documented system appears in the one-shot full exam.
    for system in all_systems:
        assert system in result.revealed
    card = _build_card(intent, engine, result)
    assert card is not None and card["kind"] == "exam"
    assert card["title"] == "Physical exam"


def test_full_exam_phrase_overrides_a_stray_focused_target():
    # Even if the classifier tags a single system, an explicit full-exam phrase
    # reveals the whole exam, not just that one system.
    engine = _engine()
    intent = NurseIntent(op="perform_exam", target="Neurological")
    result = dispatch(intent, engine, "just do a complete physical examination")
    revealed_systems = [e.system for e in engine.case.exam if e.system in (result.revealed or "")]
    assert len(revealed_systems) == len({e.system for e in engine.case.exam})


# ── LLM intent matcher (for soft action transitions) ─────────────────────────
def test_make_intent_matcher_yes(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Yes.")
    matcher = nurse_brain.make_intent_matcher()
    assert matcher("secure the airway now", "the resident opts to intubate") is True


def test_make_intent_matcher_no_and_failclosed(monkeypatch):
    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "no")
    assert nurse_brain.make_intent_matcher()("give fluids", "intubate") is False

    def boom(s, u):
        raise RuntimeError("llm down")
    monkeypatch.setattr(nurse_brain, "_complete", boom)
    assert nurse_brain.make_intent_matcher()("anything", "trigger") is False


# ── defensive JSON parsing ───────────────────────────────────────────────────
def test_parse_intent_extracts_json_from_noise():
    intent = _parse_intent('sure: {"op": "order_investigation", "target": "ecg", "action_text": ""}')
    assert intent.op == "order_investigation"
    assert intent.target == "ecg"


def test_parse_intent_falls_back_to_chitchat():
    assert _parse_intent("not json at all").op == "chitchat"
    assert _parse_intent('{"op": "teleport"}').op == "chitchat"


# ── run_turn with a stubbed LLM ──────────────────────────────────────────────
@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "nurse_brain_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-nurse-brain")
    from backend.db import run_migrations
    from backend.simulated_case_repo import seed_gold_cases
    run_migrations()
    seed_gold_cases()
    yield path


def test_run_turn_grounds_reply_in_engine(db_path, monkeypatch):
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        if "routing brain" in system_prompt:
            return '{"op": "order_investigation", "target": "ecg", "action_text": ""}'
        # speak step: the real (raw) ECG result must have reached this prompt, and
        # the speak rubric must forbid the nurse from interpreting it.
        assert "monitor" in user_prompt
        assert "do NOT add any interpretation" in user_prompt
        return "The ECG tracing is up on the monitor for you to read."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    # Stub the image generator so the turn stays offline (real ECG = an LLM call).
    monkeypatch.setattr(
        nurse_brain, "build_investigation_card",
        lambda inv, vitals, patient, authored: {
            "kind": "ecg", "id": inv.id, "title": inv.name,
            "summary": authored, "visual_payload": {"type": "svg", "svg": "<svg/>"},
        },
    )

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "pull up the ECG")

    assert turn.intent.op == "order_investigation"
    assert "monitor" in (turn.revealed or "").lower()
    assert "monitor" in turn.reply.lower()
    assert turn.vitals.get("heart_rate")
    assert turn.diagnostic and turn.diagnostic["kind"] == "ecg"   # drawer card produced
    # the exchange is recorded for context on the next turn
    assert session.history[-2:] == [
        {"role": "resident", "text": "pull up the ECG"},
        {"role": "nurse", "text": turn.reply},
    ]


# ── opening narration (the nurse's bedside handoff) ──────────────────────────
def test_opening_narration_phrases_and_records(db_path, monkeypatch):
    from backend.voice_session import start_session

    captured = {}

    def fake_complete(system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "This is E.K., here with worsening shortness of breath. Begin when ready."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    session = start_session("myasthenic_crisis", "resident-1")
    intro = nurse_brain.opening_narration(session)

    assert "bedside nurse" in captured["system"]
    # the chief complaint is handed to the phrasing step
    assert session.case.presentation.chief_complaint in captured["user"]
    assert intro.startswith("This is E.K.")
    # recorded on the nurse channel so the next turn has context
    assert session.history[-1] == {"role": "nurse", "text": intro}
    assert session.patient_history == []  # patient channel untouched


def test_opening_narration_falls_back_when_llm_down(db_path, monkeypatch):
    from backend.voice_session import start_session

    def boom(s, u):
        raise RuntimeError("llm down")

    monkeypatch.setattr(nurse_brain, "_complete", boom)
    session = start_session("myasthenic_crisis", "resident-1")
    intro = nurse_brain.opening_narration(session)

    # the deterministic fallback still names the chief complaint and is recorded
    assert session.case.presentation.chief_complaint in intro
    assert session.history[-1]["text"] == intro


def test_run_turn_consult_produces_consult_card(db_path, monkeypatch):
    from backend import consult_voice
    from backend.voice_session import start_session

    def fake_route(system_prompt, user_prompt):
        # the consultant catalogue must reach the router so it can pick a target
        assert "Consultants you can call" in user_prompt
        return '{"op": "request_consult", "target": "neuro", "action_text": ""}'

    monkeypatch.setattr(nurse_brain, "_complete", fake_route)
    # the consultant role-play rides its own seam
    monkeypatch.setattr(
        consult_voice, "_complete",
        lambda s, u: "Neurology here — intubate early and start IVIG.",
    )

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "call neurology")

    assert turn.intent.op == "request_consult"
    assert turn.speaker == "consult"                  # voiced distinctly from the nurse
    assert turn.diagnostic and turn.diagnostic["kind"] == "consult"
    assert turn.diagnostic["category"] == "consult"   # routes to the Consults pane
    assert turn.diagnostic["title"] == "Neurology consultant"
    assert "IVIG" in turn.reply
    # the consult is recorded once, under the speaker who actually voiced it (S2):
    # the consultant, not a blanket "nurse".
    assert session.history[-2:] == [
        {"role": "resident", "text": "call neurology"},
        {"role": "consult", "text": turn.reply},
    ]


def test_run_action_dispatches_without_interpret(db_path, monkeypatch):
    """A panel click goes straight to dispatch — no interpret (routing) LLM call."""
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        # the router must never be reached for a click
        assert "routing brain" not in system_prompt
        return "Cardiovascular exam: regular rhythm, no murmurs."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)
    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "perform_exam", "Cardiovascular")

    assert turn.intent.op == "perform_exam"
    assert turn.diagnostic and turn.diagnostic["category"] == "exam"
    # the click is recorded as a natural resident utterance
    assert session.history[-2] == {"role": "resident", "text": "Examine the Cardiovascular system."}


def test_run_action_consult_builds_card(db_path, monkeypatch):
    from backend import consult_voice
    from backend.voice_session import start_session

    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "routing should not run")
    monkeypatch.setattr(consult_voice, "_complete", lambda s, u: "Neurology — intubate early.")

    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "request_consult", "neuro")

    assert turn.diagnostic and turn.diagnostic["kind"] == "consult"
    assert turn.diagnostic["title"] == "Neurology consultant"
    assert session.history[-2]["text"] == "Call Neurology consultant."


def test_run_action_medication_produces_order_card(db_path, monkeypatch):
    from backend.voice_session import start_session

    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Rocuronium 0.6 mg/kg, drawing it up.")
    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "order_medication", "rocuronium", "rocuronium 0.6 mg/kg")

    assert turn.intent.op == "order_medication"
    assert turn.diagnostic and turn.diagnostic["kind"] == "order"
    assert turn.diagnostic["category"] == "orders"          # routes to the Orders pane
    assert turn.diagnostic["dose"] == "0.6 mg/kg"
    assert turn.diagnostic["status"] == "given"
    assert turn.diagnostic["id"] == "med-rocuronium"        # stable id → upserts on re-order


def test_run_action_paralytic_without_dose_requests_confirmation(db_path, monkeypatch):
    from backend.voice_session import start_session

    monkeypatch.setattr(nurse_brain, "_complete", lambda s, u: "Confirm the rocuronium dose before I draw it up.")
    session = start_session("myasthenic_crisis", "resident-1")
    turn = nurse_brain.run_action(session, "order_medication", "rocuronium", "rocuronium")

    # high-alert drug, no dose stated → the nurse holds it for confirmation
    assert turn.diagnostic["status"] == "confirm"
    assert turn.diagnostic["confirm_required"] is True
    assert turn.diagnostic["dose"] is None


def test_run_action_rejects_unknown_op(db_path):
    from backend.voice_session import start_session

    session = start_session("myasthenic_crisis", "resident-1")
    with pytest.raises(ValueError):
        nurse_brain.run_action(session, "teleport", "x")


def test_session_payload_menu_offers_only_case_items(db_path):
    from backend.voice_session import start_session
    from backend.web_app import _voice_session_payload

    session = start_session("myasthenic_crisis", "resident-1")
    menu = _voice_session_payload(session)["menu"]

    assert "Cardiovascular" in menu["exam_systems"]
    cats = {i["category"] for i in menu["investigations"]}
    assert {"lab", "ecg", "imaging", "bedside"} <= cats
    # consultants are pageable; the patient/nurse are not in the call list
    cids = {c["id"] for c in menu["consultants"]}
    assert {"neuro", "icu"} <= cids
    assert "patient" not in cids and "nurse" not in cids


def test_session_payload_roster_tags_each_voice_with_a_speaker_class(db_path):
    from backend.voice_session import start_session
    from backend.web_app import _voice_session_payload

    session = start_session("myasthenic_crisis", "resident-1")
    roster = {r["id"]: r["speaker"] for r in _voice_session_payload(session)["roster"]}

    assert roster["patient"] == "patient"
    assert roster["nurse"] == "nurse"
    assert roster["ems"] == "ems"
    assert roster["husband"] == "family"   # family bystander, not a consultant
    assert roster["rt"] == "rt"
    assert roster["neuro"] == "consult" and roster["icu"] == "consult"


def test_run_turn_exam_produces_card(db_path, monkeypatch):
    from backend.voice_session import start_session

    def fake_complete(system_prompt, user_prompt):
        if "routing brain" in system_prompt:
            return '{"op": "perform_exam", "target": "Neurological", "action_text": ""}'
        return "On the neuro exam she has bilateral ptosis and proximal weakness."

    monkeypatch.setattr(nurse_brain, "_complete", fake_complete)

    session = start_session("myasthenic_crisis", "resident-1")
    turn = run_turn(session, "examine her neuro")

    assert turn.intent.op == "perform_exam"
    assert turn.diagnostic and turn.diagnostic["kind"] == "exam"
    assert turn.diagnostic["title"] == "Neurological exam"
