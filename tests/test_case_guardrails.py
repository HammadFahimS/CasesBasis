"""Step-5 case-authored guardrails, on the synthetic case from
tests.test_engine_requirements so they stay topology-neutral:

* a state-scoped ``avoid`` + ``intercept`` medication item: the nurse withholds the
  order, re-parks it awaiting a dose, and the scorer reads the intercept log;
* consultant grounding via ``ActorRole.responses`` / ``fallback``;
* the nurse's "which tests?" for a bundle request.
"""

from __future__ import annotations

from backend import consult_voice, sim_scoring
from backend.medication import dose_phrase_matches, parse_med
from backend.nurse_brain import NurseIntent, _outcome_brief, dispatch
from backend.simulation_engine import SimulationEngine
from backend.voice_session import VoiceSession
from schemas.simulated_case import SimulatedCase
from tests.test_engine_requirements import _case


def _guarded_case() -> SimulatedCase:
    doc = _case().model_dump()
    doc["medication_rubric"] = [{
        "id": "iv_epi_perfusing", "label": "No arrest dose with a pulse",
        "drugs": ["epinephrine", "adrenaline"], "expectation": "avoid", "intercept": True,
        "dose_keywords": ["1 mg iv", "1 milligram iv", "arrest dose"],
        "states": ["a", "b"],                       # not in c (the "arrest" here)
        "note": "That's the arrest dose. With a pulse it's 0.5 milligrams IM.",
    }]
    doc["actors"].append({
        "id": "icu", "role": "ICU consultant",
        "responses": [{"states": ["seen"], "content": "We'll take him. Ventilator settings?"}],
        "fallback": "We're flat out. Diagnose and treat him first.",
    })
    doc["actors"].append({"id": "neuro", "role": "Neurology consultant"})
    return SimulatedCase.model_validate(doc)


def _session(case=None):
    case = case or _guarded_case()
    return VoiceSession(id="s", resident_id="r", case=case, engine=SimulationEngine(case))


# ── dose phrases ─────────────────────────────────────────────────────────────
def test_dose_phrases_match_whole_token_sequences():
    assert dose_phrase_matches("epinephrine 1 mg IV push", ["1 mg iv"])
    assert dose_phrase_matches("give one milligram IV", ["one milligram iv"])
    assert not dose_phrase_matches("epinephrine 0.1 mg IV", ["1 mg iv"])   # decimals intact
    assert not dose_phrase_matches("epinephrine 0.5 mg IM", ["1 mg iv"])


# ── the intercept ────────────────────────────────────────────────────────────
def test_intercepted_order_is_withheld_and_reparked_in_a_listed_state():
    s = _session(); e = s.engine
    intent = NurseIntent(op="order_medication", target="epinephrine", action_text="1 mg IV")
    result = dispatch(intent, e, "epinephrine 1 mg IV")
    assert result is None and e.state.id == "a"
    pending = e.run.pending_action
    assert pending and pending["status"] == "awaiting_info" and pending["intercepted"] == "iv_epi_perfusing"
    assert e.run.log[-1] == {**e.run.log[-1], "kind": "intercept", "detail": "iv_epi_perfusing"}
    assert "epinephrine" not in " ".join(e.run.actions)          # nothing reached the engine
    brief = _outcome_brief(intent, None, {}, engine=e, transcript="epinephrine 1 mg IV")
    assert "WITHHOLDING" in brief and "0.5 milligrams IM" in brief


def test_corrected_dose_fills_the_reparked_order_without_looping():
    s = _session(); e = s.engine
    dispatch(NurseIntent(op="order_medication", target="epinephrine", action_text="1 mg IV"), e, "epi 1 mg IV")
    result = dispatch(NurseIntent(op="chitchat"), e, "Fine, 0.5 milligrams IM.")
    assert e.run.pending_action is None
    assert result is not None
    assert e.run.actions[-1] == "epinephrine 0.5 milligrams im"   # drug, dose AND route reach the engine


def test_wrong_dose_supplied_to_a_parked_order_is_intercepted_too():
    s = _session(); e = s.engine
    dispatch(NurseIntent(op="order_medication", target="epinephrine", action_text=""), e, "give epinephrine")
    assert e.run.pending_action["status"] == "awaiting_info"        # high-alert, no dose → parked
    result = dispatch(NurseIntent(op="chitchat"), e, "1 milligram IV.")
    assert result is None
    assert e.run.pending_action["intercepted"] == "iv_epi_perfusing"


def test_same_order_executes_in_an_unlisted_state():
    s = _session(); e = s.engine
    e.goto("c")
    dispatch(NurseIntent(op="order_medication", target="epinephrine", action_text="1 mg IV"), e, "epi 1 mg IV")
    assert e.run.pending_action is None
    assert e.run.actions[-1] == "epinephrine 1 mg iv"


def test_scorer_penalises_the_intercept_but_not_the_legitimate_arrest_dose():
    clean = _session()
    clean.engine.goto("c")
    dispatch(NurseIntent(op="order_medication", target="epinephrine", action_text="1 mg IV"),
             clean.engine, "epi 1 mg IV")
    crit = next(c for c in sim_scoring.score_run(clean) if c.id == "med-iv_epi_perfusing")
    assert crit.met is True

    tripped = _session()
    dispatch(NurseIntent(op="order_medication", target="epinephrine", action_text="1 mg IV"),
             tripped.engine, "epi 1 mg IV")
    crit = next(c for c in sim_scoring.score_run(tripped) if c.id == "med-iv_epi_perfusing")
    assert crit.met is False and crit.guardrail


def test_unlisted_case_items_keep_the_old_avoid_semantics():
    """No states, no dose phrases, no intercept → the drug alone trips it (myasthenia's
    pyridostigmine item), exactly as before."""
    doc = _case().model_dump()
    doc["medication_rubric"] = [{"id": "no_pyrido", "label": "Avoided pyridostigmine",
                                 "drugs": ["pyridostigmine"], "expectation": "avoid"}]
    s = _session(SimulatedCase.model_validate(doc))
    dispatch(NurseIntent(op="order_medication", target="pyridostigmine", action_text="60 mg"),
             s.engine, "pyridostigmine 60 mg")
    from backend.sim_diagnostics import build_order_card
    s.cards.append(build_order_card("pyridostigmine", "60 mg", "", status="given"))
    crit = next(c for c in sim_scoring.score_run(s) if c.id == "med-no_pyrido")
    assert crit.met is False


# ── consultant grounding ─────────────────────────────────────────────────────
def test_consultant_voices_the_state_scoped_content_or_the_fallback(monkeypatch):
    captured = {}
    monkeypatch.setattr(consult_voice, "_complete",
                        lambda sy, us: captured.__setitem__("user", us) or "[reply]")
    s = _session()
    consult_voice.consult_reply(s, "icu", "ICU, will you take him?")
    assert "flat out" in captured["user"] and "Ventilator settings" not in captured["user"]
    s.engine.goto("seen")
    consult_voice.consult_reply(s, "icu", "ICU, he's tubed.")
    assert "Ventilator settings" in captured["user"] and "flat out" not in captured["user"]


def test_consultant_without_responses_keeps_free_role_play(monkeypatch):
    captured = {}
    monkeypatch.setattr(consult_voice, "_complete",
                        lambda sy, us: captured.__setitem__("user", us) or "[reply]")
    consult_voice.consult_reply(_session(), "neuro", "Neurology?")
    assert "What you actually tell them" not in captured["user"]


def test_schema_rejects_unknown_states_in_responses_and_rubric():
    import pytest
    doc = _guarded_case().model_dump()
    doc["actors"][-2]["responses"][0]["states"] = ["nope"]
    with pytest.raises(ValueError, match="unknown state"):
        SimulatedCase.model_validate(doc)
    doc = _guarded_case().model_dump()
    doc["medication_rubric"][0]["states"] = ["nope"]
    with pytest.raises(ValueError, match="unknown state"):
        SimulatedCase.model_validate(doc)


# ── "which tests?" ───────────────────────────────────────────────────────────
def test_bundle_request_orders_nothing_and_asks_which_tests():
    s = _session(); e = s.engine
    intent = NurseIntent(op="order_investigation", target=None, clarify=True)
    assert dispatch(intent, e, "send some bloods") is None
    assert e.run.ordered_investigations == []
    assert "Which specific tests" in _outcome_brief(intent, None, {}, engine=e, transcript="send some bloods")
