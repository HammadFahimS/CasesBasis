"""Step-1 engine features for multi-case authoring, tested on a synthetic
three-state case so they stay independent of any real case's topology:

* ``Transition.requires`` — all-of / count clauses on action + inaction edges.
* Investigation results resolve in the *post-transition* state.
* ``image_ref_by_state`` → an image card + announce-only voicing by the holder.
* The resident-only asset route under ``cases/``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.simulation_engine import LearnerAction, SimulationEngine
from schemas.simulated_case import SimulatedCase


def _case(**overrides) -> SimulatedCase:
    doc = {
        "id": "synthetic",
        "meta": {"title": "Synthetic"},
        "patient": {"age": 40, "gender": "male"},
        "presentation": {"chief_complaint": "test", "initial_vitals": {"heart_rate": 80}},
        "actors": [
            {"id": "nurse", "role": "Nurse"},
            {"id": "rt", "role": "Respiratory therapist"},
        ],
        "investigations": [
            {
                "id": "scope", "name": "Video laryngoscopy", "category": "bedside",
                "result": "No view yet", "result_actor_id": "rt",
                "result_by_state": {"seen": "View obtained; image on the screen"},
                "image_ref_by_state": {"seen": "synthetic/view.png"},
            },
        ],
        "initial_state": "a",
        "states": [
            {
                "id": "a", "name": "A", "vitals": {"heart_rate": 80},
                "transitions": [
                    # all-of gate with no keywords: the completing action fires it
                    {"to_state": "b", "description": "both halves done",
                     "requires": [{"any": ["handover", "onset"]}, {"any": ["history", "meds"]}]},
                    {"to_state": "c", "description": "idle with the gate unmet", "kind": "inaction",
                     "after_seconds": 100,
                     "requires": [{"any": ["handover", "onset"]}, {"any": ["history", "meds"]}]},
                ],
            },
            {
                "id": "b", "name": "B", "vitals": {"heart_rate": 90},
                "transitions": [
                    # planned scope (keywords AND requires) listed before the bare one
                    {"to_state": "seen", "description": "planned scope",
                     "keywords": ["laryngoscop"], "requires": [{"any": ["ketamine", "plan"]}]},
                    {"to_state": "c", "description": "bare scope", "keywords": ["laryngoscop"]},
                    # a counted requirement
                    {"to_state": "seen", "description": "two doses",
                     "requires": [{"any": ["epinephrine"], "count": 2}]},
                ],
            },
            {"id": "c", "name": "C", "vitals": {"heart_rate": 100}},
            {"id": "seen", "name": "Seen", "vitals": {"heart_rate": 85}},
        ],
    }
    doc.update(overrides)
    return SimulatedCase.model_validate(doc)


# ── Transition.requires ──────────────────────────────────────────────────────
def test_all_of_requirement_needs_every_clause_in_any_order():
    e = SimulationEngine(_case())
    e.act(LearnerAction(kind="ask", text="current meds"))
    assert e.state.id == "a"                       # one clause only
    e.act(LearnerAction(kind="ask", text="ems handover"))
    assert e.state.id == "b"                       # second clause completes it


def test_inaction_with_requires_fires_only_while_unmet():
    idle = SimulationEngine(_case())
    idle.act(LearnerAction(kind="ask", text="ems handover"))   # half the gate
    idle.tick(100)
    assert idle.state.id == "c"

    done = SimulationEngine(_case())
    done.act(LearnerAction(kind="ask", text="onset while eating"))
    done.act(LearnerAction(kind="ask", text="past history"))
    assert done.state.id == "b"


def test_keyworded_transition_with_requires_falls_through_when_unmet():
    e = SimulationEngine(_case())
    e.goto("b")
    e.act(LearnerAction(kind="intervene", text="laryngoscopy now"))
    assert e.state.id == "c"                       # bare scope wins: plan never stated

    e = SimulationEngine(_case())
    e.goto("b")
    e.act(LearnerAction(kind="medicate", text="ketamine 100 mg"))
    e.act(LearnerAction(kind="intervene", text="laryngoscopy now"))
    assert e.state.id == "seen"                    # planned scope, listed first, wins


def test_single_utterance_can_satisfy_keywords_and_requires():
    e = SimulationEngine(_case())
    e.goto("b")
    e.act(LearnerAction(kind="intervene", text="ketamine, then laryngoscopy"))
    assert e.state.id == "seen"


def test_counted_requirement_needs_that_many_actions():
    e = SimulationEngine(_case())
    e.goto("b")
    e.act(LearnerAction(kind="medicate", text="epinephrine 1 mg"))
    assert e.state.id == "b"
    e.act(LearnerAction(kind="medicate", text="epinephrine 1 mg again"))
    assert e.state.id == "seen"


def test_requires_survive_snapshot_roundtrip():
    e = SimulationEngine(_case())
    e.act(LearnerAction(kind="ask", text="ems handover"))
    snap = e.snapshot()
    f = SimulationEngine(_case())
    f.restore(snap)
    f.act(LearnerAction(kind="ask", text="meds"))
    assert f.state.id == "b"


def test_schema_rejects_empty_requirement():
    with pytest.raises(ValueError):
        _case(investigations=[], states=[{
            "id": "a", "name": "A", "vitals": {},
            "transitions": [{"to_state": "a", "description": "x", "requires": [{"any": []}]}],
        }])


# ── result timing + image ────────────────────────────────────────────────────
def test_investigation_result_resolves_in_the_state_the_order_leads_to():
    e = SimulationEngine(_case())
    e.goto("b")
    e.act(LearnerAction(kind="medicate", text="ketamine"))
    step = e.order_investigation("scope")
    assert step.transitioned and e.state.id == "seen"
    assert step.revealed == "View obtained; image on the screen"


def test_schema_rejects_result_or_image_keyed_on_unknown_state():
    with pytest.raises(ValueError, match="unknown state"):
        _case(investigations=[{
            "id": "x", "name": "X", "image_ref_by_state": {"nope": "a.png"},
        }])


def test_image_card_carries_a_url_and_no_generated_values(monkeypatch):
    from backend import sim_diagnostics

    class Boom:
        def generate(self, *a, **k):
            raise AssertionError("no value generation for an image study")
    monkeypatch.setattr(sim_diagnostics, "_lab_values_agent", lambda: Boom())
    case = _case()
    inv = case.investigations[0]
    card = sim_diagnostics.build_investigation_card(
        inv, case.states[0].vitals, case.patient, "View obtained", state_id="seen")
    assert card["kind"] == "image"
    assert card["visual_payload"]["image"] == "/cases/assets/synthetic/view.png"
    assert card["values"] == []
    # no image in another state → plain report (values generation stubbed away)
    monkeypatch.setattr(sim_diagnostics, "_structured_values", lambda *a, **k: [])
    plain = sim_diagnostics.build_investigation_card(
        inv, case.states[0].vitals, case.patient, "No view yet", state_id="a")
    assert plain["kind"] == "report" and plain["visual_payload"] is None


def test_image_study_is_announced_by_its_holder_never_described(monkeypatch):
    from backend import director
    from backend.nurse_brain import NurseIntent
    from backend.voice_session import VoiceSession

    case = _case()
    engine = SimulationEngine(case)
    engine.goto("b")
    engine.act(LearnerAction(kind="medicate", text="ketamine"))
    result = engine.order_investigation("scope")
    session = VoiceSession(id="s", resident_id="r", case=case, engine=engine)
    monkeypatch.setattr(director, "speak", lambda *a, **k: "LLM must not be used",
                        raising=False)
    intent = NurseIntent(op="order_investigation", target="scope")
    reply, speaker = director.voice_reply(session, intent, result, {}, "scope him")
    assert speaker == "rt"
    assert "screen" in reply.lower()
    assert "view obtained" not in reply.lower()


# ── asset route ──────────────────────────────────────────────────────────────
class _Req:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}


def _signed_in():
    from backend.auth import COOKIE_NAME, make_session_cookie
    return _Req({COOKIE_NAME: make_session_cookie("res-1")})


def test_asset_route_serves_media_under_cases_to_a_resident(tmp_path, monkeypatch):
    from backend.routers import pages
    (tmp_path / "fb").mkdir()
    (tmp_path / "fb" / "view.png").write_bytes(b"\x89PNG")
    (tmp_path / "fb" / "case.json").write_text("{}")
    monkeypatch.setattr(pages, "_CASES_DIR", tmp_path)

    resp = pages.case_asset("fb/view.png", _signed_in())
    assert resp.status_code == 200 and resp.path.endswith("view.png")

    with pytest.raises(HTTPException) as exc:
        pages.case_asset("fb/view.png", _Req())
    assert exc.value.status_code == 401

    for bad in ("fb/case.json", "../secrets.png", "fb/missing.png"):
        with pytest.raises(HTTPException) as exc:
            pages.case_asset(bad, _signed_in())
        assert exc.value.status_code == 404


# ── cue timing ───────────────────────────────────────────────────────────────
def test_cue_after_seconds_overrides_the_default_linger():
    case = _case(investigations=[], states=[{
        "id": "a", "name": "A", "vitals": {},
        "cues": [{"text": "early nudge", "actor_id": "nurse", "after_seconds": 20},
                 {"text": "late nudge", "actor_id": "nurse"}],
    }])
    e = SimulationEngine(case)
    e.tick(19)
    assert e.due_cues() == []
    e.tick(1)
    assert e.due_cues()[0]["text"] == "early nudge"
    assert e.due_cues() == []                      # the default one still waits 75 s
    e.tick(55)
    assert e.due_cues()[0]["text"] == "late nudge"


def test_run_scoped_requirement_remembers_a_plan_voiced_before_the_state():
    case = _case()
    # make the planned-scope requirement run-scoped
    b = next(s for s in case.states if s.id == "b")
    doc = case.model_dump()
    for st in doc["states"]:
        if st["id"] == "b":
            st["transitions"][0]["requires"][0]["scope"] = "run"
    e = SimulationEngine(SimulatedCase.model_validate(doc))
    e.act(LearnerAction(kind="medicate", text="ketamine 100 mg"))   # in state a
    e.goto("b")                                                     # per-state list cleared
    e.act(LearnerAction(kind="intervene", text="laryngoscopy now"))
    assert e.state.id == "seen"

    f = SimulationEngine(SimulatedCase.model_validate(doc))
    f.goto("b")
    f.act(LearnerAction(kind="intervene", text="laryngoscopy now"))
    assert f.state.id == "c"                        # no plan anywhere → bare attempt
