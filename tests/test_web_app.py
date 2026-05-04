from __future__ import annotations

import json

from backend.investigations import InvestigationService
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult


def sample_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "myocardial infarction",
            "patient": {
                "age": 58,
                "gender": "male",
                "history": ["hypertension", "current smoker"],
            },
            "symptoms": [
                "chest pressure",
                "shortness of breath",
                "sweating",
            ],
            "vitals": {
                "heart_rate": 104,
                "blood_pressure": "148/92",
                "respiratory_rate": 22,
                "oxygen_saturation": 95,
            },
            "progression": [
                {
                    "time_min": 0,
                    "description": "Pressure starts while walking upstairs.",
                },
                {
                    "time_min": 20,
                    "description": "Pressure intensifies with sweating.",
                },
            ],
            "risk_factors": ["smoking", "hypertension"],
            "exam_findings": ["diaphoretic", "anxious"],
            "outcome": {
                "if_treated": "Rapid emergency treatment improves outcomes.",
                "if_untreated": "Ongoing ischemia can lead to instability.",
            },
        }
    )


class FakeSimulator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_case(
        self,
        diagnosis: str,
        *,
        difficulty: str,
        session_history: list[object],
    ) -> ClinicalCase:
        self.calls.append((diagnosis, difficulty))
        return sample_case()


def test_generate_case_reuses_recent_cached_case(monkeypatch: object) -> None:
    from backend import web_app

    simulator = FakeSimulator()
    web_app.CASE_CACHE.clear()
    web_app.SESSIONS.clear()
    monkeypatch.setattr(web_app, "SIMULATOR", simulator)

    first = web_app.generate_case(
        web_app.GenerateCaseRequest(
            diagnosis="Myocardial Infarction",
            difficulty="medium",
        )
    )
    second = web_app.generate_case(
        web_app.GenerateCaseRequest(
            diagnosis="  myocardial   infarction  ",
            difficulty="Medium",
        )
    )

    assert simulator.calls == [("Myocardial Infarction", "medium")]
    assert first["session_id"] != second["session_id"]
    assert first["case_preview"] == second["case_preview"]
    assert web_app.SESSIONS[first["session_id"]].case is not web_app.SESSIONS[
        second["session_id"]
    ].case


def test_case_preview_does_not_expose_hidden_symptom_list() -> None:
    from backend import web_app

    preview = web_app.case_preview(sample_case())

    assert preview == {
        "age": 58,
        "gender": "male",
        "chief_complaint": "chest pressure",
        "vitals": {
            "heart_rate": 104,
            "blood_pressure": "148/92",
            "respiratory_rate": 22,
            "oxygen_saturation": 95,
        },
    }
    assert "presenting_symptoms" not in preview


def test_web_evaluation_preserves_exact_diagnosis_guess(monkeypatch: object) -> None:
    from backend import web_app

    class FakeEvaluator:
        def __init__(self) -> None:
            self.questions: list[str] | None = None

        def evaluate(self, case: ClinicalCase, questions: list[str], responses: list[str], **_: object) -> object:
            self.questions = questions

            class Result:
                def model_dump(self) -> dict[str, object]:
                    return {"ok": True}

            return Result()

    evaluator = FakeEvaluator()
    session_id = "diagnosis-guess-session"
    web_app.SESSIONS.clear()
    web_app.SESSIONS[session_id] = web_app.WebSession(case=sample_case())
    web_app.SESSIONS[session_id].diagnosis_guess = "Heart Attack"
    monkeypatch.setattr(web_app, "EVALUATOR", evaluator)

    web_app.evaluate(web_app.EvaluateRequest(session_id=session_id))

    assert evaluator.questions is not None
    assert "Diagnosis guess: Heart Attack" in evaluator.questions
    assert "could be heart attack" not in " ".join(evaluator.questions).lower()


def test_web_ui_includes_loading_states_for_async_actions() -> None:
    from backend import web_app

    html = web_app.HTML_PAGE

    assert 'id="askButton"' in html
    assert 'id="investigationButton"' in html
    assert 'id="doctorButton"' in html
    assert 'id="diagnosisButton"' in html
    assert 'id="evaluateButton"' in html
    assert "Asking patient..." in html
    assert "Ordering investigation..." in html
    assert "Calling doctor..." in html
    assert "Evaluating performance..." in html
    assert "Patient question failed." in html
    assert "Investigation order failed." in html
    assert "Doctor consult failed." in html
    assert "Evaluation failed." in html
    assert "result.investigations.forEach" in html
    assert ".ecg-tracing" in html
    assert "overflow-x: auto" in html


def test_ecg_student_display_hides_decoded_findings() -> None:
    from backend import web_app

    case = sample_case()
    full_result = InvestigationService().get_result(case, "ECG")
    display = web_app.student_facing_investigation_result(full_result, case)
    serialized = json.dumps(display).lower()

    assert display["summary"] == "ECG completed"
    assert display["visual_payload"]["svg"].startswith("<svg")
    assert "st elevation" not in serialized
    assert "reciprocal" not in serialized
    assert "territory" not in serialized
    assert "interpretation_hint" not in serialized


def test_troponin_student_display_hides_hints_and_flags() -> None:
    from backend import web_app

    case = sample_case()
    full_result = InvestigationService().get_result(case, "troponin")
    display = web_app.student_facing_investigation_result(full_result, case)
    serialized = json.dumps(display).lower()

    assert display["report"] == [
        {
            "name": "high-sensitivity troponin I",
            "value": 4.8,
            "unit": "ng/mL",
            "reference_range": "<0.04 ng/mL",
        }
    ]
    assert '"flag": "high"' not in serialized
    assert "flagged as high" not in serialized
    assert "ischemia" not in serialized
    assert "interpretation_hint" not in serialized
    assert "flag" not in serialized


def test_generic_investigation_student_display_hides_interpretation_text() -> None:
    from backend import web_app

    case = sample_case()
    full_result = InvestigationResult(
        investigation="Complete blood count",
        status="available",
        structured_result={
            "test": "Complete blood count",
            "findings": ["White blood cell count 14.2 10^9/L"],
            "values": [
                {
                    "name": "white blood cell count",
                    "value": 14.2,
                    "unit": "10^9/L",
                    "reference_range": "4.0-11.0 10^9/L",
                    "flag": "high",
                }
            ],
            "interpretation_hint": "This supports a bacterial infection.",
        },
        summary="CBC suggests bacterial infection.",
        visual_payload=None,
    )

    display = web_app.student_facing_investigation_result(full_result, case)
    serialized = json.dumps(display).lower()

    assert display["summary"] == "Complete blood count completed"
    assert display["report"] == [
        {
            "name": "white blood cell count",
            "value": 14.2,
            "unit": "10^9/L",
            "reference_range": "4.0-11.0 10^9/L",
        }
    ]
    assert "bacterial infection" not in serialized
    assert "interpretation_hint" not in serialized
    assert "flag" not in serialized


def test_investigation_endpoint_preserves_full_result_in_session(monkeypatch: object) -> None:
    from backend import web_app

    class FakeInvestigationService:
        def get_results(
            self,
            case: ClinicalCase,
            requested_investigations: str,
        ) -> list[InvestigationResult]:
            return [self.get_result(case, requested_investigations)]

        def get_result(
            self,
            case: ClinicalCase,
            requested_investigation: str,
        ) -> InvestigationResult:
            return InvestigationService().get_result(case, requested_investigation)

    session_id = "investigation-display-session"
    case = sample_case()
    web_app.SESSIONS.clear()
    web_app.SESSIONS[session_id] = web_app.WebSession(case=case)
    monkeypatch.setattr(web_app, "INVESTIGATION_SERVICE", FakeInvestigationService())

    display = web_app.order_investigation(
        web_app.InvestigationRequest(session_id=session_id, investigation="ECG")
    )

    assert "st_segments" not in json.dumps(display).lower()
    assert web_app.SESSIONS[session_id].investigations
    full_result = web_app.SESSIONS[session_id].investigations[0]
    assert "st_segments" in full_result.structured_result


def test_investigation_endpoint_returns_multiple_sanitized_displays(monkeypatch: object) -> None:
    from backend import web_app

    class FakeInvestigationService:
        def get_results(
            self,
            case: ClinicalCase,
            requested_investigations: str,
        ) -> list[InvestigationResult]:
            return InvestigationService().get_results(case, requested_investigations)

    session_id = "multi-investigation-session"
    case = sample_case()
    web_app.SESSIONS.clear()
    web_app.SESSIONS[session_id] = web_app.WebSession(case=case)
    monkeypatch.setattr(web_app, "INVESTIGATION_SERVICE", FakeInvestigationService())

    display = web_app.order_investigation(
        web_app.InvestigationRequest(session_id=session_id, investigation="ECG and troponin")
    )

    assert "investigations" in display
    assert len(display["investigations"]) == 2
    assert [result.investigation for result in web_app.SESSIONS[session_id].investigations] == [
        "ECG",
        "Troponin",
    ]
    serialized = json.dumps(display).lower()
    assert "st elevation" not in serialized
    assert "interpretation_hint" not in serialized
