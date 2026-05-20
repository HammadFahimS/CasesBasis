from __future__ import annotations

import json
from typing import Any

from backend.doctor_consult import DoctorConsultService
from backend.investigations import InvestigationService
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.investigation import InvestigationResult


def mi_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "myocardial infarction",
            "patient": {
                "age": 58,
                "gender": "male",
                "history": ["hypertension", "current smoker", "high cholesterol"],
            },
            "symptoms": [
                "chest pressure",
                "shortness of breath",
                "nausea",
                "sweating",
                "left arm discomfort",
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
                    "description": "Pressure intensifies with sweating and nausea.",
                },
            ],
            "risk_factors": ["smoking", "hypertension", "high cholesterol"],
            "exam_findings": {"General": ["diaphoretic", "anxious"]},
            "outcome": {
                "if_treated": "Rapid emergency treatment improves outcomes.",
                "if_untreated": "Ongoing ischemia can lead to instability.",
            },
        }
    )


def critical_case() -> ClinicalCase:
    data = mi_case().model_dump()
    data["vitals"] = {
        "heart_rate": 138,
        "blood_pressure": "82/50",
        "respiratory_rate": 30,
        "oxygen_saturation": 88,
    }
    return ClinicalCase.model_validate(data)


def consult_response(
    *,
    urgency_level: str = "urgent",
    concise_advice: str = "I only have a minute: stabilize him and get an ECG.",
    recommended_next_steps: list[str] | None = None,
    missing_information: list[str] | None = None,
    safety_warnings: list[str] | None = None,
    should_escalate: bool = True,
) -> DoctorConsultResponse:
    return DoctorConsultResponse.model_validate(
        {
            "urgency_level": urgency_level,
            "concise_advice": concise_advice,
            "recommended_next_steps": recommended_next_steps
            if recommended_next_steps is not None
            else ["Repeat vitals", "Get ECG", "Send troponin"],
            "missing_information": missing_information
            if missing_information is not None
            else ["Oxygen saturation trend", "Focused pain history"],
            "safety_warnings": safety_warnings
            if safety_warnings is not None
            else ["Escalate if pain persists or vitals worsen"],
            "should_escalate": should_escalate,
        }
    )


class CapturingConsultAgent:
    def __init__(self, response: DoctorConsultResponse | None = None) -> None:
        self.response = response or consult_response()
        self.calls: list[tuple[dict[str, Any], str]] = []

    def consult(
        self,
        redacted_context: dict[str, Any],
        consult_question: str,
    ) -> DoctorConsultResponse:
        self.calls.append((redacted_context, consult_question))
        return self.response


class ContextAwareConsultAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []

    def consult(
        self,
        redacted_context: dict[str, Any],
        consult_question: str,
    ) -> DoctorConsultResponse:
        self.calls.append((redacted_context, consult_question))
        payload = json.dumps(redacted_context).lower()
        vitals = redacted_context["vitals"]
        systolic = int(str(vitals["blood_pressure"]).split("/")[0])
        if systolic < 90 or vitals["oxygen_saturation"] < 90:
            return consult_response(
                urgency_level="critical",
                concise_advice="This patient is unstable; get senior help now.",
                recommended_next_steps=["Place on monitor", "Give oxygen", "Get ECG now"],
                missing_information=["Current mental status", "Repeat blood pressure"],
                safety_warnings=["Hypotension or hypoxia needs immediate escalation"],
                should_escalate=True,
            )
        if "st elevation" in payload or "troponin" in payload:
            return consult_response(
                recommended_next_steps=["Review ECG immediately", "Trend troponin", "Keep on monitor"],
                missing_information=["Exact symptom onset time"],
            )
        return consult_response(urgency_level="routine", should_escalate=False)


def test_consult_doctor_does_not_receive_hidden_diagnosis_or_outcome() -> None:
    agent = CapturingConsultAgent()
    service = DoctorConsultService(agent=agent)

    service.consult(
        mi_case(),
        ["Where is your pain?"],
        ["It is a heavy chest pressure."],
        [],
        "What should I do next?",
        session_context={
            "diagnosis": "myocardial infarction",
            "outcome": {"if_untreated": "hidden"},
        },
    )

    assert agent.calls
    redacted_context, consult_question = agent.calls[0]
    serialized_context = json.dumps(redacted_context).lower()
    assert "myocardial infarction" not in serialized_context
    assert "if_treated" not in serialized_context
    assert "if_untreated" not in serialized_context
    assert "outcome" not in serialized_context
    assert consult_question == "What should I do next?"


def test_direct_diagnosis_request_does_not_reveal_diagnosis() -> None:
    agent = CapturingConsultAgent(
        consult_response(
            concise_advice="This is myocardial infarction, an MI or heart attack.",
            recommended_next_steps=["Treat myocardial infarction"],
            safety_warnings=["Heart attack can deteriorate"],
        )
    )
    service = DoctorConsultService(agent=agent)

    response = service.consult(mi_case(), [], [], [], "What is the diagnosis?")
    serialized = response.model_dump_json().lower()

    assert "myocardial infarction" not in serialized
    assert "heart attack" not in serialized
    assert " mi " not in f" {serialized} "


def test_consult_recommends_next_steps_from_available_findings() -> None:
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")
    agent = ContextAwareConsultAgent()
    service = DoctorConsultService(agent=agent)

    response = service.consult(
        case,
        ["Can you describe the pain?"],
        ["It feels like chest pressure and I feel sweaty."],
        [ecg],
        "I have an ECG back, what now?",
    )

    assert "Review ECG immediately" in response.recommended_next_steps
    assert "Trend troponin" in response.recommended_next_steps


def test_consult_identifies_missing_critical_information() -> None:
    agent = CapturingConsultAgent(
        consult_response(
            missing_information=["Exact symptom onset time", "Current oxygen saturation"],
        )
    )
    service = DoctorConsultService(agent=agent)

    response = service.consult(mi_case(), [], [], [], "Any priorities?")

    assert "Exact symptom onset time" in response.missing_information
    assert "Current oxygen saturation" in response.missing_information


def test_critical_vitals_produce_escalation_and_safety_warnings() -> None:
    agent = ContextAwareConsultAgent()
    service = DoctorConsultService(agent=agent)

    response = service.consult(critical_case(), [], [], [], "Can I wait on labs?")

    assert response.urgency_level == "critical"
    assert response.should_escalate is True
    assert response.safety_warnings
    assert "immediate escalation" in response.safety_warnings[0]


def test_web_doctor_consult_path_uses_service(monkeypatch: Any) -> None:
    from backend import web_app

    class FakeDoctorConsultService:
        def consult(
            self,
            case: ClinicalCase,
            student_questions: list[str],
            patient_responses: list[str],
            investigations_ordered: list[InvestigationResult],
            consult_question: str,
            *,
            session_context: dict[str, Any] | None = None,
        ) -> DoctorConsultResponse:
            assert case.diagnosis == "myocardial infarction"
            assert consult_question == "Should I order an ECG?"
            return consult_response(
                concise_advice="Yes, get the ECG now and keep him monitored.",
            )

    session_id = "doctor-consult-session"
    web_app.SESSIONS[session_id] = web_app.WebSession(case=mi_case())
    monkeypatch.setattr(web_app, "DOCTOR_CONSULT_SERVICE", FakeDoctorConsultService())

    result = web_app.call_doctor(
        web_app.DoctorConsultRequest(
            session_id=session_id,
            question="Should I order an ECG?",
        )
    )

    assert result["concise_advice"] == "Yes, get the ECG now and keep him monitored."
    assert web_app.SESSIONS[session_id].doctor_consults
