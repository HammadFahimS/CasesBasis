from __future__ import annotations

import pytest

from agents.investigation_generator import (
    GenericInvestigationAgent,
    InvestigationGenerationError,
)
from backend.investigations import (
    InvestigationService,
    is_ecg_request,
    is_nonsensical_investigation_request,
    is_troponin_request,
    split_investigation_requests,
)
from schemas.clinical_case import ClinicalCase
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
            "exam_findings": ["diaphoretic", "anxious"],
            "outcome": {
                "if_treated": "Rapid emergency treatment improves outcomes.",
                "if_untreated": "Ongoing ischemia can lead to instability.",
            },
        }
    )


def non_cardiac_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "ankle sprain",
            "patient": {
                "age": 24,
                "gender": "female",
                "history": ["no chronic medical problems"],
            },
            "symptoms": [
                "ankle pain",
                "ankle swelling",
                "difficulty bearing weight",
            ],
            "vitals": {
                "heart_rate": 82,
                "blood_pressure": "118/72",
                "respiratory_rate": 14,
                "oxygen_saturation": 99,
            },
            "progression": [
                {
                    "time_min": 0,
                    "description": "Pain begins after twisting the ankle while running.",
                },
                {
                    "time_min": 20,
                    "description": "Swelling increases but there are no systemic symptoms.",
                },
            ],
            "risk_factors": ["recent sports activity"],
            "exam_findings": ["lateral ankle tenderness", "mild swelling"],
            "outcome": {
                "if_treated": "Pain improves with rest, ice, compression, and elevation.",
                "if_untreated": "Pain and swelling may persist longer.",
            },
        }
    )


def pneumonia_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "community-acquired pneumonia",
            "patient": {
                "age": 67,
                "gender": "female",
                "history": ["COPD", "former smoker"],
            },
            "symptoms": [
                "productive cough",
                "fever",
                "shortness of breath",
                "fatigue",
            ],
            "vitals": {
                "heart_rate": 108,
                "blood_pressure": "126/78",
                "respiratory_rate": 24,
                "oxygen_saturation": 92,
            },
            "progression": [
                {
                    "time_min": 0,
                    "description": "Cough and fatigue have been worsening for two days.",
                },
                {
                    "time_min": 60,
                    "description": "Breathing feels more labored with fever and chills.",
                },
            ],
            "risk_factors": ["COPD", "older age"],
            "exam_findings": ["crackles at the right lung base", "mild increased work of breathing"],
            "outcome": {
                "if_treated": "Respiratory symptoms improve with appropriate care.",
                "if_untreated": "Breathing may worsen and oxygen levels may fall.",
            },
        }
    )


def pulmonary_embolism_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "pulmonary embolism",
            "patient": {
                "age": 46,
                "gender": "female",
                "history": ["recent surgery", "limited mobility"],
            },
            "symptoms": [
                "episodic chest pain",
                "shortness of breath",
                "left calf pain",
            ],
            "vitals": {
                "heart_rate": 112,
                "blood_pressure": "128/78",
                "respiratory_rate": 24,
                "oxygen_saturation": 92,
            },
            "progression": [
                {
                    "time_min": 0,
                    "description": "Sharp chest discomfort begins with exertion.",
                },
                {
                    "time_min": 30,
                    "description": "Breathing feels more difficult while walking.",
                },
            ],
            "risk_factors": ["recent surgery", "immobility"],
            "exam_findings": ["tachypnea", "left calf tenderness"],
            "outcome": {
                "if_treated": "Anticoagulation and supportive care reduce risk.",
                "if_untreated": "Clot burden may worsen and cause instability.",
            },
        }
    )


class FakeGenericInvestigationGenerator:
    def __init__(self, result: InvestigationResult) -> None:
        self.result = result
        self.calls: list[tuple[ClinicalCase, str]] = []

    def generate(self, case: ClinicalCase, requested_investigation: str) -> InvestigationResult:
        self.calls.append((case, requested_investigation))
        return self.result


class FailingGenericInvestigationGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[ClinicalCase, str]] = []

    def generate(self, case: ClinicalCase, requested_investigation: str) -> InvestigationResult:
        self.calls.append((case, requested_investigation))
        raise InvestigationGenerationError("failed after retries")


def generic_result(
    *,
    investigation: str,
    findings: list[str],
    values: list[dict[str, object]],
    summary: str,
    interpretation_hint: str,
) -> InvestigationResult:
    return InvestigationResult(
        investigation=investigation,
        status="available",
        structured_result={
            "test": investigation,
            "findings": findings,
            "values": values,
            "interpretation_hint": interpretation_hint,
        },
        summary=summary,
        visual_payload=None,
    )


def test_ecg_request_detection() -> None:
    assert is_ecg_request("EKG")
    assert is_ecg_request("ECG")
    assert is_ecg_request("do an EKG please")
    assert is_ecg_request("electrocardiogram")
    assert not is_ecg_request("chest x-ray")


def test_troponin_request_detection() -> None:
    assert is_troponin_request("troponin")
    assert is_troponin_request("cardiac enzymes")
    assert is_troponin_request("blood test for heart attack")
    assert not is_troponin_request("complete blood count")


def test_nonsensical_investigation_detection() -> None:
    assert is_nonsensical_investigation_request("")
    assert is_nonsensical_investigation_request("crystal ball scan")
    assert is_nonsensical_investigation_request("xqzprtlsdfghjkl")
    assert not is_nonsensical_investigation_request("CT abdomen")
    assert not is_nonsensical_investigation_request("complete blood count")


def test_split_investigation_requests_handles_and_and_commas() -> None:
    assert split_investigation_requests("ECG and troponin") == ["ECG", "troponin"]
    assert split_investigation_requests("CBC, electrolytes, glucose") == [
        "CBC",
        "electrolytes",
        "glucose",
    ]


def test_multiple_investigation_request_runs_each_requested_test() -> None:
    fake = FakeGenericInvestigationGenerator(
        generic_result(
            investigation="generic",
            findings=["No acute abnormality"],
            values=[],
            summary="Result completed.",
            interpretation_hint="No actionable abnormality.",
        )
    )

    results = InvestigationService(generic_generator=fake).get_results(
        mi_case(),
        "CBC, electrolytes, glucose",
    )

    assert len(results) == 3
    assert [call[1] for call in fake.calls] == ["CBC", "electrolytes", "glucose"]


def test_mixed_unknown_and_known_multiple_investigation_request() -> None:
    results = InvestigationService().get_results(mi_case(), "crystal ball scan and ECG")

    assert len(results) == 2
    assert results[0].investigation == "crystal ball scan"
    assert results[0].status == "unavailable"
    assert results[1].investigation == "ECG"
    assert results[1].status == "available"


def test_ecg_result_for_mi_case_has_plausible_findings() -> None:
    result = InvestigationService().get_result(mi_case(), "do an EKG")

    assert result.investigation == "ECG"
    assert result.status == "available"
    findings = " ".join(result.structured_result["findings"])
    assert "ST elevation" in findings
    assert "reciprocal ST depression" in findings
    assert result.structured_result["st_segments"]["II"] == "elevated"
    assert result.structured_result["st_segments"]["aVL"] == "depressed"


def test_ecg_result_for_pe_chest_pain_does_not_show_mi_pattern() -> None:
    result = InvestigationService().get_result(pulmonary_embolism_case(), "ECG")

    assert result.investigation == "ECG"
    assert result.status == "available"
    findings = " ".join(result.structured_result["findings"]).lower()
    assert "st elevation in contiguous" not in findings
    assert "reciprocal st depression in leads" not in findings
    assert result.structured_result["st_segments"] == {}


def test_ecg_uses_deterministic_visual_path_without_generic_generator() -> None:
    fake = FakeGenericInvestigationGenerator(
        generic_result(
            investigation="unexpected",
            findings=[],
            values=[],
            summary="unexpected",
            interpretation_hint="unexpected",
        )
    )

    result = InvestigationService(generic_generator=fake).get_result(mi_case(), "EKG")

    assert result.visual_payload is not None
    assert result.visual_payload.type == "svg"
    assert fake.calls == []


def test_ecg_result_does_not_reveal_diagnosis_directly() -> None:
    result = InvestigationService().get_result(mi_case(), "ECG")
    payload = result.model_dump_json().lower()

    assert "myocardial infarction" not in payload


def test_nonsensical_investigation_returns_safe_unavailable_response() -> None:
    result = InvestigationService().get_result(mi_case(), "crystal ball scan")

    assert result.status == "unavailable"
    assert result.summary == "That test is not available yet"
    assert result.visual_payload is None


def test_relevant_blood_test_returns_plausible_abnormal_result() -> None:
    fake = FakeGenericInvestigationGenerator(
        generic_result(
            investigation="Complete blood count",
            findings=["Leukocytosis with neutrophil predominance"],
            values=[
                {
                    "name": "white blood cell count",
                    "value": 15.8,
                    "unit": "10^9/L",
                    "reference_range": "4.0-11.0 10^9/L",
                    "flag": "high",
                }
            ],
            summary="CBC shows an elevated white blood cell count with neutrophil predominance.",
            interpretation_hint="Inflammatory or infectious process is suggested in this clinical context.",
        )
    )

    result = InvestigationService(generic_generator=fake).get_result(
        pneumonia_case(),
        "complete blood count",
    )

    assert fake.calls
    assert result.status == "available"
    assert result.structured_result["values"][0]["flag"] == "high"
    assert "elevated white blood cell" in result.summary


def test_unrelated_reasonable_test_returns_normal_non_contributory_result() -> None:
    fake = FakeGenericInvestigationGenerator(
        generic_result(
            investigation="CT abdomen",
            findings=["No acute abnormality identified"],
            values=[],
            summary="CT abdomen is non-contributory for the current presentation.",
            interpretation_hint="No abdominal finding explains the reported symptoms.",
        )
    )

    result = InvestigationService(generic_generator=fake).get_result(
        mi_case(),
        "CT abdomen",
    )

    assert fake.calls
    assert result.status == "available"
    assert "non-contributory" in result.summary
    assert result.visual_payload is None


def test_generic_investigation_generation_failure_returns_safe_unavailable_result() -> None:
    fake = FailingGenericInvestigationGenerator()

    result = InvestigationService(generic_generator=fake).get_result(
        pneumonia_case(),
        "chest x-ray",
    )

    assert fake.calls
    assert result.investigation == "chest x-ray"
    assert result.status == "unavailable"
    assert result.summary == "That test is not available yet"
    assert "pneumonia" not in result.model_dump_json().lower()


def test_troponin_result_is_elevated_for_mi_like_case() -> None:
    result = InvestigationService().get_result(mi_case(), "cardiac enzymes")

    assert result.investigation == "Troponin"
    assert result.status == "available"
    assert result.structured_result["test"] == "high-sensitivity troponin I"
    assert result.structured_result["value"] > 0.04
    assert result.structured_result["reference_range"] == "<0.04 ng/mL"
    assert result.structured_result["flag"] == "high"
    assert result.visual_payload is None


def test_troponin_result_does_not_reveal_diagnosis_directly() -> None:
    result = InvestigationService().get_result(mi_case(), "blood test for heart attack")

    assert "myocardial infarction" not in result.model_dump_json().lower()


def test_generic_investigation_result_does_not_reveal_diagnosis_directly() -> None:
    fake = FakeGenericInvestigationGenerator(
        generic_result(
            investigation="Chest X-ray",
            findings=["Patchy right lower lung airspace opacity"],
            values=[],
            summary="Chest X-ray shows a focal right lower lung opacity.",
            interpretation_hint="Correlate with cough, fever, and oxygen saturation.",
        )
    )

    result = InvestigationService(generic_generator=fake).get_result(
        pneumonia_case(),
        "chest x-ray",
    )

    assert "community-acquired pneumonia" not in result.model_dump_json().lower()


class RevealingThenSafeInvestigationAgent(GenericInvestigationAgent):
    def __init__(self) -> None:
        self.calls = 0

    def parse_response(self, **_: object) -> InvestigationResult:
        self.calls += 1
        if self.calls == 1:
            return generic_result(
                investigation="Chest X-ray",
                findings=["Finding consistent with community-acquired pneumonia"],
                values=[],
                summary="This directly says community-acquired pneumonia.",
                interpretation_hint="Direct diagnosis reveal.",
            )
        return generic_result(
            investigation="Chest X-ray",
            findings=["Patchy right lower lung airspace opacity"],
            values=[],
            summary="Chest X-ray shows a focal right lower lung opacity.",
            interpretation_hint="Correlate with respiratory symptoms and oxygen saturation.",
        )


def test_generic_investigation_retries_when_result_reveals_diagnosis() -> None:
    agent = RevealingThenSafeInvestigationAgent()

    result = agent.generate(pneumonia_case(), "chest x-ray")

    assert agent.calls == 2
    assert "community-acquired pneumonia" not in result.model_dump_json().lower()


class AlwaysInvalidInvestigationAgent(GenericInvestigationAgent):
    def __init__(self) -> None:
        self.calls = 0

    def parse_response(self, **_: object) -> InvestigationResult:
        self.calls += 1
        raise ValueError("invalid JSON")


def test_generic_investigation_fails_after_two_invalid_json_attempts() -> None:
    agent = AlwaysInvalidInvestigationAgent()

    with pytest.raises(InvestigationGenerationError):
        agent.generate(pneumonia_case(), "chest x-ray")

    assert agent.calls == 2


def test_troponin_result_is_normal_for_non_cardiac_case() -> None:
    result = InvestigationService().get_result(non_cardiac_case(), "troponin")

    assert result.status == "available"
    assert result.structured_result["value"] == 0.01
    assert result.structured_result["flag"] == "normal"
    assert "within the reference range" in result.structured_result["interpretation_hint"]


def test_ecg_visual_payload_contains_svg_and_all_leads() -> None:
    result = InvestigationService().get_result(mi_case(), "EKG")

    assert result.visual_payload is not None
    assert result.visual_payload.type == "svg"
    assert result.visual_payload.svg.startswith("<svg")
    assert set(result.visual_payload.leads) == {
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    }
    for lead in result.visual_payload.leads:
        assert f">{lead}<" in result.visual_payload.svg
