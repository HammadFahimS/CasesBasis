from __future__ import annotations

from agents.generator import GenerationError
from backend.medical_validation import MedicalValidationError
from backend.service import ClinicalCaseSimulator
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation
from tests.test_clinical_case_schema import sample_case


class StubGenerator:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        cases: list[ClinicalCase] | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.cases = cases or []
        self.requested_diagnosis: str | None = None
        self.requested_difficulty: str | None = None
        self.requested_session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None
        self.calls = 0

    def generate(
        self,
        diagnosis: str,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        self.calls += 1
        self.requested_diagnosis = diagnosis
        self.requested_difficulty = difficulty
        self.requested_session_history = session_history
        if self.should_fail:
            raise GenerationError("case generation failed after 2 attempts")
        if self.cases:
            return self.cases.pop(0)
        return ClinicalCase.model_validate(sample_case())


class StubValidator:
    def __init__(
        self,
        *,
        corrected_case: ClinicalCase | None = None,
        strict_case: ClinicalCase | None = None,
    ) -> None:
        self.requested_diagnosis: str | None = None
        self.requested_difficulty: str | None = None
        self.requested_session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None
        self.corrected_case: ClinicalCase | None = None
        self.corrected_response = corrected_case
        self.strict_response = strict_case
        self.strict_requested_diagnosis: str | None = None
        self.strict_requested_difficulty: str | None = None
        self.strict_requested_session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None
        self.strict_failure_reason: str | None = None

    def recover(
        self,
        diagnosis: str,
        failure_reason: str | None = None,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        self.requested_diagnosis = diagnosis
        self.requested_difficulty = difficulty
        self.requested_session_history = session_history
        return ClinicalCase.model_validate(sample_case())

    def correct(self, case: ClinicalCase, failure_reason: str | None = None) -> ClinicalCase:
        self.corrected_case = case
        return self.corrected_response or ClinicalCase.model_validate(sample_case())

    def regenerate_strict(
        self,
        diagnosis: str,
        failure_reason: str,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        self.strict_requested_diagnosis = diagnosis
        self.strict_requested_difficulty = difficulty
        self.strict_requested_session_history = session_history
        self.strict_failure_reason = failure_reason
        return self.strict_response or ClinicalCase.model_validate(sample_case())


def respiratory_case_payload(diagnosis: str, symptoms: list[str]) -> dict[str, object]:
    return {
        "diagnosis": diagnosis,
        "patient": {
            "age": 42,
            "gender": "female",
            "history": ["seasonal allergies"],
        },
        "symptoms": symptoms,
        "vitals": {
            "heart_rate": 108,
            "blood_pressure": "122/76",
            "respiratory_rate": 24,
            "oxygen_saturation": 93,
        },
        "progression": [
            {
                "time_min": 0,
                "description": "Breathing symptoms begin gradually.",
            },
            {
                "time_min": 30,
                "description": "Respiratory symptoms worsen with exertion.",
            },
        ],
        "risk_factors": ["recent respiratory infection"],
        "exam_findings": {"Respiratory": ["increased work of breathing"]},
        "outcome": {
            "if_treated": "Symptoms improve with appropriate care.",
            "if_untreated": "Respiratory distress may worsen.",
        },
    }


def invalid_respiratory_case(diagnosis: str) -> ClinicalCase:
    return ClinicalCase.model_validate(
        respiratory_case_payload(
            diagnosis,
            ["cough", "shortness of breath", "itchy rash"],
        )
    )


def valid_respiratory_case(diagnosis: str, symptoms: list[str]) -> ClinicalCase:
    return ClinicalCase.model_validate(respiratory_case_payload(diagnosis, symptoms))


def test_simulator_uses_generator_only_on_success() -> None:
    generator = StubGenerator()
    validator = StubValidator()
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)

    session_history: list[ClinicalEvaluation | dict[str, object] | str] = []
    clinical_case = simulator.generate_case(
        "septic shock",
        difficulty="hard",
        session_history=session_history,
    )

    assert generator.requested_diagnosis == "septic shock"
    assert generator.requested_difficulty == "hard"
    assert generator.requested_session_history is session_history
    assert generator.calls == 1
    assert validator.requested_diagnosis is None
    assert validator.corrected_case is None
    assert clinical_case.diagnosis == "septic shock"


def test_simulator_uses_validator_only_after_generation_failure() -> None:
    generator = StubGenerator(should_fail=True)
    validator = StubValidator()
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)

    session_history: list[ClinicalEvaluation | dict[str, object] | str] = []
    clinical_case = simulator.generate_case(
        "septic shock",
        difficulty="easy",
        session_history=session_history,
    )

    assert generator.requested_diagnosis == "septic shock"
    assert validator.requested_diagnosis == "septic shock"
    assert validator.requested_difficulty == "easy"
    assert validator.requested_session_history is session_history
    assert clinical_case.diagnosis == "septic shock"


def test_simulator_corrects_case_after_medical_validation_failure() -> None:
    invalid_payload = sample_case()
    invalid_payload["vitals"] = {
        "heart_rate": 55,
        "blood_pressure": "82/48",
        "respiratory_rate": 28,
        "oxygen_saturation": 91,
    }
    generator = StubGenerator(
        cases=[
            ClinicalCase.model_validate(invalid_payload),
        ]
    )
    validator = StubValidator()
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)

    clinical_case = simulator.generate_case("septic shock")

    assert generator.calls == 1
    assert validator.requested_diagnosis is None
    assert validator.corrected_case is not None
    assert validator.corrected_case.vitals.heart_rate == 55
    assert clinical_case.vitals.heart_rate == 132


def test_simulator_strictly_regenerates_pneumonia_after_failed_correction() -> None:
    invalid_case = invalid_respiratory_case("pneumonia")
    strict_case = valid_respiratory_case(
        "pneumonia",
        ["productive cough", "fever", "shortness of breath"],
    )
    generator = StubGenerator(cases=[invalid_case])
    validator = StubValidator(corrected_case=invalid_case, strict_case=strict_case)
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)
    session_history: list[ClinicalEvaluation | dict[str, object] | str] = []

    clinical_case = simulator.generate_case(
        "pneumonia",
        difficulty="hard",
        session_history=session_history,
    )

    assert clinical_case is strict_case
    assert validator.corrected_case is invalid_case
    assert validator.strict_requested_diagnosis == "pneumonia"
    assert validator.strict_requested_difficulty == "hard"
    assert validator.strict_requested_session_history is session_history
    assert "unrelated symptom systems" in str(validator.strict_failure_reason)


def test_simulator_strictly_regenerates_asthma_after_failed_correction() -> None:
    invalid_case = invalid_respiratory_case("asthma exacerbation")
    strict_case = valid_respiratory_case(
        "asthma exacerbation",
        ["wheezing", "shortness of breath", "cough"],
    )
    generator = StubGenerator(cases=[invalid_case])
    validator = StubValidator(corrected_case=invalid_case, strict_case=strict_case)
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)

    clinical_case = simulator.generate_case("asthma exacerbation", difficulty="medium")

    assert clinical_case is strict_case
    assert validator.strict_requested_diagnosis == "asthma exacerbation"
    assert validator.strict_requested_difficulty == "medium"
    assert "unrelated symptom systems" in str(validator.strict_failure_reason)


def test_simulator_returns_clean_validation_error_when_strict_retry_fails() -> None:
    invalid_case = invalid_respiratory_case("pneumonia")
    generator = StubGenerator(cases=[invalid_case])
    validator = StubValidator(corrected_case=invalid_case, strict_case=invalid_case)
    simulator = ClinicalCaseSimulator(generator=generator, validator=validator)

    try:
        simulator.generate_case("pneumonia")
    except MedicalValidationError as exc:
        assert "unrelated symptom systems" in str(exc)
    else:
        raise AssertionError("Expected MedicalValidationError")

    assert validator.strict_requested_diagnosis == "pneumonia"
