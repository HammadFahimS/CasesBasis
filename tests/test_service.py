from __future__ import annotations

from agents.generator import GenerationError
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
    def __init__(self) -> None:
        self.requested_diagnosis: str | None = None
        self.requested_difficulty: str | None = None
        self.requested_session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None
        self.corrected_case: ClinicalCase | None = None

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
        return ClinicalCase.model_validate(sample_case())


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
