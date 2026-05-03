from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import ClinicalEvaluation
from schemas.investigation import InvestigationResult
from schemas.simulation_audit import SimulationAudit


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class StudentAction(StrictModel):
    """Next action selected by an AI student during simulation."""

    action: Literal[
        "ask_patient_question",
        "order_investigation",
        "submit_interpretation",
        "submit_diagnosis",
        "call_doctor",
        "finish",
    ]
    patient_question: str | None = None
    investigation: str | None = None
    interpretation: str | None = None
    diagnosis_guess: str | None = None
    consult_question: str | None = None

    @model_validator(mode="after")
    def require_action_payload(self) -> "StudentAction":
        required_field_by_action = {
            "ask_patient_question": "patient_question",
            "order_investigation": "investigation",
            "submit_interpretation": "interpretation",
            "submit_diagnosis": "diagnosis_guess",
            "call_doctor": "consult_question",
        }
        field_name = required_field_by_action.get(self.action)
        if field_name and not str(getattr(self, field_name) or "").strip():
            raise ValueError(f"{field_name} is required for {self.action}")
        return self


class SimulationStep(StrictModel):
    """One executed action in a simulated encounter."""

    step_number: int
    action: str
    request: str | None = None
    response: object | None = None


class SimulationResult(StrictModel):
    """Complete result from one simulated clinical encounter."""

    diagnosis: str
    difficulty: Literal["easy", "medium", "hard"]
    student_questions: list[str]
    patient_responses: list[str]
    investigations_ordered: list[InvestigationResult]
    student_interpretations: list[str]
    doctor_consults: list[DoctorConsultResponse]
    diagnosis_guess: str | None
    steps: list[SimulationStep]
    evaluation: ClinicalEvaluation
    audit: SimulationAudit | None = None


class SimulationSummary(StrictModel):
    """Aggregate summary across repeated simulated encounters."""

    runs: int
    average_score: float
    common_missed_symptoms: list[str] = Field(default_factory=list)
    common_reasoning_weaknesses: list[str] = Field(default_factory=list)
    diagnosis_accuracy_rate: float
    audit_summary: dict[str, object] | None = None
