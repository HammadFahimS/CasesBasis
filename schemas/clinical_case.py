from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class Patient(StrictModel):
    """Patient demographics and relevant history."""

    age: int
    gender: str
    name: str | None = None
    race: str | None = None
    occupation: str | None = None
    persona_description: str = ""
    history: list[str]

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if not 0 <= value <= 130:
            raise ValueError("age must be realistic")
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"male", "female"}:
            raise ValueError("gender must be male or female")
        return normalized

    @field_validator("history")
    @classmethod
    def require_history(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("history must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("history must not contain empty strings")
        return value


class Vitals(StrictModel):
    """Initial vital signs for the clinical case."""

    heart_rate: int
    blood_pressure: str
    respiratory_rate: int
    oxygen_saturation: int

    @field_validator("blood_pressure", mode="before")
    @classmethod
    def normalize_blood_pressure_field(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return normalize_blood_pressure(value)

    @model_validator(mode="after")
    def validate_ranges(self) -> "Vitals":
        if not 0 <= self.heart_rate <= 260:
            raise ValueError("heart_rate must be realistic")
        if not 0 <= self.respiratory_rate <= 80:
            raise ValueError("respiratory_rate must be realistic")
        if not 0 <= self.oxygen_saturation <= 100:
            raise ValueError("oxygen_saturation must be between 0 and 100")
        if not re.fullmatch(r"\d{2,3}/\d{2,3}", self.blood_pressure.strip()):
            raise ValueError("blood_pressure must use systolic/diastolic format")
        systolic, diastolic = [int(part) for part in self.blood_pressure.split("/")]
        if not 40 <= systolic <= 300 or not 20 <= diastolic <= 200:
            raise ValueError("blood_pressure values must be realistic")
        if systolic < diastolic:
            raise ValueError("systolic blood pressure must be >= diastolic blood pressure")
        return self


class ProgressionEvent(StrictModel):
    """One event in the case timeline."""

    time_min: int
    description: str

    @field_validator("time_min")
    @classmethod
    def validate_time_min(cls, value: int) -> int:
        if value < 0:
            raise ValueError("time_min must be non-negative")
        return value


class Outcome(StrictModel):
    """Expected case outcomes with and without treatment."""

    if_treated: str
    if_untreated: str


class ClinicalCase(StrictModel):
    """Source-of-truth schema for generated clinical cases."""

    diagnosis: str
    patient: Patient
    symptoms: list[str]
    vitals: Vitals
    progression: list[ProgressionEvent]
    risk_factors: list[str]
    exam_findings: dict[str, list[str]]
    outcome: Outcome

    @model_validator(mode="before")
    @classmethod
    def normalize_case_payload(cls, value: object) -> object:
        return normalize_case_data(value)

    @field_validator("symptoms", "risk_factors")
    @classmethod
    def require_non_empty_string_lists(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("list fields must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("list fields must not contain empty strings")
        return value

    @field_validator("exam_findings")
    @classmethod
    def validate_exam_findings(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if not value:
            raise ValueError("exam_findings must not be empty")
        for system, findings in value.items():
            if not system.strip():
                raise ValueError("exam_findings system keys must not be empty")
            if not findings:
                raise ValueError(f"exam_findings[{system}] must not be empty")
            if any(not f.strip() for f in findings):
                raise ValueError(f"exam_findings[{system}] must not contain empty strings")
        return value

    @property
    def exam_findings_flat(self) -> list[str]:
        return [f for findings in self.exam_findings.values() for f in findings]

    @field_validator("progression")
    @classmethod
    def require_progression(cls, value: list[ProgressionEvent]) -> list[ProgressionEvent]:
        if not value:
            raise ValueError("progression must not be empty")
        return value

    @model_validator(mode="after")
    def validate_case(self) -> "ClinicalCase":
        if not self.diagnosis.strip():
            raise ValueError("diagnosis is required")
        if len(self.symptoms) < 3 or len(self.symptoms) > 6:
            raise ValueError("symptoms must include 3 to 6 items")
        return self


def normalize_blood_pressure(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    normalized = re.sub(r"\bbp\b[:\s-]*", "", normalized)
    normalized = re.sub(r"\bmm\s*hg\b|\bmmhg\b", "", normalized)
    normalized = re.sub(r"\bover\b", "/", normalized)
    match = re.search(r"(?<!\d)(\d{2,3})\s*(?:/|-)\s*(\d{2,3})(?!\d)", normalized)
    if not match:
        return value
    return f"{int(match.group(1))}/{int(match.group(2))}"


def normalize_case_data(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = clean_nested_strings(value)
    if not isinstance(normalized, dict):
        return normalized

    patient = normalized.get("patient")
    if isinstance(patient, dict):
        normalized_patient = dict(patient)
        normalized_patient["age"] = normalize_int_value(normalized_patient.get("age"))
        normalized["patient"] = normalized_patient

    vitals = normalized.get("vitals")
    if isinstance(vitals, dict):
        normalized_vitals = dict(vitals)
        normalized_vitals["heart_rate"] = normalize_int_value(normalized_vitals.get("heart_rate"))
        normalized_vitals["respiratory_rate"] = normalize_int_value(
            normalized_vitals.get("respiratory_rate")
        )
        normalized_vitals["oxygen_saturation"] = normalize_int_value(
            normalized_vitals.get("oxygen_saturation")
        )
        normalized_vitals["blood_pressure"] = normalize_blood_pressure(
            normalized_vitals.get("blood_pressure")
        )
        normalized["vitals"] = normalized_vitals

    exam_findings = normalized.get("exam_findings")
    if isinstance(exam_findings, list):
        normalized["exam_findings"] = {"General": exam_findings}

    progression = normalized.get("progression")
    if isinstance(progression, list):
        normalized_progression: list[object] = []
        for event in progression:
            if isinstance(event, dict):
                normalized_event = dict(event)
                normalized_event["time_min"] = normalize_int_value(normalized_event.get("time_min"))
                normalized_progression.append(normalized_event)
            else:
                normalized_progression.append(event)
        normalized["progression"] = normalized_progression

    return normalized


def clean_nested_strings(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return [clean_nested_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            key: clean_nested_strings(item)
            for key, item in value.items()
        }
    return value


def normalize_int_value(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", normalized)
    if not match:
        return value
    return int(match.group(1))
