from __future__ import annotations

import pytest

import backend.medical_validation as medical_validation
from backend.medical_validation import MedicalValidationError, validate_medical_realism
from schemas.clinical_case import ClinicalCase
from tests.test_clinical_case_schema import sample_case


def build_case(overrides: dict[str, object] | None = None) -> ClinicalCase:
    payload = sample_case()
    if overrides:
        payload.update(overrides)
    return ClinicalCase.model_validate(payload)


def test_medical_validation_accepts_realistic_case() -> None:
    validate_medical_realism(build_case())


def test_medical_validation_does_not_use_disease_specific_symptom_profiles() -> None:
    assert not hasattr(medical_validation, "SYMPTOM_PROFILES")


def test_medical_validation_rejects_mismatched_body_system_symptoms() -> None:
    clinical_case = build_case(
        {
            "diagnosis": "pneumonia",
            "symptoms": ["headache", "photophobia", "neck stiffness"],
            "exam_findings": {"Respiratory": ["normal breath sounds"]},
            "progression": [{"time_min": 0, "description": "Neurologic symptoms worsen."}],
        }
    )

    with pytest.raises(MedicalValidationError, match="symptoms"):
        validate_medical_realism(clinical_case)


def test_medical_validation_rejects_unreasonable_vitals() -> None:
    clinical_case = build_case(
        {
            "vitals": {
                "heart_rate": 28,
                "blood_pressure": "82/48",
                "respiratory_rate": 28,
                "oxygen_saturation": 91,
            }
        }
    )

    with pytest.raises(MedicalValidationError, match="heart_rate"):
        validate_medical_realism(clinical_case)


def test_medical_validation_rejects_hypotension_with_low_hr_without_justification() -> None:
    clinical_case = build_case(
        {
            "vitals": {
                "heart_rate": 55,
                "blood_pressure": "82/48",
                "respiratory_rate": 28,
                "oxygen_saturation": 91,
            }
        }
    )

    with pytest.raises(MedicalValidationError, match="hypotension"):
        validate_medical_realism(clinical_case)


def test_medical_validation_rejects_unjustified_unrelated_symptom_systems() -> None:
    clinical_case = build_case(
        {
            "diagnosis": "acute coronary syndrome",
            "symptoms": ["chest pressure", "shortness of breath", "itchy rash"],
            "vitals": {
                "heart_rate": 110,
                "blood_pressure": "132/84",
                "respiratory_rate": 20,
                "oxygen_saturation": 96,
            },
            "exam_findings": {"General": ["diaphoretic"], "Dermatological": ["normal skin exam"]},
            "progression": [
                {"time_min": 0, "description": "Chest pressure begins with exertion."},
                {"time_min": 20, "description": "Chest pressure worsens and patient becomes sweaty."},
            ],
        }
    )

    with pytest.raises(MedicalValidationError, match="unrelated symptom systems"):
        validate_medical_realism(clinical_case)


def test_medical_validation_rejects_stated_severity_with_normal_vitals() -> None:
    clinical_case = build_case(
        {
            "diagnosis": "severe respiratory illness",
            "symptoms": ["cough", "shortness of breath", "fatigue"],
            "vitals": {
                "heart_rate": 76,
                "blood_pressure": "122/78",
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
            },
            "progression": [
                {"time_min": 0, "description": "Severe symptoms are reported but patient appears stable."},
                {"time_min": 30, "description": "Symptoms remain severe without objective change."},
            ],
        }
    )

    with pytest.raises(MedicalValidationError, match="vital signs"):
        validate_medical_realism(clinical_case)


def test_medical_validation_rejects_incoherent_progression_times() -> None:
    clinical_case = build_case(
        {
            "progression": [
                {"time_min": 30, "description": "Patient becomes more confused."},
                {"time_min": 0, "description": "Patient arrives dizzy and febrile."},
            ]
        }
    )

    with pytest.raises(MedicalValidationError, match="progression times"):
        validate_medical_realism(clinical_case)
