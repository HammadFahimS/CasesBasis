from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.clinical_case import ClinicalCase, normalize_blood_pressure, normalize_case_data


def sample_case() -> dict[str, object]:
    return {
        "diagnosis": "septic shock",
        "patient": {
            "age": 72,
            "gender": "female",
            "history": ["type 2 diabetes", "recurrent urinary tract infections"],
        },
        "symptoms": [
            "fever",
            "confusion",
            "dizziness",
            "shortness of breath",
            "decreased urine output",
        ],
        "vitals": {
            "heart_rate": 132,
            "blood_pressure": "82/48",
            "respiratory_rate": 28,
            "oxygen_saturation": 91,
        },
        "progression": [
            {
                "time_min": 0,
                "description": "Patient arrives febrile, tachycardic, hypotensive, and confused.",
            },
            {
                "time_min": 30,
                "description": "Without rapid fluids and antibiotics, hypotension and mental status worsen.",
            },
        ],
        "risk_factors": ["diabetes", "recent urinary symptoms"],
        "exam_findings": ["cool extremities", "delayed capillary refill", "suprapubic tenderness"],
        "outcome": {
            "if_treated": "Improves with early broad-spectrum antibiotics, fluids, and vasopressors if needed.",
            "if_untreated": "Progresses to multiorgan failure and high risk of death.",
        },
    }


def test_clinical_case_accepts_valid_case() -> None:
    clinical_case = ClinicalCase.model_validate(sample_case())

    assert clinical_case.diagnosis == "septic shock"
    assert clinical_case.patient.gender == "female"


def test_clinical_case_rejects_extra_fields() -> None:
    payload = sample_case()
    payload["unplanned_field"] = "not allowed"

    with pytest.raises(ValidationError):
        ClinicalCase.model_validate(payload)


def test_clinical_case_rejects_impossible_vitals() -> None:
    payload = sample_case()
    payload["vitals"]["oxygen_saturation"] = 130

    with pytest.raises(ValidationError):
        ClinicalCase.model_validate(payload)


def test_clinical_case_requires_three_to_six_symptoms() -> None:
    payload = sample_case()
    payload["symptoms"] = ["fever", "confusion"]

    with pytest.raises(ValidationError):
        ClinicalCase.model_validate(payload)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("120 over 80", "120/80"),
        ("120-80", "120/80"),
        ("BP 120/80", "120/80"),
        ("120 / 80 mmHg", "120/80"),
    ],
)
def test_normalize_blood_pressure_common_local_llm_formats(
    raw_value: str,
    expected: str,
) -> None:
    assert normalize_blood_pressure(raw_value) == expected

    payload = sample_case()
    payload["vitals"]["blood_pressure"] = raw_value

    clinical_case = ClinicalCase.model_validate(payload)

    assert clinical_case.vitals.blood_pressure == expected


def test_normalize_blood_pressure_leaves_unrecognized_format_for_validation_failure() -> None:
    payload = sample_case()
    payload["vitals"]["blood_pressure"] = "one twenty over eighty"

    with pytest.raises(ValidationError):
        ClinicalCase.model_validate(payload)


def test_normalize_case_data_coerces_numeric_strings_and_trims_text() -> None:
    payload = sample_case()
    payload["diagnosis"] = "  septic shock  "
    payload["patient"]["age"] = "72 years"
    payload["patient"]["history"] = ["  type 2 diabetes  "]
    payload["symptoms"] = [" fever ", " confusion ", " dizziness "]
    payload["vitals"] = {
        "heart_rate": "132 bpm",
        "blood_pressure": "120 / 80 mmHg",
        "respiratory_rate": "28 breaths/min",
        "oxygen_saturation": "91%",
    }
    payload["progression"][0]["time_min"] = "0 min"

    normalized = normalize_case_data(payload)
    clinical_case = ClinicalCase.model_validate(normalized)

    assert clinical_case.diagnosis == "septic shock"
    assert clinical_case.patient.age == 72
    assert clinical_case.patient.history == ["type 2 diabetes"]
    assert clinical_case.symptoms == ["fever", "confusion", "dizziness"]
    assert clinical_case.vitals.heart_rate == 132
    assert clinical_case.vitals.blood_pressure == "120/80"
    assert clinical_case.vitals.respiratory_rate == 28
    assert clinical_case.vitals.oxygen_saturation == 91
    assert clinical_case.progression[0].time_min == 0
