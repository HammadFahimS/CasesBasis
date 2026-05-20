from __future__ import annotations

import re

from schemas.clinical_case import ClinicalCase


class MedicalValidationError(ValueError):
    """Raised when deterministic medical realism checks fail."""

    pass


BODY_SYSTEM_KEYWORDS = {
    "cardiac": {
        "cardiac",
        "heart",
        "coronary",
        "chest",
        "pressure",
        "palpitation",
        "syncope",
        "arrhythmia",
    },
    "respiratory": {
        "respiratory",
        "lung",
        "pulmonary",
        "pneumon",
        "asthma",
        "copd",
        "cough",
        "wheeze",
        "dyspnea",
        "breath",
        "hypoxia",
    },
    "neurologic": {
        "neuro",
        "brain",
        "stroke",
        "seizure",
        "headache",
        "confusion",
        "weakness",
        "numbness",
        "speech",
        "dizzy",
        "dizziness",
    },
    "gastrointestinal": {
        "abdominal",
        "abdomen",
        "gastric",
        "intestinal",
        "bowel",
        "liver",
        "pancrea",
        "append",
        "nausea",
        "vomit",
        "diarrhea",
    },
    "genitourinary": {
        "urinary",
        "urine",
        "renal",
        "kidney",
        "bladder",
        "flank",
        "dysuria",
        "pelvic",
    },
    "dermatologic": {
        "skin",
        "rash",
        "hives",
        "itch",
        "lesion",
        "blister",
    },
    "musculoskeletal": {
        "joint",
        "muscle",
        "bone",
        "back",
        "limb",
        "swelling",
        "trauma",
        "fracture",
    },
    "systemic": {
        "sepsis",
        "septic",
        "shock",
        "infection",
        "fever",
        "chills",
        "fatigue",
        "malaise",
        "weakness",
        "sweat",
        "dehydration",
        # non-specific symptoms that appear across many system diseases
        "nausea",
        "vomit",
        "diaphoresis",
        "pallor",
        "anorexia",
        "anxiety",
        "restless",
        "pain",
    },
}

GENERAL_SYMPTOM_SYSTEMS = {"systemic"}
SEVERE_TERMS = {
    "shock",
    "severe",
    "critical",
    "unstable",
    "hypotension",
    "hypotensive",
    "respiratory distress",
    "altered mental",
    "failure",
}
MILD_TERMS = {"mild", "early", "stable", "minor", "gradual"}
LOW_HR_JUSTIFICATION_TERMS = {"bradycardia", "heart block", "beta blocker", "neurogenic", "spinal"}


def validate_medical_realism(case: ClinicalCase) -> None:
    issues = [
        *check_symptoms_match_diagnosis(case),
        *check_reasonable_vitals(case),
        *check_vitals_reflect_severity(case),
        *check_contradictions(case),
        *check_progression_coherence(case),
    ]
    if issues:
        raise MedicalValidationError("; ".join(issues))


def check_symptoms_match_diagnosis(case: ClinicalCase) -> list[str]:
    diagnosis_systems = classify_text(case.diagnosis)
    if not diagnosis_systems or "systemic" in diagnosis_systems:
        return []

    issues: list[str] = []
    justified_text = non_symptom_case_text(case)
    related_count = 0
    unrelated_symptoms: list[str] = []

    for symptom in case.symptoms:
        symptom_systems = classify_text(symptom)
        if not symptom_systems:
            continue
        if symptom_systems.intersection(diagnosis_systems | GENERAL_SYMPTOM_SYSTEMS):
            related_count += 1
            continue
        if symptom_is_justified(symptom, justified_text):
            continue
        unrelated_symptoms.append(symptom)

    if related_count == 0:
        issues.append("symptoms do not logically relate to the diagnosis body system")
    if len(unrelated_symptoms) >= 1:
        issues.append("unrelated symptom systems are present without justification")
    return issues


def check_reasonable_vitals(case: ClinicalCase) -> list[str]:
    issues: list[str] = []
    vitals = case.vitals
    systolic, diastolic = parse_blood_pressure(vitals.blood_pressure)

    if not 30 <= vitals.heart_rate <= 220:
        issues.append("heart_rate is outside reasonable clinical range")
    if not 5 <= vitals.respiratory_rate <= 60:
        issues.append("respiratory_rate is outside reasonable clinical range")
    if not 50 <= vitals.oxygen_saturation <= 100:
        issues.append("oxygen_saturation is outside reasonable clinical range")
    if not 50 <= systolic <= 260 or not 30 <= diastolic <= 160:
        issues.append("blood_pressure is outside reasonable clinical range")
    return issues


def check_vitals_reflect_severity(case: ClinicalCase) -> list[str]:
    issues: list[str] = []
    vitals = case.vitals
    systolic, _ = parse_blood_pressure(vitals.blood_pressure)
    case_text = all_case_text(case)

    has_abnormal_vital = (
        vitals.heart_rate > 100
        or vitals.heart_rate < 50
        or systolic < 100
        or vitals.respiratory_rate > 22
        or vitals.oxygen_saturation < 94
    )
    has_extreme_vital = (
        vitals.heart_rate > 140
        or systolic < 80
        or vitals.respiratory_rate > 35
        or vitals.oxygen_saturation < 88
    )

    if contains_any(case_text, SEVERE_TERMS) and not has_abnormal_vital:
        issues.append("vital signs do not reflect stated severity")
    if contains_any(case_text, MILD_TERMS) and has_extreme_vital:
        issues.append("vital signs are too severe for a mild or early presentation")
    return issues


def check_contradictions(case: ClinicalCase) -> list[str]:
    issues: list[str] = []
    case_text = all_case_text(case)
    systolic, _ = parse_blood_pressure(case.vitals.blood_pressure)
    heart_rate = case.vitals.heart_rate
    diagnosis_systems = classify_text(case.diagnosis)

    if systolic < 90 and heart_rate < 60 and not contains_any(case_text, LOW_HR_JUSTIFICATION_TERMS):
        issues.append("hypotension with low heart rate needs justification")
    if "respiratory" in diagnosis_systems and case.vitals.respiratory_rate < 10:
        issues.append("respiratory presentation has unexpectedly low respiratory rate")
    return issues


def check_progression_coherence(case: ClinicalCase) -> list[str]:
    issues: list[str] = []
    times = [event.time_min for event in case.progression]
    descriptions = [normalize_text(event.description) for event in case.progression]

    if times != sorted(times):
        issues.append("progression times are not coherent")
    if len(descriptions) > 1 and len(set(descriptions)) == 1:
        issues.append("progression does not show meaningful change over time")
    if any(not description for description in descriptions):
        issues.append("progression contains an empty description")
    return issues


def parse_blood_pressure(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{2,3})/(\d{2,3})", value.strip())
    if not match:
        raise MedicalValidationError("blood_pressure must use systolic/diastolic format")
    return int(match.group(1)), int(match.group(2))


def classify_text(value: str) -> set[str]:
    normalized = normalize_text(value)
    systems: set[str] = set()
    for system, keywords in BODY_SYSTEM_KEYWORDS.items():
        if contains_any(normalized, keywords):
            systems.add(system)
    return systems


def symptom_is_justified(symptom: str, justified_text: str) -> bool:
    tokens = {token for token in tokenize(symptom) if len(token) >= 4}
    return bool(tokens and tokens.intersection(tokenize(justified_text)))


def non_symptom_case_text(case: ClinicalCase) -> str:
    return normalize_text(
        " ".join(
            [
                *case.patient.history,
                *case.risk_factors,
                *case.exam_findings_flat,
                *(event.description for event in case.progression),
                case.outcome.if_treated,
                case.outcome.if_untreated,
            ]
        )
    )


def all_case_text(case: ClinicalCase) -> str:
    return normalize_text(
        " ".join(
            [
                case.diagnosis,
                *case.patient.history,
                *case.symptoms,
                *case.risk_factors,
                *case.exam_findings_flat,
                *(event.description for event in case.progression),
                case.outcome.if_treated,
                case.outcome.if_untreated,
            ]
        )
    )


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def contains_any(value: str, terms: set[str]) -> bool:
    return any(term in value for term in terms)
