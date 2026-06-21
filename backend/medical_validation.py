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
        "arrhythmia",
        "angina",
        "myocardial",
        "atrial",
        "ventricular",
        "endocard",
        "myocard",
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
        "hypoxemia",
        "hypercapnia",
        "sputum",
        "pleurit",
        # Chest symptoms belong to respiratory just as much as cardiac
        "chest",
    },
    "neurologic": {
        "neuro",
        "brain",
        "stroke",
        "seizure",
        "numbness",
        "speech",
        "aphasia",
        "ataxia",
        "nystagmus",
        "hemiplegia",
        "meningism",
        "meningeal",
        # Specific neurologic signs not seen as secondary effects of other systems
        "photophobia",
        "phonophobia",
        "nuchal",
        "focal",
    },
    "gastrointestinal": {
        "abdominal",
        "abdomen",
        "gastric",
        "intestinal",
        "bowel",
        "liver",
        "hepat",
        "pancrea",
        "append",
        "diarrhea",
        "melena",
        "hematemesis",
        "ileus",
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
        "hematuria",
    },
    "dermatologic": {
        "skin",
        "rash",
        "hives",
        "itch",
        "lesion",
        "blister",
        "urticaria",
        "erythema",
    },
    "musculoskeletal": {
        "joint",
        "muscle",
        "bone",
        "back",
        "limb",
        "fracture",
        "myalgia",
        "arthralgia",
        "arthritis",
        "trauma",
    },
    # Systemic covers non-specific symptoms that appear as secondary effects across
    # many primary-system diseases. Symptoms matched here are never "unrelated".
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
        "nausea",
        "vomit",
        "diaphoresis",
        "pallor",
        "anorexia",
        "anxiety",
        "restless",
        "pain",
        # Non-specific neurological: these are secondary effects of hypoxia, metabolic
        # disturbance, or low perfusion — not primary neurologic disease.
        "confusion",
        "altered",
        "agitation",
        "lethargy",
        "drowsy",
        "headache",
        "dizzy",
        "dizziness",
        "lightheaded",
        "syncope",
        # Haemodynamic / secondary circulatory symptoms
        "tachycardia",
        "palpitation",
        "pressure",
        "hypotension",
        # Fluid / perfusion secondary symptoms
        "edema",
        "swelling",
        "cyanosis",
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

# Documented pathophysiological relationships: when the primary diagnosis system
# is X, symptoms from these additional systems are clinically expected as
# downstream / secondary manifestations.
#
# NOTE: "neurologic" is intentionally absent from "respiratory" because
# hypoxia-driven confusion, altered mental status, and lethargy have been
# moved to the `systemic` bucket above — no explicit cross-system mapping
# is needed, and including it would incorrectly allow specific focal
# neurologic presentations (e.g. meningism) on respiratory diagnoses.
EXPECTED_SECONDARY_SYSTEMS: dict[str, set[str]] = {
    "respiratory": {
        "cardiac",          # cor pulmonale, compensatory tachycardia (also in systemic)
        "musculoskeletal",  # accessory muscle use during respiratory distress
    },
    "cardiac": {
        "respiratory",      # pulmonary oedema → dyspnoea, tachypnoea
        "gastrointestinal", # hepatic congestion, nausea
    },
    "neurologic": {
        "musculoskeletal",  # motor deficits, paralysis
        "gastrointestinal", # dysphagia, aspiration
    },
    "gastrointestinal": {
        "musculoskeletal",  # referred pain, guarding, rigidity
    },
    "genitourinary": {
        "gastrointestinal", # referred / peritoneal pain
        "musculoskeletal",  # flank radiation
    },
    "musculoskeletal": {
        "neurologic",       # nerve compression, radiculopathy
    },
}


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

    # Primary systems + universal systemic bucket + known downstream secondaries
    allowed_systems = diagnosis_systems | GENERAL_SYMPTOM_SYSTEMS
    for primary in diagnosis_systems:
        allowed_systems |= EXPECTED_SECONDARY_SYSTEMS.get(primary, set())

    issues: list[str] = []
    justified_text = non_symptom_case_text(case)
    related_count = 0
    unrelated_symptoms: list[str] = []
    classified_count = 0

    for symptom in case.symptoms:
        symptom_systems = classify_text(symptom)
        if not symptom_systems:
            # Unclassifiable symptom — skip, do not penalise
            continue
        classified_count += 1
        if symptom_systems.intersection(allowed_systems):
            related_count += 1
            continue
        if symptom_is_justified(symptom, justified_text):
            continue
        unrelated_symptoms.append(symptom)

    if classified_count > 0 and related_count == 0:
        issues.append("symptoms do not logically relate to the diagnosis body system")

    # Require a proportional fraction of symptoms to be unrelated before flagging.
    # Real presentations routinely involve 1-2 cross-system symptoms; we only fire
    # when a meaningful fraction of the symptom list is clearly off-system.
    unrelated_threshold = max(1, round(classified_count * 0.4))
    if len(unrelated_symptoms) >= unrelated_threshold:
        issues.append("symptoms from unrelated symptom systems are present without justification")

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
