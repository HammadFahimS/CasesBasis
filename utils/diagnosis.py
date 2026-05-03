from __future__ import annotations

import re
from typing import Literal

DiagnosisMatchLevel = Literal["correct", "partial", "incorrect"]

EXACT_DIAGNOSIS_ALIAS_GROUPS: dict[str, set[str]] = {
    "myocardial infarction": {"myocardial infarction", "mi", "heart attack", "stemi", "nstemi"},
    "stroke": {"stroke", "cva", "cerebrovascular accident"},
    "asthma exacerbation": {"asthma exacerbation", "asthma attack"},
    "pneumonia": {"pneumonia", "lung infection", "chest infection"},
    "pulmonary embolism": {"pulmonary embolism", "pe", "blood clot in lung"},
    "heart failure": {"heart failure", "chf", "congestive heart failure"},
    "copd exacerbation": {"copd exacerbation", "copd flare up", "copd flare-up"},
    "sepsis": {"sepsis", "bloodstream infection", "systemic infection"},
    "diabetic ketoacidosis": {"diabetic ketoacidosis", "dka"},
    "hypoglycemia": {"hypoglycemia", "low blood sugar"},
    "hyperglycemia": {"hyperglycemia", "high blood sugar"},
    "appendicitis": {"appendicitis", "inflamed appendix"},
    "cholecystitis": {"cholecystitis", "gallbladder inflammation"},
    "pancreatitis": {"pancreatitis", "inflamed pancreas"},
    "urinary tract infection": {"urinary tract infection", "uti"},
    "pyelonephritis": {"pyelonephritis", "kidney infection"},
    "meningitis": {"meningitis", "brain infection"},
    "anaphylaxis": {"anaphylaxis", "severe allergic reaction"},
    "seizure": {"seizure", "convulsion"},
    "syncope": {"syncope", "fainting"},
    "gastroenteritis": {"gastroenteritis", "stomach flu"},
    "gerd": {"gerd", "acid reflux"},
    "anxiety attack": {"anxiety attack", "panic attack"},
    "ectopic pregnancy": {"ectopic pregnancy", "tubal pregnancy"},
}

PARTIAL_DIAGNOSIS_GROUPS: dict[str, set[str]] = {
    "myocardial infarction": {
        "acs",
        "acute coronary syndrome",
        "unstable angina",
        "cardiac ischemia",
        "cardiac ischaemia",
    },
    "stroke": {"tia", "transient ischemic attack", "transient ischaemic attack"},
    "asthma exacerbation": {"bronchospasm", "reactive airway disease"},
    "pulmonary embolism": {"dvt", "deep vein thrombosis"},
}

CONTEXTUAL_PARTIALS: dict[str, set[str]] = {
    "gastroenteritis": {"dehydration"},
    "diabetic ketoacidosis": {"dehydration"},
}

DEHYDRATION_SUPPORT_TERMS = {
    "vomit",
    "vomiting",
    "diarrhea",
    "diarrhoea",
    "poor intake",
    "dry",
    "thirst",
    "tachycardia",
    "polyuria",
    "ketone",
    "dehydration",
    "dehydrated",
}


def normalize_diagnosis_text(value: str) -> str:
    """Normalize diagnosis text for alias-aware comparisons."""

    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(normalized.split())


def diagnosis_match_level(
    expected: str,
    guess: str,
    *,
    case_context: str | None = None,
) -> DiagnosisMatchLevel:
    """Classify a diagnosis guess as exact-equivalent, partial, or incorrect."""

    expected_canonical = canonical_diagnosis(expected)
    guess_canonical = canonical_diagnosis(guess)
    if expected_canonical and guess_canonical and expected_canonical == guess_canonical:
        return "correct"
    if partial_diagnosis_match(
        expected_canonical or normalize_diagnosis_text(expected),
        guess,
        case_context=case_context,
    ):
        return "partial"
    return "incorrect"


def diagnosis_alias_match(expected: str, guess: str) -> bool:
    """Return true when a diagnosis guess is an exact-equivalent alias."""

    return diagnosis_match_level(expected, guess) == "correct"


def diagnosis_partial_match(
    expected: str,
    guess: str,
    *,
    case_context: str | None = None,
) -> bool:
    """Return true when a diagnosis guess is clinically related but not exact."""

    return diagnosis_match_level(expected, guess, case_context=case_context) == "partial"


def diagnosis_alias_in_text(expected: str, text: str) -> str | None:
    """Return the exact-equivalent diagnosis phrase from text when present."""

    expected_canonical = canonical_diagnosis(expected)
    candidates = aliases_for_canonical(expected_canonical) if expected_canonical else {expected}
    return first_phrase_in_text(text, candidates)


def canonical_diagnosis(value: str) -> str | None:
    normalized = normalize_diagnosis_text(value)
    if not normalized:
        return None
    for canonical, aliases in normalized_alias_groups().items():
        if normalized in aliases:
            return canonical
    for canonical, aliases in normalized_alias_groups().items():
        for alias in sorted(aliases, key=len, reverse=True):
            if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
    return normalized


def partial_diagnosis_match(
    expected_canonical: str,
    guess: str,
    *,
    case_context: str | None,
) -> bool:
    normalized_guess = normalize_diagnosis_text(guess)
    partial_aliases = normalized_partial_groups().get(expected_canonical, set())
    if normalized_guess in partial_aliases:
        return True
    contextual_aliases = normalized_contextual_partials().get(expected_canonical, set())
    if normalized_guess in contextual_aliases:
        return contextual_partial_supported(expected_canonical, normalized_guess, case_context)
    return False


def contextual_partial_supported(
    expected_canonical: str,
    normalized_guess: str,
    case_context: str | None,
) -> bool:
    if normalized_guess != "dehydration":
        return False
    if expected_canonical not in {"gastroenteritis", "diabetic ketoacidosis"}:
        return False
    normalized_context = normalize_diagnosis_text(case_context or "")
    return any(term in normalized_context for term in DEHYDRATION_SUPPORT_TERMS)


def aliases_for_canonical(canonical: str | None) -> set[str]:
    if canonical is None:
        return set()
    normalized_groups = normalized_alias_groups()
    return normalized_groups.get(canonical, {canonical})


def first_phrase_in_text(text: str, phrases: set[str]) -> str | None:
    normalized_text = normalize_diagnosis_text(text)
    for phrase in sorted(phrases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_text):
            return extract_original_phrase(text, phrase) or phrase
    return None


def extract_original_phrase(text: str, normalized_phrase: str) -> str | None:
    words_pattern = r"\W+".join(re.escape(part) for part in normalized_phrase.split())
    match = re.search(rf"\b{words_pattern}\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def normalized_alias_groups() -> dict[str, set[str]]:
    return {
        normalize_diagnosis_text(canonical): {
            normalize_diagnosis_text(alias)
            for alias in aliases
        }
        for canonical, aliases in EXACT_DIAGNOSIS_ALIAS_GROUPS.items()
    }


def normalized_partial_groups() -> dict[str, set[str]]:
    return {
        normalize_diagnosis_text(canonical): {
            normalize_diagnosis_text(alias)
            for alias in aliases
        }
        for canonical, aliases in PARTIAL_DIAGNOSIS_GROUPS.items()
    }


def normalized_contextual_partials() -> dict[str, set[str]]:
    return {
        normalize_diagnosis_text(canonical): {
            normalize_diagnosis_text(alias)
            for alias in aliases
        }
        for canonical, aliases in CONTEXTUAL_PARTIALS.items()
    }
