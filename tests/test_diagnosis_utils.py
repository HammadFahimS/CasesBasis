from __future__ import annotations

from utils.diagnosis import diagnosis_alias_match, diagnosis_match_level


def test_diagnosis_alias_match_accepts_common_equivalents() -> None:
    assert diagnosis_alias_match("myocardial infarction", "heart attack")
    assert diagnosis_alias_match("myocardial infarction", "MI")
    assert diagnosis_alias_match("myocardial infarction", "STEMI")
    assert diagnosis_alias_match("myocardial infarction", "NSTEMI")
    assert diagnosis_alias_match("cerebrovascular accident", "stroke")
    assert diagnosis_alias_match("CVA", "stroke")
    assert diagnosis_alias_match("pulmonary embolism", "PE")
    assert diagnosis_alias_match("diabetic ketoacidosis", "DKA")
    assert diagnosis_alias_match("urinary tract infection", "UTI")
    assert diagnosis_alias_match("asthma attack", "asthma exacerbation")
    assert diagnosis_alias_match("pneumonia", "lung infection")
    assert diagnosis_alias_match("heart failure", "CHF")
    assert diagnosis_alias_match("hypoglycemia", "low blood sugar")
    assert diagnosis_alias_match("ectopic pregnancy", "tubal pregnancy")


def test_diagnosis_alias_match_rejects_unrelated_guess() -> None:
    assert not diagnosis_alias_match("myocardial infarction", "pneumonia")


def test_diagnosis_match_level_identifies_partial_related_diagnoses() -> None:
    assert diagnosis_match_level("myocardial infarction", "acute coronary syndrome") == "partial"
    assert diagnosis_match_level("myocardial infarction", "ACS") == "partial"
    assert diagnosis_match_level("myocardial infarction", "cardiac ischemia") == "partial"
    assert diagnosis_match_level("stroke", "TIA") == "partial"
    assert diagnosis_match_level("asthma exacerbation", "bronchospasm") == "partial"
    assert diagnosis_match_level("pulmonary embolism", "DVT") == "partial"


def test_diagnosis_match_level_rejects_false_positive_partials() -> None:
    assert diagnosis_match_level("myocardial infarction", "GERD") == "incorrect"
    assert diagnosis_match_level("stroke", "seizure") == "incorrect"
    assert diagnosis_match_level("pulmonary embolism", "heart failure") == "incorrect"
    assert diagnosis_match_level("pneumonia", "chest pain") == "incorrect"


def test_contextual_dehydration_partial_requires_supporting_case_findings() -> None:
    assert (
        diagnosis_match_level(
            "gastroenteritis",
            "dehydration",
            case_context="vomiting diarrhea dry mucous membranes tachycardia",
        )
        == "partial"
    )
    assert diagnosis_match_level("gastroenteritis", "dehydration", case_context="mild nausea") == "incorrect"
    assert diagnosis_match_level("pneumonia", "dehydration", case_context="dry") == "incorrect"
