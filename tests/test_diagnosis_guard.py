"""Diverse unit tests for the central diagnosis guard.

Covers:
- Literal diagnosis substitution
- Alias coverage across many concept classes (cardiac, pulmonary, neuro, GI…)
- Interpretive-phrase detection AND truncation
- Case insensitivity and word-boundary correctness
- Edge cases: empty input, partial matches, mixed-content findings
"""

from __future__ import annotations

import pytest

from backend.diagnosis_guard import (
    contains_diagnosis_leak,
    hidden_terms,
    scrub_finding_text,
)


# ──────────────────────────────────────────────────────────────────────────────
# hidden_terms()
# ──────────────────────────────────────────────────────────────────────────────


class TestHiddenTerms:
    def test_empty_diagnosis_returns_empty(self) -> None:
        assert hidden_terms("") == []
        assert hidden_terms("   ") == []

    def test_unknown_diagnosis_returns_only_literal(self) -> None:
        # "myxoma" isn't in our concept dict — only the literal should come back.
        result = hidden_terms("atrial myxoma")
        assert result == ["atrial myxoma"]

    @pytest.mark.parametrize(
        "diagnosis,must_contain",
        [
            ("myocardial infarction", ["heart attack", "mi", "stemi", "nstemi", "acs"]),
            ("Anterior STEMI", ["heart attack", "mi", "stemi", "acute coronary syndrome"]),
            ("inferior MI", ["myocardial infarction", "heart attack", "stemi"]),
            ("ACS", ["myocardial infarction", "heart attack"]),
            ("stroke", ["cva", "cerebrovascular accident", "brain attack"]),
            ("CVA", ["stroke", "cerebrovascular accident"]),
            ("pulmonary embolism", ["pe", "lung clot", "blood clot in lung"]),
            ("pneumonia", ["lung infection", "chest infection"]),
            ("DKA", ["diabetic ketoacidosis", "diabetic acidosis"]),
            ("anaphylaxis", ["anaphylactic shock", "severe allergic reaction"]),
            ("sepsis", ["septic shock", "septicemia"]),
            ("appendicitis", ["acute appendicitis"]),
            ("atrial fibrillation", ["afib", "atrial fib"]),
            ("complete heart block", ["3rd degree av block", "third-degree av block"]),
            ("CHF", ["heart failure", "congestive heart failure"]),
        ],
    )
    def test_aliases_expand_for_known_concepts(
        self, diagnosis: str, must_contain: list[str]
    ) -> None:
        terms = [t.lower() for t in hidden_terms(diagnosis)]
        for expected in must_contain:
            assert expected.lower() in terms, (
                f"{expected!r} missing from hidden_terms({diagnosis!r}) = {terms}"
            )

    def test_first_concept_wins_no_cross_pollution(self) -> None:
        # "atrial fibrillation" must NOT pull in MI aliases.
        terms = [t.lower() for t in hidden_terms("atrial fibrillation")]
        assert "heart attack" not in terms
        assert "stemi" not in terms

    def test_literal_diagnosis_always_first_and_present(self) -> None:
        terms = hidden_terms("Anterior STEMI")
        assert terms[0] == "Anterior STEMI"


# ──────────────────────────────────────────────────────────────────────────────
# contains_diagnosis_leak() — detection
# ──────────────────────────────────────────────────────────────────────────────


class TestLeakDetection:
    def test_empty_text_is_safe(self) -> None:
        assert contains_diagnosis_leak("", "myocardial infarction") is False
        assert contains_diagnosis_leak("   ", "myocardial infarction") is False

    def test_pure_observation_passes(self) -> None:
        assert (
            contains_diagnosis_leak(
                "Heart sounds dual, no added sounds, no murmurs.",
                "myocardial infarction",
            )
            is False
        )
        assert (
            contains_diagnosis_leak(
                "Bilateral basal fine crackles to mid-zones.",
                "community-acquired pneumonia",
            )
            is False
        )

    @pytest.mark.parametrize(
        "text,diagnosis",
        [
            ("Findings suggest a heart attack.", "myocardial infarction"),
            ("Likely an MI, get a troponin.", "myocardial infarction"),
            ("Picture is consistent with STEMI.", "myocardial infarction"),
            ("This is an ACS until proven otherwise.", "myocardial infarction"),
            ("Cardiac event with anterior changes.", "myocardial infarction"),
            ("Consistent with pulmonary embolism.", "PE"),
            ("DVT until proven otherwise.", "deep vein thrombosis"),
            ("Looks like a brain attack.", "stroke"),
            ("This is a CVA.", "stroke"),
            ("Suggestive of pneumonia.", "community-acquired pneumonia"),
            ("Probably DKA.", "diabetic ketoacidosis"),
            ("Concerning for sepsis.", "sepsis"),
        ],
    )
    def test_alias_leaks_are_caught(self, text: str, diagnosis: str) -> None:
        assert contains_diagnosis_leak(text, diagnosis) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Consistent with normal findings.",
            "Likely benign.",
            "Probably nothing.",
            "Differential includes several things.",
            "Diagnosis is unclear.",
            "Most likely cardiac in origin.",
            "Raises concern for the leading process.",
            "Pathognomonic for a known condition.",
        ],
    )
    def test_interpretive_phrases_caught_even_without_diagnosis_name(
        self, text: str
    ) -> None:
        # The diagnosis itself isn't named, but the LLM is reasoning out loud —
        # we still want to scrub. (Diagnosis arg is irrelevant here.)
        assert contains_diagnosis_leak(text, "irrelevant") is True

    def test_word_boundaries_prevent_false_positives(self) -> None:
        # "MI" must not match "summing" or "communication" or "miami"
        assert (
            contains_diagnosis_leak(
                "Patient minimising her symptoms.", "myocardial infarction"
            )
            is False
        )
        assert (
            contains_diagnosis_leak(
                "From Miami, no recent travel.", "myocardial infarction"
            )
            is False
        )
        # And "PE" must not match "shapeshift" or "peptic"
        assert (
            contains_diagnosis_leak("Peptic discomfort reported.", "pulmonary embolism")
            is False
        )


# ──────────────────────────────────────────────────────────────────────────────
# scrub_finding_text() — full pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestScrubFindingText:
    def test_pure_observation_passes_unchanged_except_period(self) -> None:
        out = scrub_finding_text(
            "Bilateral basal crackles to mid-zones.", "pneumonia"
        )
        assert out == "Bilateral basal crackles to mid-zones."

    def test_literal_diagnosis_is_substituted(self) -> None:
        out = scrub_finding_text(
            "Findings of pneumonia are present.", "pneumonia"
        )
        assert "pneumonia" not in out.lower()
        assert "the condition" in out.lower()

    def test_alias_is_substituted(self) -> None:
        out = scrub_finding_text(
            "Findings of heart attack confirmed.", "myocardial infarction"
        )
        assert "heart attack" not in out.lower()
        assert "myocardial infarction" not in out.lower()

    def test_interpretive_clause_dropped_observation_kept(self) -> None:
        out = scrub_finding_text(
            "S3 gallop at the apex, suggesting LV failure.",
            "heart failure",
        )
        assert "S3 gallop at the apex" in out
        assert "suggesting" not in out.lower()
        assert "lv failure" not in out.lower()

    def test_em_dash_clause_dropped(self) -> None:
        out = scrub_finding_text(
            "Bilateral basal crackles — consistent with pulmonary oedema.",
            "heart failure",
        )
        assert "crackles" in out.lower()
        assert "consistent with" not in out.lower()
        assert "pulmonary" not in out.lower()

    def test_multiple_sentences_filter_independently(self) -> None:
        out = scrub_finding_text(
            (
                "Tender right upper quadrant. Murphy's sign positive. "
                "Likely acute cholecystitis given the history."
            ),
            "cholecystitis",
        )
        assert "Tender right upper quadrant" in out
        assert "Murphy's sign positive" in out
        assert "likely" not in out.lower()
        assert "cholecystitis" not in out.lower()

    def test_entirely_interpretive_finding_returns_empty(self) -> None:
        out = scrub_finding_text(
            "Likely heart attack, consistent with ACS.",
            "myocardial infarction",
        )
        assert out == ""

    def test_empty_input_returns_empty(self) -> None:
        assert scrub_finding_text("", "myocardial infarction") == ""

    def test_diagnosis_with_no_aliases_still_redacts_literal(self) -> None:
        out = scrub_finding_text(
            "Examination reveals atrial myxoma.", "atrial myxoma"
        )
        assert "atrial myxoma" not in out.lower()
        assert "the condition" in out.lower()

    @pytest.mark.parametrize(
        "raw,diagnosis,banned",
        [
            (
                "Diaphoretic, chest pain — concerning for MI.",
                "myocardial infarction",
                ["MI", "concerning for"],
            ),
            (
                "Reduced breath sounds at the right base, raises concern for pneumonia.",
                "pneumonia",
                ["pneumonia", "raises concern for"],
            ),
            (
                "Pansystolic murmur radiating to the axilla, probably mitral regurgitation.",
                "mitral regurgitation",
                ["probably", "mitral regurgitation"],
            ),
            (
                "Right calf tender and swollen. Differential includes DVT.",
                "deep vein thrombosis",
                ["DVT", "differential"],
            ),
            (
                "Tachypneic, hypoxic on room air, in keeping with PE.",
                "pulmonary embolism",
                ["PE", "in keeping with"],
            ),
            (
                "Glasgow 12, right-sided weakness, looks like a stroke.",
                "stroke",
                ["stroke", "looks like"],
            ),
        ],
    )
    def test_no_banned_word_survives(
        self, raw: str, diagnosis: str, banned: list[str]
    ) -> None:
        out = scrub_finding_text(raw, diagnosis).lower()
        for bad in banned:
            assert bad.lower() not in out, f"banned word {bad!r} survived in {out!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Case-insensitivity sanity check
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,diagnosis",
    [
        ("This MIGHT be a heart attack.", "myocardial infarction"),
        ("DKA cannot be ruled out.", "diabetic ketoacidosis"),
        ("Cva picture.", "stroke"),
        ("Most LIKELY cardiac.", "irrelevant"),
    ],
)
def test_detection_is_case_insensitive(text: str, diagnosis: str) -> None:
    assert contains_diagnosis_leak(text, diagnosis) is True
