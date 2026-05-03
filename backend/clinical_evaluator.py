from __future__ import annotations

import re
from typing import Any

from agents.reasoning_reviewer import ReasoningReviewAgent
from backend.patient_response import normalize_case
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import (
    ClinicalEvaluation,
    DiagnosisGuessEvaluation,
    InvestigationInterpretationFeedback,
    LearningRecommendations,
    QuestionRelevanceEvaluation,
    ReasoningFeedback,
)
from schemas.investigation import InvestigationResult
from utils.common import most_common
from utils.diagnosis import diagnosis_alias_in_text, diagnosis_match_level

MIN_TOKEN_LENGTH = 3
CLINICAL_QUESTION_TERMS = {
    "symptom",
    "pain",
    "fever",
    "cough",
    "breath",
    "breathing",
    "dizzy",
    "dizziness",
    "confusion",
    "urine",
    "urinating",
    "peeing",
    "nausea",
    "vomit",
    "history",
    "medication",
    "allergy",
    "risk",
    "vitals",
    "pulse",
    "heart",
    "pressure",
    "oxygen",
    "exam",
    "started",
    "progress",
    "worse",
    "better",
    "diagnosis",
    "diagnosed",
}
KNOWN_SYMPTOM_GUESSES = {
    "abdominal pain",
    "chest pain",
    "chest pressure",
    "confusion",
    "cough",
    "dizziness",
    "dyspnea",
    "fatigue",
    "fever",
    "headache",
    "nausea",
    "shortness of breath",
    "sweating",
    "vomiting",
}


class ClinicalEvaluator:
    """Scores simulated student interactions against generated case data."""

    def __init__(
        self,
        *,
        enable_reasoning_review: bool = False,
        reasoning_reviewer: ReasoningReviewAgent | None = None,
    ) -> None:
        self.enable_reasoning_review = enable_reasoning_review
        self.reasoning_reviewer = reasoning_reviewer

    def evaluate(
        self,
        case: ClinicalCase | dict[str, Any] | str,
        student_questions: list[str],
        patient_responses: list[str],
        previous_results: list[ClinicalEvaluation | dict[str, Any] | str] | None = None,
        session_history: list[ClinicalEvaluation | dict[str, Any] | str] | None = None,
        investigations_ordered: list[InvestigationResult | dict[str, Any] | str] | None = None,
        student_interpretations: list[str] | None = None,
    ) -> ClinicalEvaluation:
        clinical_case = normalize_case(case)
        raw_questions = [" ".join(question.split()).strip() for question in student_questions if question.strip()]
        questions = [normalize_text(question) for question in raw_questions]
        responses = [normalize_text(response) for response in patient_responses if response.strip()]

        identified_symptoms = identify_symptoms(clinical_case, questions, responses)
        missed_symptoms = [symptom for symptom in clinical_case.symptoms if symptom not in identified_symptoms]
        relevance = evaluate_question_relevance(clinical_case, questions)
        diagnosis_guess = evaluate_diagnosis_guess(clinical_case, raw_questions)
        investigation_feedback = evaluate_investigation_interpretations(
            case=clinical_case,
            diagnosis_guess=diagnosis_guess,
            investigations_ordered=investigations_ordered or [],
            student_interpretations=student_interpretations or [],
        )
        reasoning_feedback = build_reasoning_feedback(
            case=clinical_case,
            questions=questions,
            identified_symptoms=identified_symptoms,
            relevance=relevance,
            diagnosis_guess=diagnosis_guess,
        )
        score = score_performance(
            clinical_case=clinical_case,
            identified_symptoms=identified_symptoms,
            relevance=relevance,
            diagnosis_guess=diagnosis_guess,
            missed_reasoning_steps=reasoning_feedback.missed_reasoning_steps,
            investigation_feedback=investigation_feedback,
        )

        evaluation = ClinicalEvaluation(
            correct_diagnosis_guess=diagnosis_guess,
            key_symptoms_identified=identified_symptoms,
            key_symptoms_missed=missed_symptoms,
            relevance_of_questions=relevance,
            overall_score=score,
            brief_feedback=build_feedback(
                identified_count=len(identified_symptoms),
                total_symptoms=len(clinical_case.symptoms),
                relevance=relevance,
                diagnosis_guess=diagnosis_guess,
            ),
            reasoning_feedback=reasoning_feedback,
            learning_recommendations=build_learning_recommendations(
                current_score=score,
                total_symptoms=len(clinical_case.symptoms),
                identified_symptoms=identified_symptoms,
                missed_symptoms=missed_symptoms,
                relevance=relevance,
                diagnosis_guess=diagnosis_guess,
                reasoning_feedback=reasoning_feedback,
                previous_results=[*(previous_results or []), *(session_history or [])],
            ),
            investigation_interpretation_feedback=investigation_feedback,
        )
        if self.should_review_reasoning(evaluation, question_count=len(questions)):
            reviewed_feedback = self.get_reasoning_reviewer().review(
                case=clinical_case,
                evaluation=evaluation,
                student_questions=questions,
                patient_responses=responses,
            )
            return evaluation.model_copy(update={"reasoning_feedback": reviewed_feedback})
        return evaluation

    def should_review_reasoning(
        self,
        evaluation: ClinicalEvaluation,
        *,
        question_count: int,
    ) -> bool:
        if not self.enable_reasoning_review:
            return False
        return 40 <= evaluation.overall_score <= 70 or reasoning_feedback_has_low_confidence(
            evaluation,
            question_count=question_count,
        )

    def get_reasoning_reviewer(self) -> ReasoningReviewAgent:
        if self.reasoning_reviewer is None:
            self.reasoning_reviewer = ReasoningReviewAgent()
        return self.reasoning_reviewer


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def tokenize(value: str) -> set[str]:
    return {
        normalize_clinical_token(token)
        for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) >= MIN_TOKEN_LENGTH
    }


def normalize_clinical_token(token: str) -> str:
    if token.startswith("fever"):
        return "fever"
    if token.startswith("confus"):
        return "confusion"
    if token.startswith("dizz"):
        return "dizziness"
    if token.startswith("urin") or token.startswith("pee"):
        return "urine"
    if token.startswith("breath"):
        return "breath"
    return token


def identify_symptoms(
    case: ClinicalCase,
    questions: list[str],
    responses: list[str],
) -> list[str]:
    identified: list[str] = []
    question_text = normalize_text(" ".join(questions))
    response_text = normalize_text(" ".join(responses))
    for symptom in case.symptoms:
        if symptom_was_directly_asked(symptom, question_text) or symptom_was_revealed(
            symptom,
            response_text,
        ):
            identified.append(symptom)
    return identified


def symptom_was_directly_asked(symptom: str, question_text: str) -> bool:
    return symptom_mentioned_in_text(symptom, question_text)


def symptom_was_revealed(symptom: str, response_text: str) -> bool:
    return symptom_mentioned_in_text(symptom, response_text)


def symptom_mentioned_in_text(symptom: str, text: str) -> bool:
    if not text.strip():
        return False
    symptom_text = normalize_text(symptom)
    if symptom_text in text:
        return True
    symptom_tokens = meaningful_symptom_tokens(symptom_text)
    if not symptom_tokens:
        return False
    text_tokens = tokenize(text)
    if len(symptom_tokens) == 1:
        return bool(symptom_tokens.intersection(text_tokens))
    distinctive_tokens = {"urine", "nausea", "vomit", "sweating", "cough", "fever", "breath"}
    if symptom_tokens.intersection(distinctive_tokens).intersection(text_tokens):
        return True
    return len(symptom_tokens.intersection(text_tokens)) >= min(2, len(symptom_tokens))


def meaningful_symptom_tokens(symptom_text: str) -> set[str]:
    low_signal = {"pain", "feels", "feeling", "mild", "severe"}
    return {token for token in tokenize(symptom_text) if token not in low_signal}


def evaluate_question_relevance(
    case: ClinicalCase,
    questions: list[str],
) -> QuestionRelevanceEvaluation:
    relevant_questions: list[str] = []
    irrelevant_questions: list[str] = []
    case_tokens = case_relevance_tokens(case)

    for question in questions:
        question_tokens = tokenize(question)
        if (
            question_tokens.intersection(case_tokens)
            or question_tokens.intersection(CLINICAL_QUESTION_TERMS)
            or diagnosis_alias_in_text(case.diagnosis, question) is not None
        ):
            relevant_questions.append(question)
        else:
            irrelevant_questions.append(question)

    return QuestionRelevanceEvaluation(
        relevant_count=len(relevant_questions),
        irrelevant_count=len(irrelevant_questions),
        relevant_questions=relevant_questions,
        irrelevant_questions=irrelevant_questions,
    )


def case_relevance_tokens(case: ClinicalCase) -> set[str]:
    visible_case_text = " ".join(
        [
            *case.patient.history,
            *case.symptoms,
            *case.risk_factors,
            *case.exam_findings,
            *(event.description for event in case.progression),
            str(case.patient.age),
            case.patient.gender,
            str(case.vitals.heart_rate),
            case.vitals.blood_pressure,
            str(case.vitals.respiratory_rate),
            str(case.vitals.oxygen_saturation),
        ]
    )
    return tokenize(visible_case_text)


def evaluate_diagnosis_guess(
    case: ClinicalCase,
    questions: list[str],
) -> DiagnosisGuessEvaluation:
    for question in questions:
        matched_guess = diagnosis_alias_in_text(case.diagnosis, question)
        if matched_guess:
            return DiagnosisGuessEvaluation(
                provided=True,
                guess=matched_guess,
                is_correct=True,
                diagnosis_match_level="correct",
            )
    guessed_condition = extract_condition_guess(
        [question for question in questions if is_diagnosis_guess_question(question)]
    )
    if guessed_condition:
        if is_invalid_diagnosis_guess(guessed_condition):
            return DiagnosisGuessEvaluation(
                provided=False,
                guess=None,
                is_correct=None,
                diagnosis_match_level=None,
            )
        match_level = diagnosis_match_level(
            case.diagnosis,
            guessed_condition,
            case_context=diagnosis_context(case),
        )
        return DiagnosisGuessEvaluation(
            provided=True,
            guess=guessed_condition,
            is_correct=match_level == "correct",
            diagnosis_match_level=match_level,
        )
    return DiagnosisGuessEvaluation(
        provided=False,
        guess=None,
        is_correct=None,
        diagnosis_match_level=None,
    )


def is_invalid_diagnosis_guess(guess: str) -> bool:
    return normalize_text(guess) in KNOWN_SYMPTOM_GUESSES


def diagnosis_context(case: ClinicalCase) -> str:
    return " ".join(
        [
            case.diagnosis,
            *case.symptoms,
            *case.risk_factors,
            *case.exam_findings,
            *(event.description for event in case.progression),
        ]
    )


def extract_condition_guess(questions: list[str]) -> str | None:
    patterns = (
        r"diagnosis\s+guess\s*:?\s*([a-z][a-z0-9 \-]{1,60})",
        r"(?:is|could this be|do they have|does the patient have|diagnosis is|diagnosed with)\s+([a-z][a-z0-9 \-]{2,60})",
        r"(?:i think|suspect|guess)\s+(?:it is|it's|this is|the diagnosis is)?\s*([a-z][a-z0-9 \-]{2,60})",
    )
    for question in questions:
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                guess = match.group(1).strip(" ?.!")
                if guess:
                    return guess
    return None


def evaluate_investigation_interpretations(
    *,
    case: ClinicalCase,
    diagnosis_guess: DiagnosisGuessEvaluation,
    investigations_ordered: list[InvestigationResult | dict[str, Any] | str],
    student_interpretations: list[str],
) -> InvestigationInterpretationFeedback:
    results = normalize_investigation_results(investigations_ordered)
    interpretations = [
        normalize_text(interpretation)
        for interpretation in student_interpretations
        if interpretation.strip()
    ]
    interpretation_text = " ".join(interpretations)
    correctly_interpreted: list[str] = []
    missed_findings: list[str] = []
    incorrect_interpretations: list[str] = []

    for result in results:
        abnormal_findings = important_abnormal_findings(result)
        if not abnormal_findings:
            continue
        if interpretation_calls_abnormal_result_normal(result, interpretation_text):
            incorrect_interpretations.append(
                f"Called abnormal {safe_investigation_name(result)} findings normal."
            )
            continue
        for finding in abnormal_findings:
            if interpretation_matches_finding(finding, result, interpretation_text):
                correctly_interpreted.append(finding.feedback_text)
            else:
                missed_findings.append(finding.feedback_text)

    feedback = InvestigationInterpretationFeedback(
        correctly_interpreted_findings=dedupe_preserve_order(correctly_interpreted),
        missed_findings=dedupe_preserve_order(missed_findings),
        incorrect_interpretations=dedupe_preserve_order(incorrect_interpretations),
        summary=build_investigation_interpretation_summary(
            results=results,
            interpretations=interpretations,
            correctly_interpreted=correctly_interpreted,
            missed_findings=missed_findings,
            incorrect_interpretations=incorrect_interpretations,
        ),
    )
    if diagnosis_guess.is_correct is True:
        return feedback
    return hide_diagnosis_in_investigation_feedback(feedback, case.diagnosis)


class InterpretableFinding:
    """Internal representation of an investigation finding students can interpret."""

    def __init__(self, *, feedback_text: str, keywords: set[str]) -> None:
        self.feedback_text = feedback_text
        self.keywords = keywords


def normalize_investigation_results(
    investigations_ordered: list[InvestigationResult | dict[str, Any] | str],
) -> list[InvestigationResult]:
    results: list[InvestigationResult] = []
    for investigation in investigations_ordered:
        try:
            if isinstance(investigation, InvestigationResult):
                result = investigation
            elif isinstance(investigation, str):
                result = InvestigationResult.model_validate_json(investigation)
            else:
                result = InvestigationResult.model_validate(investigation)
        except Exception:
            continue
        if result.status == "available":
            results.append(result)
    return results


def important_abnormal_findings(result: InvestigationResult) -> list[InterpretableFinding]:
    findings: list[InterpretableFinding] = []
    structured = result.structured_result
    investigation_name = safe_investigation_name(result)

    st_segments = structured.get("st_segments")
    if isinstance(st_segments, dict):
        elevated_leads = [lead for lead, status in st_segments.items() if status == "elevated"]
        depressed_leads = [lead for lead, status in st_segments.items() if status == "depressed"]
        if elevated_leads:
            findings.append(
                InterpretableFinding(
                    feedback_text=f"Recognized ST elevation in {', '.join(elevated_leads)}.",
                    keywords={"st", "elevation", "elevated", "ischemic", "ischemia", "acute", "abnormal"},
                )
            )
        if depressed_leads:
            findings.append(
                InterpretableFinding(
                    feedback_text=f"Recognized reciprocal ST depression in {', '.join(depressed_leads)}.",
                    keywords={"st", "depression", "depressed", "reciprocal", "ischemic", "ischemia", "abnormal"},
                )
            )

    value_flag = structured.get("flag")
    if is_abnormal_flag(value_flag):
        name = str(structured.get("test") or investigation_name)
        value = structured.get("value")
        unit = structured.get("unit")
        reference = structured.get("reference_range")
        value_text = format_value_text(value=value, unit=unit, reference=reference)
        findings.append(
            InterpretableFinding(
                feedback_text=f"Recognized abnormal {name}{value_text}.",
                keywords=keywords_for_abnormal_value(name, value_flag),
            )
        )

    values = structured.get("values")
    if isinstance(values, list):
        for value_result in values:
            if not isinstance(value_result, dict) or not is_abnormal_flag(value_result.get("flag")):
                continue
            name = str(value_result.get("name") or value_result.get("test") or investigation_name)
            value_text = format_value_text(
                value=value_result.get("value"),
                unit=value_result.get("unit"),
                reference=value_result.get("reference_range"),
            )
            findings.append(
                InterpretableFinding(
                    feedback_text=f"Recognized abnormal {name}{value_text}.",
                    keywords=keywords_for_abnormal_value(name, value_result.get("flag")),
                )
            )

    listed_findings = structured.get("findings")
    if isinstance(listed_findings, list):
        for raw_finding in listed_findings:
            finding_text = str(raw_finding).strip()
            if finding_text and finding_sounds_abnormal(finding_text):
                findings.append(
                    InterpretableFinding(
                        feedback_text=f"Recognized {finding_text}.",
                        keywords=tokenize(finding_text).union(tokens_for_test_name(investigation_name)),
                    )
                )
    return dedupe_findings(findings)


def is_abnormal_flag(flag: object) -> bool:
    normalized = normalize_text(str(flag or ""))
    return normalized not in {"", "normal", "non-contributory", "noncontributory", "within range"}


def finding_sounds_abnormal(finding: str) -> bool:
    normalized = normalize_text(finding)
    if normalized.startswith(("no ", "normal ", "within ", "unremarkable")):
        return False
    abnormal_terms = {
        "abnormal",
        "elevated",
        "depressed",
        "high",
        "low",
        "positive",
        "opacity",
        "infiltrate",
        "fracture",
        "leukocytosis",
        "consolidation",
        "reciprocal",
        "st",
        "ischemic",
    }
    return bool(tokenize(normalized).intersection(abnormal_terms))


def interpretation_matches_finding(
    finding: InterpretableFinding,
    result: InvestigationResult,
    interpretation_text: str,
) -> bool:
    if not interpretation_text:
        return False
    interpretation_tokens = tokenize(interpretation_text)
    if tokens_for_test_name(result.investigation).intersection({"ecg", "ekg"}):
        return interpretation_matches_ecg(interpretation_text)
    if "troponin" in normalize_text(result.investigation) or "troponin" in normalize_text(
        str(result.structured_result.get("test", ""))
    ):
        return bool(
            interpretation_tokens.intersection({"elevated", "high", "positive", "abnormal", "raised"})
        )
    shared_keywords = finding.keywords.intersection(interpretation_tokens)
    return len(shared_keywords) >= 2 or bool(
        shared_keywords.intersection({"elevated", "high", "low", "positive", "abnormal"})
    )


def interpretation_matches_ecg(interpretation_text: str) -> bool:
    tokens = tokenize(interpretation_text)
    if "ecg" in tokens or "ekg" in tokens:
        return bool(tokens.intersection({"abnormal", "ischemic", "ischemia"})) or bool(
            {"st", "elevation"}.issubset(tokens)
        ) or bool({"st", "depression"}.issubset(tokens)) or bool({"wave", "changes"}.issubset(tokens))
    ecg_phrases = (
        "st elevation",
        "st depression",
        "t wave",
        "t-wave",
        "ischemic changes",
        "ischemia",
        "reciprocal changes",
        "reciprocal depression",
    )
    return any(phrase in interpretation_text for phrase in ecg_phrases)


def interpretation_calls_abnormal_result_normal(
    result: InvestigationResult,
    interpretation_text: str,
) -> bool:
    if not interpretation_text:
        return False
    if any(phrase in interpretation_text for phrase in ("not normal", "abnormal", "elevated", "high", "low")):
        return False
    normal_phrases = (
        "normal",
        "looks fine",
        "unremarkable",
        "non-contributory",
        "nothing concerning",
        "no acute findings",
    )
    if not any(phrase in interpretation_text for phrase in normal_phrases):
        return False
    test_tokens = tokens_for_test_name(result.investigation).union(
        tokens_for_test_name(str(result.structured_result.get("test", "")))
    )
    interpretation_tokens = tokenize(interpretation_text)
    return not test_tokens or bool(test_tokens.intersection(interpretation_tokens))


def keywords_for_abnormal_value(name: str, flag: object) -> set[str]:
    tokens = tokens_for_test_name(name)
    tokens.update(tokenize(str(flag)))
    tokens.update({"abnormal"})
    if normalize_text(str(flag)) in {"high", "mildly high"}:
        tokens.update({"elevated", "raised", "high"})
    if normalize_text(str(flag)) in {"low", "mildly low"}:
        tokens.update({"low", "decreased"})
    return tokens


def tokens_for_test_name(name: str) -> set[str]:
    tokens = tokenize(name)
    if "ecg" in tokens or "ekg" in tokens:
        tokens.update({"ecg", "ekg", "st"})
    if "troponin" in tokens:
        tokens.update({"troponin", "cardiac", "enzyme"})
    return tokens


def format_value_text(*, value: object, unit: object, reference: object) -> str:
    parts: list[str] = []
    if value is not None:
        parts.append(str(value))
    if unit:
        parts.append(str(unit))
    value_text = f" at {' '.join(parts)}" if parts else ""
    if reference:
        value_text += f" (reference {reference})"
    return value_text


def safe_investigation_name(result: InvestigationResult) -> str:
    return result.investigation.strip() or "investigation"


def build_investigation_interpretation_summary(
    *,
    results: list[InvestigationResult],
    interpretations: list[str],
    correctly_interpreted: list[str],
    missed_findings: list[str],
    incorrect_interpretations: list[str],
) -> str:
    if not results:
        return "No investigation results were available to interpret."
    if correctly_interpreted and not missed_findings and not incorrect_interpretations:
        return "Ordered investigation results were interpreted in a clinically reasonable way."
    if incorrect_interpretations:
        return "At least one abnormal investigation result was interpreted as normal or non-contributory."
    if missed_findings:
        if interpretations:
            return "Important abnormal investigation findings were not incorporated into the interpretation."
        return "Important ordered investigation results were not interpreted."
    return "Ordered investigations were normal or low-yield, so no interpretation penalty was applied."


def hide_diagnosis_in_investigation_feedback(
    feedback: InvestigationInterpretationFeedback,
    diagnosis: str,
) -> InvestigationInterpretationFeedback:
    return InvestigationInterpretationFeedback(
        correctly_interpreted_findings=[
            remove_hidden_diagnosis(text, diagnosis)
            for text in feedback.correctly_interpreted_findings
        ],
        missed_findings=[
            remove_hidden_diagnosis(text, diagnosis)
            for text in feedback.missed_findings
        ],
        incorrect_interpretations=[
            remove_hidden_diagnosis(text, diagnosis)
            for text in feedback.incorrect_interpretations
        ],
        summary=remove_hidden_diagnosis(feedback.summary, diagnosis),
    )


def remove_hidden_diagnosis(text: str, diagnosis: str) -> str:
    if not diagnosis.strip():
        return text
    return re.sub(re.escape(diagnosis), "the hidden diagnosis", text, flags=re.IGNORECASE)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def dedupe_findings(findings: list[InterpretableFinding]) -> list[InterpretableFinding]:
    seen: set[str] = set()
    deduped: list[InterpretableFinding] = []
    for finding in findings:
        if finding.feedback_text in seen:
            continue
        seen.add(finding.feedback_text)
        deduped.append(finding)
    return deduped


def score_performance(
    *,
    clinical_case: ClinicalCase,
    identified_symptoms: list[str],
    relevance: QuestionRelevanceEvaluation,
    diagnosis_guess: DiagnosisGuessEvaluation,
    missed_reasoning_steps: list[str] | None = None,
    investigation_feedback: InvestigationInterpretationFeedback | None = None,
) -> int:
    symptom_score = round((len(identified_symptoms) / len(clinical_case.symptoms)) * 45)
    total_questions = relevance.relevant_count + relevance.irrelevant_count
    relevance_score = 25 if total_questions == 0 else round((relevance.relevant_count / total_questions) * 25)
    diagnosis_score = 0
    if diagnosis_guess.provided:
        if diagnosis_guess.diagnosis_match_level == "correct":
            diagnosis_score = 20
        elif diagnosis_guess.diagnosis_match_level == "partial":
            diagnosis_score = 10
    missed_penalty = max(0, len(clinical_case.symptoms) - len(identified_symptoms) - 1) * 3
    irrelevant_penalty = relevance.irrelevant_count * 4
    reasoning_penalty = missed_reasoning_penalty(missed_reasoning_steps or [])
    investigation_adjustment = investigation_interpretation_score_adjustment(
        investigation_feedback
    )
    score = (
        symptom_score
        + relevance_score
        + diagnosis_score
        + 10
        + investigation_adjustment
        - missed_penalty
        - irrelevant_penalty
        - reasoning_penalty
    )
    return max(0, min(100, score))


def missed_reasoning_penalty(missed_reasoning_steps: list[str]) -> int:
    if not missed_reasoning_steps:
        return 0
    return min(25, 6 * len(missed_reasoning_steps))


def investigation_interpretation_score_adjustment(
    feedback: InvestigationInterpretationFeedback | None,
) -> int:
    if feedback is None:
        return 0
    reward = min(10, 5 * len(feedback.correctly_interpreted_findings))
    missed_penalty = min(18, 8 * len(feedback.missed_findings))
    incorrect_penalty = min(20, 10 * len(feedback.incorrect_interpretations))
    return reward - missed_penalty - incorrect_penalty


def build_feedback(
    *,
    identified_count: int,
    total_symptoms: int,
    relevance: QuestionRelevanceEvaluation,
    diagnosis_guess: DiagnosisGuessEvaluation,
) -> str:
    parts: list[str] = []
    if identified_count >= max(1, total_symptoms - 1):
        parts.append("Good coverage of the main patient complaints.")
    else:
        parts.append("Several important symptoms were not elicited.")
    if relevance.irrelevant_count:
        parts.append("Some questions were not clinically relevant.")
    else:
        parts.append("Questioning stayed focused on the presentation.")
    if diagnosis_guess.provided and diagnosis_guess.is_correct:
        parts.append("The diagnosis guess was appropriate.")
    elif diagnosis_guess.provided:
        parts.append("The diagnosis guess was not supported by the case.")
    else:
        parts.append("No diagnosis guess was documented.")
    return " ".join(parts)


def build_reasoning_feedback(
    *,
    case: ClinicalCase,
    questions: list[str],
    identified_symptoms: list[str],
    relevance: QuestionRelevanceEvaluation,
    diagnosis_guess: DiagnosisGuessEvaluation,
) -> ReasoningFeedback:
    missed_steps = missed_reasoning_steps(
        case=case,
        questions=questions,
        identified_symptoms=identified_symptoms,
        diagnosis_guess=diagnosis_guess,
    )
    return ReasoningFeedback(
        diagnostic_path=diagnostic_path_feedback(
            identified_count=len(identified_symptoms),
            total_symptoms=len(case.symptoms),
            relevance=relevance,
            diagnosis_guess=diagnosis_guess,
        ),
        question_order=question_order_feedback(questions),
        missed_reasoning_steps=missed_steps,
    )


def diagnostic_path_feedback(
    *,
    identified_count: int,
    total_symptoms: int,
    relevance: QuestionRelevanceEvaluation,
    diagnosis_guess: DiagnosisGuessEvaluation,
) -> str:
    if identified_count >= max(1, total_symptoms - 1) and relevance.irrelevant_count == 0:
        if diagnosis_guess.provided:
            return "The student followed a logical path from presenting complaints toward a working diagnosis."
        return "The student gathered a coherent clinical picture but did not document a working diagnosis."
    if relevance.irrelevant_count > relevance.relevant_count:
        return "The diagnostic path was scattered because many questions did not relate to the presentation."
    return "The student partially followed the clinical picture but left important reasoning gaps."


def question_order_feedback(questions: list[str]) -> str:
    if not questions:
        return "No question sequence was available to assess."
    early_questions = questions[: min(3, len(questions))]
    if any(is_diagnosis_guess_question(question) for question in early_questions):
        return "The student moved to a diagnosis guess before enough supporting details were established."
    if any(question_category(question) == "symptom" for question in early_questions):
        if any(question_category(question) in {"history", "progression", "vitals"} for question in questions[1:]):
            return "The order was useful: initial symptoms were explored before context and severity."
        return "The order started appropriately with symptoms but did not expand enough into context or severity."
    return "The order was less useful because early questions did not establish the main complaint clearly."


def missed_reasoning_steps(
    *,
    case: ClinicalCase,
    questions: list[str],
    identified_symptoms: list[str],
    diagnosis_guess: DiagnosisGuessEvaluation,
) -> list[str]:
    steps: list[str] = []
    categories = {question_category(question) for question in questions}
    missed_count = len(case.symptoms) - len(identified_symptoms)

    if missed_count > 0:
        steps.append("Elicit more of the key presenting symptoms before narrowing the diagnosis.")
    if "progression" not in categories:
        steps.append("Clarify timing and progression of symptoms.")
    if "history" not in categories:
        steps.append("Ask about relevant history or risk factors.")
    if "vitals" not in categories:
        steps.append("Assess severity using vital-sign or stability questions.")
    if not diagnosis_guess.provided:
        steps.append("State a working diagnosis or differential after gathering supporting evidence.")
    elif diagnosis_guess.is_correct is False:
        steps.append("Connect the diagnosis guess more closely to the elicited findings.")
    return steps[:4]


def question_category(question: str) -> str:
    tokens = tokenize(question)
    if is_diagnosis_guess_question(question):
        return "diagnosis"
    if tokens.intersection({"when", "started", "start", "progress", "worse", "better", "long", "time"}):
        return "progression"
    if tokens.intersection({"history", "risk", "diabetes", "medication", "allergy", "surgery", "pregnant"}):
        return "history"
    if tokens.intersection({"vitals", "pulse", "heart", "pressure", "oxygen", "breathing", "stable"}):
        return "vitals"
    if tokens.intersection(CLINICAL_QUESTION_TERMS):
        return "symptom"
    return "other"


def is_diagnosis_guess_question(question: str) -> bool:
    normalized = normalize_text(question)
    return any(
        term in normalized
        for term in (
            "diagnosis",
            "diagnosed",
            "could this be",
            "do they have",
            "does the patient have",
            "i think",
            "suspect",
            "guess",
        )
    )


def reasoning_feedback_has_low_confidence(
    evaluation: ClinicalEvaluation,
    *,
    question_count: int,
) -> bool:
    if question_count < 2:
        return True
    if evaluation.relevance_of_questions.relevant_count == 0:
        return True
    if (
        len(evaluation.key_symptoms_identified)
        + evaluation.relevance_of_questions.relevant_count
        <= 1
    ):
        return True
    return False


def build_learning_recommendations(
    *,
    current_score: int,
    total_symptoms: int,
    identified_symptoms: list[str],
    missed_symptoms: list[str],
    relevance: QuestionRelevanceEvaluation,
    diagnosis_guess: DiagnosisGuessEvaluation,
    reasoning_feedback: ReasoningFeedback,
    previous_results: list[ClinicalEvaluation | dict[str, Any] | str],
) -> LearningRecommendations:
    prior_evaluations = normalize_previous_evaluations(previous_results)
    recent_scores = [evaluation.overall_score for evaluation in prior_evaluations[-5:]]
    recent_average = sum(recent_scores) / len(recent_scores) if recent_scores else None
    repeated_weakness = most_common(
        (
            weakness
            for evaluation in prior_evaluations[-5:]
            for weakness in evaluation.reasoning_feedback.missed_reasoning_steps
        )
    )
    current_weakness = reasoning_feedback.missed_reasoning_steps[0] if reasoning_feedback.missed_reasoning_steps else None

    missed_ratio = len(missed_symptoms) / total_symptoms if total_symptoms else 0
    if recent_average is not None and recent_average < 50:
        primary_gap = "foundational data gathering"
        focus_next = "Use a consistent opening sequence before narrowing the problem."
    elif repeated_weakness:
        primary_gap = summarize_reasoning_gap(repeated_weakness)
        focus_next = action_for_reasoning_gap(repeated_weakness)
    elif missed_ratio >= 0.4:
        primary_gap = "key symptom elicitation"
        focus_next = "Ask broader symptom follow-up before moving to interpretation."
    elif relevance.irrelevant_count > 0:
        primary_gap = "question relevance"
        focus_next = "Tie each question to the patient presentation or severity assessment."
    elif not diagnosis_guess.provided:
        primary_gap = "diagnostic synthesis"
        focus_next = "State a working diagnosis or short differential after collecting evidence."
    elif diagnosis_guess.is_correct is False:
        primary_gap = "evidence-based diagnosis selection"
        focus_next = "Match the diagnosis guess to the findings already elicited."
    elif current_score > 75:
        primary_gap = "advanced prioritization"
        focus_next = "Practice prioritizing the highest-yield follow-up questions earlier."
    else:
        primary_gap = summarize_reasoning_gap(current_weakness) if current_weakness else "clinical reasoning consistency"
        focus_next = action_for_reasoning_gap(current_weakness) if current_weakness else "Keep using a structured symptom, timing, risk, and severity approach."

    return LearningRecommendations(
        primary_skill_gap=primary_gap,
        focus_next=focus_next,
        action_items=build_action_items(
            primary_gap=primary_gap,
            current_weakness=current_weakness,
            recent_average=recent_average,
            identified_count=len(identified_symptoms),
            total_symptoms=total_symptoms,
        ),
    )


def normalize_previous_evaluations(
    previous_results: list[ClinicalEvaluation | dict[str, Any] | str],
) -> list[ClinicalEvaluation]:
    evaluations: list[ClinicalEvaluation] = []
    for result in previous_results:
        if isinstance(result, ClinicalEvaluation):
            evaluations.append(result)
        elif isinstance(result, str):
            evaluations.append(ClinicalEvaluation.model_validate_json(result))
        else:
            evaluations.append(ClinicalEvaluation.model_validate(result))
    return evaluations


def summarize_reasoning_gap(reasoning_step: str | None) -> str:
    text = normalize_text(reasoning_step or "")
    if "timing" in text or "progression" in text:
        return "illness timeline clarification"
    if "history" in text or "risk" in text:
        return "risk factor exploration"
    if "severity" in text or "vital" in text or "stability" in text:
        return "severity assessment"
    if "working diagnosis" in text or "differential" in text:
        return "diagnostic synthesis"
    if "key presenting symptoms" in text:
        return "key symptom elicitation"
    return "clinical reasoning consistency"


def action_for_reasoning_gap(reasoning_step: str | None) -> str:
    text = normalize_text(reasoning_step or "")
    if "timing" in text or "progression" in text:
        return "Ask when symptoms started, how they changed, and what made them worse or better."
    if "history" in text or "risk" in text:
        return "Add one focused history or risk-factor question after the main complaint."
    if "severity" in text or "vital" in text or "stability" in text:
        return "Check severity early with stability, breathing, perfusion, or vital-sign questions."
    if "working diagnosis" in text or "differential" in text:
        return "Summarize the evidence and name a working diagnosis or differential."
    if "key presenting symptoms" in text:
        return "Ask at least two follow-up questions about the main complaint before interpreting."
    return "Use a repeatable structure: symptoms, timing, risk factors, severity, then synthesis."


def build_action_items(
    *,
    primary_gap: str,
    current_weakness: str | None,
    recent_average: float | None,
    identified_count: int,
    total_symptoms: int,
) -> list[str]:
    items = [f"Next case focus: {primary_gap}."]
    if current_weakness:
        items.append(action_for_reasoning_gap(current_weakness))
    elif identified_count < total_symptoms:
        items.append("Before guessing, ask one more question about an unaddressed symptom area.")
    else:
        items.append("Keep the same structure and make the next question more targeted.")
    if recent_average is not None and recent_average < 50:
        items.append("Prioritize consistency over speed until scores stabilize.")
    return items[:3]
