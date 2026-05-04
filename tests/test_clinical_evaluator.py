from __future__ import annotations

from backend.investigations import InvestigationService
from backend.clinical_evaluator import ClinicalEvaluator
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation, LearningRecommendations, ReasoningFeedback
from schemas.investigation import InvestigationResult
from tests.test_clinical_case_schema import sample_case
from tests.test_investigations import mi_case, pneumonia_case


def build_case() -> ClinicalCase:
    return ClinicalCase.model_validate(sample_case())


class StubReasoningReviewer:
    def __init__(self) -> None:
        self.calls = 0
        self.last_evaluation: ClinicalEvaluation | None = None

    def review(
        self,
        *,
        case: ClinicalCase,
        evaluation: ClinicalEvaluation,
        student_questions: list[str],
        patient_responses: list[str],
    ) -> ReasoningFeedback:
        self.calls += 1
        self.last_evaluation = evaluation
        return ReasoningFeedback(
            diagnostic_path="Reviewed diagnostic path with a concise strength and weakness.",
            question_order="Reviewed question order with clearer sequencing feedback.",
            missed_reasoning_steps=["Reviewed missed reasoning step."],
        )


def test_evaluator_rewards_relevant_questions_and_correct_guess() -> None:
    evaluator = ClinicalEvaluator()
    questions = [
        "Do you have fever?",
        "Are you confused or dizzy?",
        "Have you been urinating less?",
        "Do they have septic shock?",
    ]
    responses = [
        "Yes, I feel feverish.",
        "I think I am confused and dizzy.",
        "I have barely been peeing.",
        "I'm not sure.",
    ]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert evaluation.correct_diagnosis_guess.guess == "septic shock"
    assert "fever" in evaluation.key_symptoms_identified
    assert evaluation.relevance_of_questions.irrelevant_count == 0
    assert evaluation.overall_score >= 60
    assert evaluation.reasoning_feedback.diagnostic_path
    assert "appropriately with symptoms" in evaluation.reasoning_feedback.question_order
    assert evaluation.learning_recommendations.primary_skill_gap
    assert len(evaluation.learning_recommendations.action_items) <= 3


def test_evaluator_accepts_common_diagnosis_alias() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()

    evaluation = evaluator.evaluate(
        case,
        [
            "Can you describe the chest pressure?",
            "Does it go to your left arm?",
            "Diagnosis guess: Heart Attack",
        ],
        [
            "It feels like heavy chest pressure.",
            "Yes, it goes to my left arm.",
            "I'm not sure.",
        ],
    )

    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.guess == "Heart Attack"
    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert evaluation.correct_diagnosis_guess.diagnosis_match_level == "correct"
    assert evaluation.overall_score > 0


def test_evaluator_preserves_exact_parenthetical_diagnosis_guess() -> None:
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        ClinicalCase.model_validate(
            {
                "diagnosis": "diabetic ketoacidosis",
                "patient": {
                    "age": 19,
                    "gender": "female",
                    "history": ["type 1 diabetes"],
                },
                "symptoms": [
                    "fatigue",
                    "polyuria",
                    "fruity-smelling breath",
                ],
                "vitals": {
                    "heart_rate": 112,
                    "blood_pressure": "104/68",
                    "respiratory_rate": 24,
                    "oxygen_saturation": 98,
                },
                "progression": [
                    {
                        "time_min": 0,
                        "description": "Fatigue and thirst worsen over two days.",
                    }
                ],
                "risk_factors": ["missed insulin"],
                "exam_findings": ["dry mucous membranes"],
                "outcome": {
                    "if_treated": "Improves with fluids and insulin.",
                    "if_untreated": "Acidosis and dehydration worsen.",
                },
            }
        ),
        ["Diagnosis guess: Diabetic Ketoacidosis (DKA)"],
        ["I'm not sure."],
    )

    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.guess == "Diabetic Ketoacidosis (DKA)"
    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert evaluation.correct_diagnosis_guess.diagnosis_match_level == "correct"


def test_evaluator_treats_symptom_guess_as_no_diagnosis() -> None:
    evaluator = ClinicalEvaluator()

    for symptom_guess in ("chest pain", "fever", "shortness of breath"):
        evaluation = evaluator.evaluate(
            mi_case(),
            [f"Diagnosis guess: {symptom_guess}"],
            ["I'm not sure."],
        )

        assert evaluation.correct_diagnosis_guess.provided is False
        assert evaluation.correct_diagnosis_guess.guess is None
        assert evaluation.correct_diagnosis_guess.is_correct is None
        assert evaluation.correct_diagnosis_guess.diagnosis_match_level is None
        assert "diagnosis guess was not supported" not in evaluation.brief_feedback.lower()


def test_evaluator_does_not_extract_diagnosis_from_symptom_history_question() -> None:
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        mi_case(),
        [
            "Have you experienced this type of chest pain before, and does it radiate to any other parts of your body?",
        ],
        ["It feels like pressure in my chest and goes toward my left arm."],
    )

    assert evaluation.correct_diagnosis_guess.provided is False
    assert evaluation.correct_diagnosis_guess.guess is None
    assert evaluation.correct_diagnosis_guess.is_correct is None
    assert evaluation.correct_diagnosis_guess.diagnosis_match_level is None


def test_evaluator_accepts_real_diagnosis_guess_as_valid() -> None:
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        mi_case(),
        ["Diagnosis guess: myocardial infarction"],
        ["I'm not sure."],
    )

    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.guess == "myocardial infarction"
    assert evaluation.correct_diagnosis_guess.is_correct is True


def test_evaluator_marks_related_diagnosis_as_partial() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()

    partial = evaluator.evaluate(
        case,
        ["Diagnosis guess: acute coronary syndrome"],
        ["I'm not sure."],
    )
    incorrect = evaluator.evaluate(
        case,
        ["Diagnosis guess: GERD"],
        ["I'm not sure."],
    )

    assert partial.correct_diagnosis_guess.guess == "acute coronary syndrome"
    assert partial.correct_diagnosis_guess.is_correct is False
    assert partial.correct_diagnosis_guess.diagnosis_match_level == "partial"
    assert incorrect.correct_diagnosis_guess.diagnosis_match_level == "incorrect"
    assert partial.overall_score > incorrect.overall_score


def test_evaluator_diagnosis_match_levels_for_stroke_and_pe_partials() -> None:
    evaluator = ClinicalEvaluator()
    stroke_case = pneumonia_case().model_copy(update={"diagnosis": "stroke"})
    pe_case = pneumonia_case().model_copy(update={"diagnosis": "pulmonary embolism"})

    stroke_correct = evaluator.evaluate(
        stroke_case,
        ["Diagnosis guess: CVA"],
        ["I'm not sure."],
    )
    stroke_partial = evaluator.evaluate(
        stroke_case,
        ["Diagnosis guess: TIA"],
        ["I'm not sure."],
    )
    pe_partial = evaluator.evaluate(
        pe_case,
        ["Diagnosis guess: DVT"],
        ["I'm not sure."],
    )

    assert stroke_correct.correct_diagnosis_guess.diagnosis_match_level == "correct"
    assert stroke_correct.correct_diagnosis_guess.is_correct is True
    assert stroke_partial.correct_diagnosis_guess.diagnosis_match_level == "partial"
    assert pe_partial.correct_diagnosis_guess.diagnosis_match_level == "partial"


def test_correct_alias_with_limited_questioning_gets_partial_credit_not_zero() -> None:
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        mi_case(),
        ["Diagnosis guess: heart attack"],
        ["I'm not sure."],
    )

    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert evaluation.correct_diagnosis_guess.diagnosis_match_level == "correct"
    assert 0 < evaluation.overall_score < 70


def test_symptom_not_asked_or_revealed_remains_missed() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()

    evaluation = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "Are you short of breath?"],
        ["It feels heavy in my chest.", "A little short of breath."],
    )

    assert "nausea" in case.symptoms
    assert "nausea" not in evaluation.key_symptoms_identified
    assert "nausea" in evaluation.key_symptoms_missed


def test_evaluator_score_cannot_be_100_with_missed_reasoning_steps() -> None:
    evaluator = ClinicalEvaluator()
    questions = [
        "Do you have fever?",
        "Are you confused or dizzy?",
        "Have you been urinating less?",
        "Do you have shortness of breath?",
        "Could this be septic shock?",
    ]
    responses = [
        "Yes, I feel feverish.",
        "Yes, confused and dizzy.",
        "Yes, much less than usual.",
        "Yes.",
        "I'm not sure.",
    ]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert evaluation.reasoning_feedback.missed_reasoning_steps
    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert not evaluation.key_symptoms_missed
    assert evaluation.overall_score < 100


def test_evaluator_penalizes_irrelevant_questions_and_missed_symptoms() -> None:
    evaluator = ClinicalEvaluator()
    questions = [
        "What is your favorite movie?",
        "Do you like coffee?",
        "Could this be meningitis?",
    ]
    responses = ["I don't know.", "Maybe.", "I'm not sure."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.is_correct is False
    assert evaluation.key_symptoms_identified == []
    assert len(evaluation.key_symptoms_missed) == len(build_case().symptoms)
    assert evaluation.relevance_of_questions.irrelevant_count >= 2
    assert evaluation.overall_score < 50


def test_evaluator_does_not_reveal_diagnosis_without_student_guess() -> None:
    evaluator = ClinicalEvaluator()
    questions = ["Do you have fever?", "Are you dizzy?"]
    responses = ["Yes, I feel feverish.", "Yes, very dizzy."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)
    payload = evaluation.model_dump()

    assert evaluation.correct_diagnosis_guess.provided is False
    assert evaluation.correct_diagnosis_guess.guess is None
    assert "septic shock" not in str(payload).lower()


def test_evaluator_accepts_raw_case_json() -> None:
    evaluator = ClinicalEvaluator()
    raw_case_json = build_case().model_dump_json()

    evaluation = evaluator.evaluate(
        raw_case_json,
        ["Do you have shortness of breath?"],
        ["Yes, I feel short of breath."],
    )

    assert "shortness of breath" in evaluation.key_symptoms_identified


def test_evaluator_reasoning_feedback_flags_premature_diagnosis_guess() -> None:
    evaluator = ClinicalEvaluator()
    questions = [
        "Could this be septic shock?",
        "Do you have fever?",
    ]
    responses = ["I'm not sure.", "Yes, I feel feverish."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert "before enough supporting details" in evaluation.reasoning_feedback.question_order


def test_evaluator_reasoning_feedback_identifies_missed_steps() -> None:
    evaluator = ClinicalEvaluator()
    questions = ["Do you have fever?"]
    responses = ["Yes."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    missed_steps = evaluation.reasoning_feedback.missed_reasoning_steps
    assert "Clarify timing and progression of symptoms." in missed_steps
    assert "Ask about relevant history or risk factors." in missed_steps
    assert "Assess severity using vital-sign or stability questions." in missed_steps


def test_evaluator_reasoning_feedback_does_not_reveal_hidden_diagnosis() -> None:
    evaluator = ClinicalEvaluator()
    questions = ["Do you have fever?", "When did it start?"]
    responses = ["Yes.", "Earlier today."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert "septic shock" not in evaluation.reasoning_feedback.model_dump_json().lower()


def test_evaluator_learning_recommendations_do_not_reveal_hidden_diagnosis() -> None:
    evaluator = ClinicalEvaluator()
    questions = ["Do you have fever?", "When did it start?"]
    responses = ["Yes.", "Earlier today."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert "septic shock" not in evaluation.learning_recommendations.model_dump_json().lower()


def test_evaluator_learning_recommendations_target_symptom_elicitation() -> None:
    evaluator = ClinicalEvaluator()
    questions = ["Do you have fever?"]
    responses = ["Yes."]

    evaluation = evaluator.evaluate(build_case(), questions, responses)

    assert evaluation.learning_recommendations.primary_skill_gap == "key symptom elicitation"
    assert "Ask broader symptom follow-up" in evaluation.learning_recommendations.focus_next


def test_evaluator_learning_recommendations_use_prior_reasoning_weaknesses() -> None:
    prior = ClinicalEvaluation(
        correct_diagnosis_guess=prior_guess(None),
        key_symptoms_identified=["fever"],
        key_symptoms_missed=[],
        relevance_of_questions=prior_relevance(),
        overall_score=65,
        brief_feedback="Moderate score.",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="Partial path.",
            question_order="Partial order.",
            missed_reasoning_steps=["Clarify timing and progression of symptoms."],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="illness timeline clarification",
            focus_next="Ask about timing.",
            action_items=["Practice timeline questions."],
        ),
    )
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        build_case(),
        [
            "Do you have fever?",
            "Are you confused?",
            "Have you been urinating less?",
            "What is your oxygen level?",
        ],
        ["Yes.", "Yes.", "Yes.", "I am not sure."],
        previous_results=[prior.model_dump(), prior.model_dump_json()],
    )

    assert evaluation.learning_recommendations.primary_skill_gap == "illness timeline clarification"
    assert "when symptoms started" in evaluation.learning_recommendations.focus_next


def test_evaluator_learning_recommendations_accept_session_history() -> None:
    prior = ClinicalEvaluation(
        correct_diagnosis_guess=prior_guess(None),
        key_symptoms_identified=["fever"],
        key_symptoms_missed=[],
        relevance_of_questions=prior_relevance(),
        overall_score=65,
        brief_feedback="Moderate score.",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="Partial path.",
            question_order="Partial order.",
            missed_reasoning_steps=["Assess severity using vital-sign or stability questions."],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="severity assessment",
            focus_next="Check vitals.",
            action_items=["Practice severity questions."],
        ),
    )
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        build_case(),
        ["Do you have fever?", "Are you confused?", "When did it start?"],
        ["Yes.", "Yes.", "Today."],
        session_history=[prior.model_dump()],
    )

    assert evaluation.learning_recommendations.primary_skill_gap == "severity assessment"
    assert "severity early" in evaluation.learning_recommendations.focus_next


def test_evaluator_learning_recommendations_use_recent_low_performance() -> None:
    low_result = ClinicalEvaluation(
        correct_diagnosis_guess=prior_guess(False),
        key_symptoms_identified=[],
        key_symptoms_missed=["fever"],
        relevance_of_questions=prior_relevance(),
        overall_score=35,
        brief_feedback="Low score.",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="Scattered.",
            question_order="Unclear.",
            missed_reasoning_steps=["Ask about relevant history or risk factors."],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="risk factor exploration",
            focus_next="Ask focused history.",
            action_items=["Practice history questions."],
        ),
    )
    evaluator = ClinicalEvaluator()

    evaluation = evaluator.evaluate(
        build_case(),
        ["Do you have fever?", "Are you confused?", "When did it start?"],
        ["Yes.", "Yes.", "Today."],
        previous_results=[low_result],
    )

    assert evaluation.learning_recommendations.primary_skill_gap == "foundational data gathering"
    assert "consistent opening sequence" in evaluation.learning_recommendations.focus_next


def test_evaluator_scores_correct_ecg_interpretation() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")

    evaluation = evaluator.evaluate(
        case,
        [
            "Can you describe the chest pressure?",
            "Does it go to the left arm?",
            "When did it start?",
        ],
        [
            "It is heavy pressure.",
            "It goes to my left arm.",
            "About 30 minutes ago.",
        ],
        investigations_ordered=[ecg],
        student_interpretations=[
            "The ECG is abnormal with ST elevation and reciprocal changes.",
        ],
    )

    feedback = evaluation.investigation_interpretation_feedback
    assert feedback.correctly_interpreted_findings
    assert any("ST elevation" in finding for finding in feedback.correctly_interpreted_findings)
    assert not feedback.incorrect_interpretations
    assert "clinically reasonable" in feedback.summary


def test_evaluator_does_not_credit_vague_elevated_levels_as_ecg_interpretation() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")

    evaluation = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=["There are elevated levels."],
    )

    feedback = evaluation.investigation_interpretation_feedback
    assert not feedback.correctly_interpreted_findings
    assert feedback.missed_findings


def test_evaluator_ties_interpretations_to_correct_ordered_test() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")
    troponin = InvestigationService().get_result(case, "troponin")

    ecg_only = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=["There are elevated levels."],
    )
    troponin_ordered = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[troponin],
        student_interpretations=["There are elevated levels."],
    )

    assert not ecg_only.investigation_interpretation_feedback.correctly_interpreted_findings
    assert troponin_ordered.investigation_interpretation_feedback.correctly_interpreted_findings

    both_ordered = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg, troponin],
        student_interpretations=["The troponin has elevated levels."],
    )

    feedback = both_ordered.investigation_interpretation_feedback
    assert any("troponin" in finding.lower() for finding in feedback.correctly_interpreted_findings)
    assert not any("ST elevation" in finding for finding in feedback.correctly_interpreted_findings)


def test_evaluator_penalizes_ecg_called_normal() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")

    correct = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=["The ECG has ST elevation and looks abnormal."],
    )
    incorrect = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=["The ECG is normal and not concerning."],
    )
    negated_abnormal = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=[
            "The ECG shows normal sinus rhythm with no abnormalities noted. There are no signs of ischemia or infarction.",
        ],
    )
    negated_st_changes = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=[
            "The ECG shows left ventricular hypertrophy. There are no significant ST-T wave changes.",
        ],
    )

    assert incorrect.investigation_interpretation_feedback.incorrect_interpretations
    assert negated_abnormal.investigation_interpretation_feedback.incorrect_interpretations
    assert not negated_abnormal.investigation_interpretation_feedback.correctly_interpreted_findings
    assert negated_st_changes.investigation_interpretation_feedback.incorrect_interpretations
    assert not negated_st_changes.investigation_interpretation_feedback.correctly_interpreted_findings
    assert incorrect.overall_score < correct.overall_score


def test_evaluator_penalizes_unrelated_ecg_interpretation() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    ecg = InvestigationService().get_result(case, "ECG")

    evaluation = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[ecg],
        student_interpretations=[
            "The ECG shows a regular rhythm. The U waves are prominent, suggesting hypokalemia, and the QT interval is prolonged.",
        ],
    )

    feedback = evaluation.investigation_interpretation_feedback
    assert feedback.incorrect_interpretations
    assert not feedback.correctly_interpreted_findings
    assert feedback.missed_findings


def test_evaluator_scores_correct_troponin_interpretation() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    troponin = InvestigationService().get_result(case, "troponin")

    evaluation = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "Are you short of breath?"],
        ["Heavy pressure.", "A little."],
        investigations_ordered=[troponin],
        student_interpretations=["The troponin is elevated above the reference range."],
    )

    feedback = evaluation.investigation_interpretation_feedback
    assert any("troponin" in finding.lower() for finding in feedback.correctly_interpreted_findings)
    assert not feedback.missed_findings
    assert not feedback.incorrect_interpretations


def test_evaluator_penalizes_ignored_abnormal_troponin() -> None:
    evaluator = ClinicalEvaluator()
    case = mi_case()
    troponin = InvestigationService().get_result(case, "troponin")

    interpreted = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "Are you short of breath?"],
        ["Heavy pressure.", "A little."],
        investigations_ordered=[troponin],
        student_interpretations=["The troponin is elevated above the reference range."],
    )
    ignored = evaluator.evaluate(
        case,
        ["Can you describe the chest pressure?", "Are you short of breath?"],
        ["Heavy pressure.", "A little."],
        investigations_ordered=[troponin],
        student_interpretations=[],
    )

    assert ignored.investigation_interpretation_feedback.missed_findings
    assert "not interpreted" in ignored.investigation_interpretation_feedback.summary
    assert ignored.overall_score < interpreted.overall_score


def test_evaluator_does_not_penalize_unrelated_normal_test() -> None:
    evaluator = ClinicalEvaluator()
    normal_ct = InvestigationResult(
        investigation="CT abdomen",
        status="available",
        structured_result={
            "test": "CT abdomen",
            "findings": ["No acute abnormality identified"],
            "values": [],
            "interpretation_hint": "No abdominal finding explains the current symptoms.",
        },
        summary="CT abdomen is non-contributory.",
        visual_payload=None,
    )

    evaluation = evaluator.evaluate(
        mi_case(),
        ["Can you describe the chest pressure?", "When did it start?"],
        ["Heavy pressure.", "About 30 minutes ago."],
        investigations_ordered=[normal_ct],
        student_interpretations=[],
    )

    feedback = evaluation.investigation_interpretation_feedback
    assert feedback.correctly_interpreted_findings == []
    assert feedback.missed_findings == []
    assert feedback.incorrect_interpretations == []
    assert "no interpretation penalty" in feedback.summary


def test_investigation_interpretation_feedback_does_not_leak_hidden_diagnosis() -> None:
    evaluator = ClinicalEvaluator()
    revealing_result = InvestigationResult(
        investigation="Special blood test",
        status="available",
        structured_result={
            "test": "community-acquired pneumonia marker",
            "findings": ["Elevated community-acquired pneumonia marker"],
            "values": [
                {
                    "name": "community-acquired pneumonia marker",
                    "value": 12,
                    "unit": "units",
                    "reference_range": "0-3 units",
                    "flag": "high",
                }
            ],
            "interpretation_hint": "Abnormal marker.",
        },
        summary="Marker is elevated.",
        visual_payload=None,
    )

    evaluation = evaluator.evaluate(
        pneumonia_case(),
        ["Do you have cough?", "Do you have fever?"],
        ["Yes.", "Yes."],
        investigations_ordered=[revealing_result],
        student_interpretations=["The marker is elevated."],
    )

    assert "community-acquired pneumonia" not in (
        evaluation.investigation_interpretation_feedback.model_dump_json().lower()
    )


def test_optional_reasoning_review_is_not_used_when_disabled() -> None:
    reviewer = StubReasoningReviewer()
    evaluator = ClinicalEvaluator(reasoning_reviewer=reviewer)

    evaluation = evaluator.evaluate(
        build_case(),
        ["Do you have fever?", "Are you confused?"],
        ["Yes.", "Yes."],
    )

    assert reviewer.calls == 0
    assert "Reviewed" not in evaluation.reasoning_feedback.diagnostic_path


def test_optional_reasoning_review_triggers_for_borderline_score_only_changes_reasoning() -> None:
    reviewer = StubReasoningReviewer()
    deterministic = ClinicalEvaluator().evaluate(
        build_case(),
        ["Do you have fever?", "Are you confused?", "Have you been urinating less?", "Could this be septic shock?"],
        ["Yes.", "Yes.", "Yes.", "I'm not sure."],
    )
    evaluator = ClinicalEvaluator(enable_reasoning_review=True, reasoning_reviewer=reviewer)

    reviewed = evaluator.evaluate(
        build_case(),
        ["Do you have fever?", "Are you confused?", "Have you been urinating less?", "Could this be septic shock?"],
        ["Yes.", "Yes.", "Yes.", "I'm not sure."],
    )

    assert 40 <= deterministic.overall_score <= 70
    assert reviewer.calls == 1
    assert reviewed.overall_score == deterministic.overall_score
    assert reviewed.correct_diagnosis_guess == deterministic.correct_diagnosis_guess
    assert reviewed.key_symptoms_identified == deterministic.key_symptoms_identified
    assert reviewed.key_symptoms_missed == deterministic.key_symptoms_missed
    assert reviewed.relevance_of_questions == deterministic.relevance_of_questions
    assert reviewed.brief_feedback == deterministic.brief_feedback
    assert reviewed.reasoning_feedback.diagnostic_path.startswith("Reviewed")


def test_optional_reasoning_review_triggers_for_low_confidence_feedback() -> None:
    reviewer = StubReasoningReviewer()
    evaluator = ClinicalEvaluator(enable_reasoning_review=True, reasoning_reviewer=reviewer)

    reviewed = evaluator.evaluate(build_case(), ["Do you have fever?"], ["Yes."])

    assert reviewed.overall_score < 40
    assert reviewer.calls == 1
    assert reviewed.reasoning_feedback.missed_reasoning_steps == ["Reviewed missed reasoning step."]


def prior_guess(is_correct: bool | None) -> object:
    from schemas.evaluation import DiagnosisGuessEvaluation

    return DiagnosisGuessEvaluation(
        provided=is_correct is not None,
        guess="prior guess" if is_correct is not None else None,
        is_correct=is_correct,
    )


def prior_relevance() -> object:
    from schemas.evaluation import QuestionRelevanceEvaluation

    return QuestionRelevanceEvaluation(
        relevant_count=1,
        irrelevant_count=1,
        relevant_questions=["do you have fever?"],
        irrelevant_questions=["favorite movie?"],
    )
