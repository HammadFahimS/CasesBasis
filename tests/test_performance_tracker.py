from __future__ import annotations

import pytest

from backend.performance_tracker import StudentPerformanceTracker, recommend_difficulty
from schemas.evaluation import (
    ClinicalEvaluation,
    DiagnosisGuessEvaluation,
    LearningRecommendations,
    QuestionRelevanceEvaluation,
    ReasoningFeedback,
)


def make_evaluation(
    score: int,
    *,
    missed_symptoms: list[str] | None = None,
    missed_steps: list[str] | None = None,
) -> ClinicalEvaluation:
    return ClinicalEvaluation(
        correct_diagnosis_guess=DiagnosisGuessEvaluation(
            provided=False,
            guess=None,
            is_correct=None,
        ),
        key_symptoms_identified=[],
        key_symptoms_missed=missed_symptoms or [],
        relevance_of_questions=QuestionRelevanceEvaluation(
            relevant_count=1,
            irrelevant_count=0,
            relevant_questions=["do you have fever?"],
            irrelevant_questions=[],
        ),
        overall_score=score,
        brief_feedback="Feedback.",
        reasoning_feedback=ReasoningFeedback(
            diagnostic_path="Path.",
            question_order="Order.",
            missed_reasoning_steps=missed_steps or [],
        ),
        learning_recommendations=LearningRecommendations(
            primary_skill_gap="clinical reasoning consistency",
            focus_next="Keep a structured approach.",
            action_items=["Use symptoms, timing, risk, severity, then synthesis."],
        ),
    )


def test_recommend_difficulty_thresholds() -> None:
    assert recommend_difficulty(76) == "hard"
    assert recommend_difficulty(75) == "medium"
    assert recommend_difficulty(50) == "medium"
    assert recommend_difficulty(49) == "easy"
    assert recommend_difficulty(None) == "medium"


def test_tracker_summarizes_recent_scores_and_common_misses() -> None:
    tracker = StudentPerformanceTracker()
    results = [
        make_evaluation(80, missed_symptoms=["fever"], missed_steps=["Ask about timing."]),
        make_evaluation(90, missed_symptoms=["fever", "dizziness"], missed_steps=["Ask about timing."]),
        make_evaluation(70, missed_symptoms=["dizziness"], missed_steps=["Check vitals."]),
    ]

    summary = tracker.summarize(student_id="student-1", previous_results=results)

    assert summary.student_id == "student-1"
    assert summary.recent_scores == [80, 90, 70]
    assert summary.average_score == 80
    assert summary.recommended_difficulty == "hard"
    assert summary.common_missed_symptoms[:2] == ["fever", "dizziness"]
    assert summary.reasoning_weaknesses[0] == "Ask about timing."


def test_tracker_accepts_dict_and_json_results() -> None:
    tracker = StudentPerformanceTracker()
    evaluation = make_evaluation(45, missed_symptoms=["confusion"], missed_steps=["Check vitals."])
    results = [evaluation.model_dump(), evaluation.model_dump_json()]

    summary = tracker.summarize(student_id="student-2", previous_results=results)

    assert summary.recent_scores == [45, 45]
    assert summary.average_score == 45
    assert summary.recommended_difficulty == "easy"
    assert summary.common_missed_symptoms == ["confusion"]


def test_tracker_limits_recent_history() -> None:
    tracker = StudentPerformanceTracker()
    results = [make_evaluation(score) for score in [10, 20, 80, 90]]

    summary = tracker.summarize(
        student_id="student-3",
        previous_results=results,
        history_limit=2,
    )

    assert summary.recent_scores == [80, 90]
    assert summary.average_score == 85
    assert summary.recommended_difficulty == "hard"


def test_tracker_requires_student_id() -> None:
    tracker = StudentPerformanceTracker()

    with pytest.raises(ValueError):
        tracker.summarize(student_id=" ", previous_results=[])
