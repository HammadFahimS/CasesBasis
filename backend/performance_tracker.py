from __future__ import annotations

from typing import Any

from schemas.evaluation import ClinicalEvaluation
from schemas.performance import StudentPerformanceSummary
from utils.common import most_common_values

DEFAULT_HISTORY_LIMIT = 5


class StudentPerformanceTracker:
    """Summarizes prior evaluation results and recommends adaptive difficulty."""

    def summarize(
        self,
        *,
        student_id: str,
        previous_results: list[ClinicalEvaluation | dict[str, Any] | str],
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> StudentPerformanceSummary:
        if not student_id.strip():
            raise ValueError("student_id is required")
        evaluations = [normalize_evaluation(result) for result in previous_results]
        recent_evaluations = evaluations[-history_limit:] if history_limit > 0 else evaluations
        recent_scores = [evaluation.overall_score for evaluation in recent_evaluations]
        average_score = average(recent_scores)

        return StudentPerformanceSummary(
            student_id=student_id,
            recent_scores=recent_scores,
            average_score=average_score,
            common_missed_symptoms=most_common_values(
                (
                    symptom
                    for evaluation in recent_evaluations
                    for symptom in evaluation.key_symptoms_missed
                ),
                limit=5,
            ),
            reasoning_weaknesses=most_common_values(
                (
                    weakness
                    for evaluation in recent_evaluations
                    for weakness in evaluation.reasoning_feedback.missed_reasoning_steps
                ),
                limit=5,
            ),
            recommended_difficulty=recommend_difficulty(average_score),
        )


def normalize_evaluation(result: ClinicalEvaluation | dict[str, Any] | str) -> ClinicalEvaluation:
    if isinstance(result, ClinicalEvaluation):
        return result
    if isinstance(result, str):
        return ClinicalEvaluation.model_validate_json(result)
    return ClinicalEvaluation.model_validate(result)


def average(scores: list[int]) -> float | None:
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def recommend_difficulty(average_score: float | None) -> str:
    if average_score is None:
        return "medium"
    if average_score > 75:
        return "hard"
    if average_score < 50:
        return "easy"
    return "medium"
