from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from schemas.evaluation import ClinicalEvaluation
from utils.common import most_common_values

DEFAULT_SESSION_HISTORY_LIMIT = 5


class SessionContext(BaseModel):
    """Lightweight summary of prior session patterns for adaptation."""

    model_config = ConfigDict(extra="forbid")

    repeated_missed_symptoms: list[str]
    repeated_reasoning_weaknesses: list[str]
    learning_focus_areas: list[str]
    recent_scores: list[int]


def build_session_context(
    session_history: list[ClinicalEvaluation | dict[str, Any] | str] | None,
    *,
    history_limit: int = DEFAULT_SESSION_HISTORY_LIMIT,
) -> SessionContext:
    evaluations = normalize_session_history(session_history or [])
    recent = evaluations[-history_limit:] if history_limit > 0 else evaluations
    return SessionContext(
        repeated_missed_symptoms=most_common_values(
            (
                symptom
                for evaluation in recent
                for symptom in evaluation.key_symptoms_missed
            ),
            limit=3,
        ),
        repeated_reasoning_weaknesses=most_common_values(
            (
                weakness
                for evaluation in recent
                for weakness in evaluation.reasoning_feedback.missed_reasoning_steps
            ),
            limit=3,
        ),
        learning_focus_areas=most_common_values(
            (evaluation.learning_recommendations.primary_skill_gap for evaluation in recent),
            limit=3,
        ),
        recent_scores=[evaluation.overall_score for evaluation in recent],
    )


def normalize_session_history(
    session_history: list[ClinicalEvaluation | dict[str, Any] | str],
) -> list[ClinicalEvaluation]:
    evaluations: list[ClinicalEvaluation] = []
    for result in session_history:
        if isinstance(result, ClinicalEvaluation):
            evaluations.append(result)
        elif isinstance(result, str):
            evaluations.append(ClinicalEvaluation.model_validate_json(result))
        else:
            evaluations.append(ClinicalEvaluation.model_validate(result))
    return evaluations
