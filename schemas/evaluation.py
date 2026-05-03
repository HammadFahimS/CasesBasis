from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class DiagnosisGuessEvaluation(StrictModel):
    """Assessment of a student's diagnosis guess."""

    provided: bool
    guess: str | None
    is_correct: bool | None
    diagnosis_match_level: Literal["correct", "partial", "incorrect"] | None = None


class QuestionRelevanceEvaluation(StrictModel):
    """Counts and examples of relevant and irrelevant questions."""

    relevant_count: int
    irrelevant_count: int
    relevant_questions: list[str]
    irrelevant_questions: list[str]


class ReasoningFeedback(StrictModel):
    """Structured feedback on diagnostic reasoning process."""

    diagnostic_path: str
    question_order: str
    missed_reasoning_steps: list[str]


class LearningRecommendations(StrictModel):
    """Targeted next-step learning guidance for the student."""

    primary_skill_gap: str
    focus_next: str
    action_items: list[str]


class InvestigationInterpretationFeedback(StrictModel):
    """Feedback on interpretation of ordered investigation results."""

    correctly_interpreted_findings: list[str] = Field(default_factory=list)
    missed_findings: list[str] = Field(default_factory=list)
    incorrect_interpretations: list[str] = Field(default_factory=list)
    summary: str = "No investigation interpretation was provided."


class ClinicalEvaluation(StrictModel):
    """Complete structured evaluation for a student interaction."""

    correct_diagnosis_guess: DiagnosisGuessEvaluation
    key_symptoms_identified: list[str]
    key_symptoms_missed: list[str]
    relevance_of_questions: QuestionRelevanceEvaluation
    overall_score: int
    brief_feedback: str
    reasoning_feedback: ReasoningFeedback
    learning_recommendations: LearningRecommendations
    investigation_interpretation_feedback: InvestigationInterpretationFeedback = Field(
        default_factory=InvestigationInterpretationFeedback
    )

    @field_validator("overall_score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("overall_score must be between 0 and 100")
        return value
