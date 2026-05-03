from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class StudentPerformanceSummary(StrictModel):
    """Stateless summary of recent student performance."""

    student_id: str
    recent_scores: list[int]
    average_score: float | None
    common_missed_symptoms: list[str]
    reasoning_weaknesses: list[str]
    recommended_difficulty: str

    @field_validator("recent_scores")
    @classmethod
    def validate_scores(cls, value: list[int]) -> list[int]:
        if any(score < 0 or score > 100 for score in value):
            raise ValueError("recent_scores must contain values between 0 and 100")
        return value

    @field_validator("recommended_difficulty")
    @classmethod
    def validate_difficulty(cls, value: str) -> str:
        if value not in {"easy", "medium", "hard"}:
            raise ValueError("recommended_difficulty must be easy, medium, or hard")
        return value
