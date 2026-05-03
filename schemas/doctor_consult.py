from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class DoctorConsultResponse(StrictModel):
    """Structured guidance returned by the doctor consultation service."""

    urgency_level: Literal["routine", "urgent", "critical"]
    concise_advice: str
    recommended_next_steps: list[str]
    missing_information: list[str]
    safety_warnings: list[str]
    should_escalate: bool

    @field_validator("concise_advice")
    @classmethod
    def require_advice(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("concise_advice is required")
        return cleaned

    @field_validator("recommended_next_steps", "missing_information", "safety_warnings")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned
