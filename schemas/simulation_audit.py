from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class SimulationAuditFinding(StrictModel):
    """One QA finding from an internal simulation audit."""

    category: Literal["confirmed_bug", "possible_model_quality_issue", "expected_behavior"]
    severity: Literal["low", "medium", "high"]
    issue: str
    evidence: str
    suggested_fix: str

    @field_validator("issue", "evidence", "suggested_fix")
    @classmethod
    def require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("audit text fields must not be empty")
        return cleaned


class SimulationAudit(StrictModel):
    """Internal QA review of one completed simulation run."""

    overall_quality: Literal["pass", "warning", "fail"]
    findings: list[SimulationAuditFinding]
    should_fix: bool

    @model_validator(mode="after")
    def normalize_should_fix(self) -> "SimulationAudit":
        has_confirmed_bug = any(finding.category == "confirmed_bug" for finding in self.findings)
        self.should_fix = has_confirmed_bug
        return self
