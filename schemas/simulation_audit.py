from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


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

    @field_validator("should_fix")
    @classmethod
    def require_fix_for_confirmed_bugs(cls, value: bool, info: object) -> bool:
        data = getattr(info, "data", {})
        findings = data.get("findings", []) if isinstance(data, dict) else []
        has_confirmed_bug = any(
            getattr(finding, "category", None) == "confirmed_bug"
            for finding in findings
        )
        if has_confirmed_bug and not value:
            raise ValueError("should_fix must be true when confirmed bugs are present")
        return value
