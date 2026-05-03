from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class ECGVisualPayload(StrictModel):
    """Schematic SVG payload for ECG visualization."""

    type: Literal["svg"]
    leads: list[str]
    svg: str


class InvestigationResult(StrictModel):
    """Structured result returned for an ordered investigation."""

    investigation: str
    status: Literal["available", "unavailable"]
    structured_result: dict[str, object]
    summary: str
    visual_payload: ECGVisualPayload | None
