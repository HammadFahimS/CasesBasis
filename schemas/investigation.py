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


class CXRVisualPayload(StrictModel):
    """Base64-encoded JPEG chest X-ray image payload."""

    type: Literal["image"]
    format: Literal["jpeg"]
    data: str  # base64-encoded JPEG
    finding: str  # matched finding category (e.g. "Pneumonia")


class InvestigationResult(StrictModel):
    """Structured result returned for an ordered investigation."""

    investigation: str
    status: Literal["available", "unavailable"]
    structured_result: dict[str, object]
    summary: str
    visual_payload: ECGVisualPayload | CXRVisualPayload | None
