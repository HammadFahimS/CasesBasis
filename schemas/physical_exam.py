from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EXAM_SYSTEMS = Literal[
    "General",
    "Cardiovascular",
    "Respiratory",
    "Abdominal",
    "Neurological",
    "Musculoskeletal",
    "Dermatological",
    "ENT/Head & Neck",
    "Peripheral Vascular",
]

# Maps each system to its pin coordinates on the body SVG (viewBox 0 0 120 300)
SYSTEM_REGIONS: dict[str, dict[str, object]] = {
    "General":             {"x": 60,  "y": 28,  "label": "General"},
    "ENT/Head & Neck":     {"x": 60,  "y": 52,  "label": "ENT / Head & Neck"},
    "Cardiovascular":      {"x": 44,  "y": 110, "label": "Cardiovascular"},
    "Respiratory":         {"x": 76,  "y": 110, "label": "Respiratory"},
    "Abdominal":           {"x": 60,  "y": 155, "label": "Abdominal"},
    "Musculoskeletal":     {"x": 22,  "y": 140, "label": "Musculoskeletal"},
    "Dermatological":      {"x": 100, "y": 155, "label": "Dermatological"},
    "Neurological":        {"x": 60,  "y": 18,  "label": "Neurological"},
    "Peripheral Vascular": {"x": 60,  "y": 255, "label": "Peripheral Vascular"},
}


class ExamFinding(StrictModel):
    system: str
    findings: list[str]
    summary: str
    abnormal: bool


class PhysicalExamResponse(StrictModel):
    results: list[ExamFinding]


class PhysicalExamRequest(BaseModel):
    session_id: str
    systems: list[str]
