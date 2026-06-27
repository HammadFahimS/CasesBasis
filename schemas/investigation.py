from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


class LeadIIMorphology(StrictModel):
    """Compact Lead II morphology snapshot for the live patient-panel monitor.

    Carries just enough state for the JS Gaussian-sum renderer to mirror the
    case's actual Lead II waveform — rate, intervals, P/Q/T flags, QRS shape,
    ST offset, amplitude scale, and the rhythm label.
    """

    hr: int
    pr_ms: int
    qrs_ms: int
    qt_ms: int
    p_absent: bool = False
    q_path: bool = False
    t_inv: bool = False
    qrs: Literal["normal", "lbbb", "rbbb", "wide", "delta"] = "normal"
    st_mv: float = 0.0
    amp_scale: float = 1.0
    rhythm_label: str = ""


class ECGVisualPayload(StrictModel):
    """ECG visualization payload.

    Two render modes:
    - type="svg": inline SVG drawn by the legacy numpy synthesizer.
    - type="image": base64-encoded PNG of a paper-style ECG generated from a
      real PTB-XL recording via ECG-Image-Kit.
    """

    type: Literal["svg", "image"]
    leads: list[str]
    svg: str = ""
    data: str = ""
    format: Literal["png", ""] = ""
    source_record: str = ""
    lead_ii_morphology: LeadIIMorphology | None = None


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
