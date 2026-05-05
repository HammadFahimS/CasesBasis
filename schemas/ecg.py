from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

Rhythm = Literal[
    "sinus",
    "sinus_tachycardia",
    "sinus_bradycardia",
    "af",
    "vt",
    "vf",
    "svt",
    "flutter",
    "complete_heart_block",
]

QRSMorphology = Literal["normal", "lbbb", "rbbb", "wide", "delta"]


class LeadFindings(BaseModel):
    """ECG findings for one lead — fed directly into the waveform synthesizer."""

    st_elevation_mv: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="ST elevation (+) or depression (−) in millivolts. 0 = isoelectric.",
    )
    t_inverted: bool = Field(default=False)
    pathological_q: bool = Field(
        default=False,
        description="Q wave > 25 % of R amplitude or > 40 ms wide.",
    )
    p_wave_absent: bool = Field(default=False)
    qrs_morphology: QRSMorphology = Field(default="normal")
    amplitude_scale: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description=(
            "Relative voltage scale for this lead. "
            "1.0 = normal. 0.3–0.4 = low voltage. 1.4–1.6 = LVH / hyperkalemia."
        ),
    )


class ECGParameters(BaseModel):
    """
    Complete 12-lead ECG parameters.

    Every field is consumed directly by the numpy synthesizer, so the types and
    ranges are enforced by Pydantic before synthesis begins.
    """

    heart_rate: int = Field(ge=15, le=300, description="Ventricular rate in bpm.")
    rhythm: Rhythm
    pr_interval_ms: int = Field(
        default=160,
        ge=0,
        le=400,
        description="PR interval in ms. Set 0 when P waves are absent.",
    )
    qrs_duration_ms: int = Field(
        default=90,
        ge=40,
        le=300,
        description="QRS duration in ms. Normal < 120 ms.",
    )
    qtc_ms: int = Field(
        default=420,
        ge=200,
        le=700,
        description="Rate-corrected QT interval in ms.",
    )
    axis_degrees: int = Field(
        default=60,
        ge=-180,
        le=180,
        description="Cardiac axis in degrees. Normal −30 to +90.",
    )
    leads: dict[str, LeadFindings] = Field(
        description="Per-lead findings. Must include all 12 leads.",
    )
