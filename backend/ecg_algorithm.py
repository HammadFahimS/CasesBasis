"""Doctor-facing trace of the synthetic ECG generation algorithm.

Given an `ECGParameters` object (the LLM's output) plus the original prompts,
produces a step-by-step markdown explanation of how the SVG was built. Used by
the `/ecg-algorithm` review page so clinicians can audit the pipeline.
"""

from __future__ import annotations

import json
from typing import Any

from schemas.ecg import ECGParameters

_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


_RHYTHM_EXPLANATIONS: dict[str, str] = {
    "sinus": "Standard sinus rhythm — beat templates are tiled at the beat interval with no jitter.",
    "sinus_tachycardia": "Sinus rhythm with a shortened beat interval driven by the chosen `heart_rate`.",
    "sinus_bradycardia": "Sinus rhythm with a lengthened beat interval driven by the chosen `heart_rate`.",
    "af": "Beat interval is jittered by ±15 % per beat (irregularly irregular). P waves are suppressed across all leads.",
    "vt": "Wide-QRS beats tiled at a rapid HR; P waves suppressed; QRS morphology forced to wide.",
    "vf": "No coherent beats — chaotic Gaussian noise (σ ≈ 0.18 mV) plus 5.5 Hz and 11 Hz sinusoidal carriers.",
    "svt": "Narrow QRS at very fast HR (typically 180 bpm) with P waves suppressed.",
    "flutter": "Sawtooth atrial activity at 300 bpm (interval 0.20 s) overlaid; QRS templates added on top at the chosen HR.",
    "complete_heart_block": "Independent atrial P waves at ~75 bpm AND escape QRS at `heart_rate` — P–R dissociation is visible.",
}


def _build_user_prompt(diagnosis: str, heart_rate: int | None) -> str:
    user_prompt = f"Generate 12-lead ECG parameters for: {diagnosis}"
    if heart_rate is not None:
        user_prompt += (
            f"\n\nIMPORTANT: The patient's actual heart rate is {heart_rate} bpm. "
            f"Set heart_rate to exactly {heart_rate} in your output."
        )
    return user_prompt


def _findings_table(params: ECGParameters) -> list[str]:
    lines = [
        "| Lead | ST elev (mV) | T inv | Path Q | P absent | QRS morph | Scale |",
        "|------|--------------|:-----:|:------:|:--------:|-----------|:-----:|",
    ]
    for lead in _LEAD_ORDER:
        f = params.leads.get(lead)
        if f is None:
            lines.append(f"| {lead} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {lead} | {f.st_elevation_mv:+.2f} | "
            f"{'✓' if f.t_inverted else ''} | "
            f"{'✓' if f.pathological_q else ''} | "
            f"{'✓' if f.p_wave_absent else ''} | "
            f"{f.qrs_morphology} | "
            f"{f.amplitude_scale:.2f} |"
        )
    return lines


def _morphology_modifiers(params: ECGParameters) -> list[str]:
    morphs = {f.qrs_morphology for f in params.leads.values()}
    out: list[str] = []
    if "lbbb" in morphs:
        out.append(
            "- **LBBB (notched R)** — a second Gaussian R-shoulder is added at "
            "`R + QRS × 0.30`, σ smaller, amplitude `0.40 × scale`."
        )
    if "rbbb" in morphs:
        out.append(
            "- **RBBB (RSR′)** — an extra positive deflection at "
            "`S + QRS × 0.35`, amplitude `0.45 × scale`."
        )
    if "delta" in morphs:
        out.append(
            "- **Delta wave (WPW)** — a slurred upstroke pre-Q at "
            "`Q − QRS × 0.15`, σ larger, amplitude `0.30 × scale`."
        )
    if "wide" in morphs:
        out.append(
            "- **Wide QRS** — σ on R and S waves increased ×1.67, broadening the complex visually."
        )
    if not out:
        out.append("- All leads use normal QRS morphology — no shape modifiers applied.")
    return out


def build_algorithm_markdown(
    *,
    diagnosis: str,
    heart_rate: int | None,
    params: ECGParameters,
    system_prompt: str,
    provider: str,
    model: str,
    case_summary: dict[str, Any] | None = None,
) -> str:
    """Return a markdown trace of how the synthetic ECG was constructed."""
    pr_s = params.pr_interval_ms / 1000.0
    qrs_s = params.qrs_duration_ms / 1000.0
    qt_s = params.qtc_ms / 1000.0
    beat_interval_s = 60.0 / params.heart_rate

    user_prompt = _build_user_prompt(diagnosis, heart_rate)
    findings_rows = _findings_table(params)
    morph_lines = _morphology_modifiers(params)

    lines: list[str] = []
    lines.append("# ECG Synthesis — Algorithm Trace")
    lines.append("")
    lines.append(
        "_This document traces every step the synthetic ECG generator took. "
        "Each section corresponds to a real stage in the pipeline._"
    )
    lines.append("")

    # ── 0. Input ──────────────────────────────────────────────────────────────
    lines.append("## 0. Input")
    lines.append(f"- **Diagnosis:** `{diagnosis}`")
    if heart_rate is not None:
        lines.append(f"- **Heart rate (from case vitals):** {heart_rate} bpm")
    if case_summary:
        if case_summary.get("age") is not None:
            lines.append(f"- **Patient:** {case_summary['age']}yo {case_summary.get('gender', '')}".rstrip())
        bp = case_summary.get("blood_pressure")
        if bp:
            lines.append(f"- **BP:** {bp}")
    lines.append("")

    # ── 1. LLM call ───────────────────────────────────────────────────────────
    lines.append("## 1. LLM parameter generation")
    lines.append(
        f"The diagnosis is sent to **{provider}** (`{model}`) with structured-JSON output "
        f"constrained to the `ECGParameters` Pydantic schema."
    )
    lines.append("")
    lines.append("### System prompt")
    lines.append("```")
    lines.append(system_prompt.strip())
    lines.append("```")
    lines.append("")
    lines.append("### User prompt")
    lines.append("```")
    lines.append(user_prompt)
    lines.append("```")
    lines.append("")

    # ── 2. LLM response ───────────────────────────────────────────────────────
    lines.append("## 2. LLM response (parsed `ECGParameters`)")
    lines.append("```json")
    lines.append(json.dumps(params.model_dump(), indent=2))
    lines.append("```")
    lines.append("")

    # ── 3. Global timing ──────────────────────────────────────────────────────
    lines.append("## 3. Global rhythm and timing")
    lines.append(f"- **Rhythm:** `{params.rhythm}`")
    lines.append(f"- **PR interval:** {params.pr_interval_ms} ms ({pr_s:.3f} s)")
    lines.append(f"- **QRS duration:** {params.qrs_duration_ms} ms ({qrs_s:.3f} s)")
    lines.append(f"- **QTc:** {params.qtc_ms} ms ({qt_s:.3f} s)")
    lines.append(
        f"- **Heart rate:** {params.heart_rate} bpm → beat interval = "
        f"60 ÷ {params.heart_rate} = **{beat_interval_s*1000:.0f} ms**"
    )
    lines.append(f"- **Cardiac axis:** {params.axis_degrees}°")
    lines.append("")

    # ── 4. Per-lead findings ──────────────────────────────────────────────────
    lines.append("## 4. Per-lead findings (as returned by the LLM)")
    lines.extend(findings_rows)
    lines.append("")

    # ── 5. Beat template ──────────────────────────────────────────────────────
    lines.append("## 5. Beat template (per lead)")
    lines.append(
        "Each lead is built as the sum of five Gaussian waves (P, Q, R, S, T) "
        "tiled at the beat interval. Centres and widths derive from the LLM's "
        "PR / QRS / QT values:"
    )
    lines.append("")
    lines.append("| Wave | Centre (s) | σ (s) | Amplitude (mV) |")
    lines.append("|------|-----------:|------:|----------------|")
    lines.append(
        f"| P | {pr_s*0.50:.3f} | {pr_s*0.12:.3f} | "
        "`0.15 × scale` (zero if `p_wave_absent`) |"
    )
    lines.append(
        f"| Q | {pr_s:.3f} | {qrs_s*0.06:.3f} | "
        "`−0.25 × scale` if pathological Q else `−0.08 × scale` |"
    )
    lines.append(
        f"| R | {pr_s + qrs_s*0.35:.3f} | {qrs_s*0.06:.3f} | `0.90 × scale` |"
    )
    lines.append(
        f"| S | {pr_s + qrs_s*0.72:.3f} | {qrs_s*0.055:.3f} | `−0.15 × scale` |"
    )
    lines.append(
        f"| T | {pr_s + qt_s*0.65:.3f} | {(qt_s - qrs_s)*0.18:.3f} | "
        "`0.35 × scale + st × 0.06` (or `−0.20 × scale` if T-inverted) |"
    )
    lines.append("")

    # ── 6. ST segment ─────────────────────────────────────────────────────────
    lines.append("## 6. ST segment")
    lines.append(
        "Between QRS-end (`S + QRS × 0.08`) and T-onset (`T_peak − (QT − QRS) × 0.25`), "
        "a flat offset of `st_elevation_mv × 0.30` is added. "
        "Positive values raise the ST; negative values depress it."
    )
    lines.append("")

    # ── 7. Morphology modifiers ───────────────────────────────────────────────
    lines.append("## 7. QRS morphology modifiers")
    lines.extend(morph_lines)
    lines.append("")

    # ── 8. Rhythm-specific logic ──────────────────────────────────────────────
    lines.append("## 8. Rhythm-specific logic")
    explanation = _RHYTHM_EXPLANATIONS.get(
        params.rhythm,
        "Default beat tiling (no special rhythm handler).",
    )
    lines.append(f"**`{params.rhythm}`** — {explanation}")
    lines.append("")

    # ── 9. Rendering ──────────────────────────────────────────────────────────
    lines.append("## 9. Rendering")
    lines.append("The 12 leads are laid out in a 3 × 4 grid on red ECG-paper:")
    lines.append("")
    lines.append("- Each panel: **218 × 130 px**")
    lines.append("- Strip duration per panel: **2.5 s**")
    lines.append("- Sampling rate: 500 Hz internally, sub-sampled to ~100 Hz for compact SVG paths")
    lines.append("- Minor grid: 0.04 s × 0.1 mV (`#ffdddd`)")
    lines.append("- Major grid: 0.20 s × 0.5 mV (`#ffaaaa`)")
    lines.append("- Iso-electric baseline drawn at 0 mV")
    lines.append("- Y-range: −0.8 to +1.6 mV (0 mV sits slightly above midline)")
    lines.append("")
    lines.append(
        "_The waveform polyline is clipped to the panel and rendered with a "
        "1.2 px stroke. The diagnosis itself is **not** drawn on the image — "
        "the doctor is meant to read it from the morphology alone._"
    )
    lines.append("")

    return "\n".join(lines)
