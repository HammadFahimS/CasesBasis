from __future__ import annotations

import re
from typing import Any

from agents.investigation_generator import (
    GenericInvestigationAgent,
    sanitize_requested_investigation,
)
from backend.patient_response import normalize_case
from schemas.clinical_case import ClinicalCase
from schemas.investigation import ECGVisualPayload, InvestigationResult

ECG_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_REQUEST_TERMS = ("ecg", "ekg", "electrocardiogram")
TROPONIN_REQUEST_TERMS = ("troponin", "cardiac enzymes", "blood test for heart attack")
IMPOSSIBLE_INVESTIGATION_TERMS = (
    "crystal ball",
    "magic",
    "telepathy",
    "mind reading",
    "time travel",
    "banana test",
    "moonlight",
    "spell",
)


class InvestigationService:
    """Routes investigation requests to deterministic or generic result generation."""

    def __init__(
        self,
        *,
        generic_generator: GenericInvestigationAgent | None = None,
    ) -> None:
        self.generic_generator = generic_generator

    def get_result(
        self,
        case: ClinicalCase | dict[str, Any] | str,
        requested_investigation: str,
    ) -> InvestigationResult:
        clinical_case = normalize_case(case)
        if is_nonsensical_investigation_request(requested_investigation):
            return unavailable_result(requested_investigation)

        sanitized_request = sanitize_requested_investigation(requested_investigation)
        if is_ecg_request(sanitized_request):
            return build_ecg_result(clinical_case)
        if is_troponin_request(sanitized_request):
            return build_troponin_result(clinical_case)
        return self._generic_generator().generate(clinical_case, sanitized_request)

    def get_results(
        self,
        case: ClinicalCase | dict[str, Any] | str,
        requested_investigations: str,
    ) -> list[InvestigationResult]:
        return [
            self.get_result(case, requested_investigation)
            for requested_investigation in split_investigation_requests(requested_investigations)
        ]

    def _generic_generator(self) -> GenericInvestigationAgent:
        if self.generic_generator is None:
            self.generic_generator = GenericInvestigationAgent()
        return self.generic_generator


def split_investigation_requests(requested_investigations: str) -> list[str]:
    try:
        sanitized = sanitize_requested_investigation(requested_investigations)
    except ValueError:
        return [requested_investigations]
    parts = [
        clean_investigation_request_part(part)
        for part in re.split(r"\s*(?:,|;|\+|&|\band\b|\bplus\b)\s*", sanitized)
    ]
    cleaned = [part for part in parts if part]
    return cleaned or [sanitized]


def clean_investigation_request_part(requested_investigation: str) -> str:
    cleaned = requested_investigation.strip(" .")
    cleaned = re.sub(r"^(?:order|do|get|check|send|run)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .")


def is_ecg_request(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower()
    return any(term in normalized for term in ECG_REQUEST_TERMS)


def is_troponin_request(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower()
    return any(term in normalized for term in TROPONIN_REQUEST_TERMS)


def is_nonsensical_investigation_request(requested_investigation: str) -> bool:
    try:
        normalized = sanitize_requested_investigation(requested_investigation).lower()
    except ValueError:
        return True
    if not re.search(r"[a-z]", normalized):
        return True
    if len(normalized) < 2:
        return True
    if any(term in normalized for term in IMPOSSIBLE_INVESTIGATION_TERMS):
        return True
    words = re.findall(r"[a-z]+", normalized)
    if len(words) == 1 and len(words[0]) > 14 and not re.search(r"[aeiou]", words[0]):
        return True
    return False


def contains_medical_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def build_ecg_result(case: ClinicalCase) -> InvestigationResult:
    ischemic_case = has_ischemic_features(case)
    if ischemic_case:
        findings = [
            "ST elevation in contiguous inferior leads II, III, and aVF",
            "reciprocal ST depression in leads I and aVL",
            "hyperacute T wave changes in the affected territory",
        ]
        interpretation = "Acute ischemic pattern; urgent clinician review is indicated."
        rhythm = "sinus tachycardia"
        rate = max(90, min(case.vitals.heart_rate, 140))
        territory = "inferior"
        st_segments = {
            "II": "elevated",
            "III": "elevated",
            "aVF": "elevated",
            "I": "depressed",
            "aVL": "depressed",
        }
    else:
        findings = [
            "sinus rhythm",
            "no clear ST elevation",
            "no obvious reciprocal ST depression",
        ]
        interpretation = "No acute ischemic pattern is evident on this schematic tracing."
        rhythm = "sinus rhythm"
        rate = max(50, min(case.vitals.heart_rate, 120))
        territory = "none"
        st_segments = {}

    structured_result: dict[str, object] = {
        "test": "ECG",
        "rhythm": rhythm,
        "rate_bpm": rate,
        "findings": findings,
        "st_segments": st_segments,
        "territory_pattern": territory,
        "interpretation_hint": interpretation,
    }
    return InvestigationResult(
        investigation="ECG",
        status="available",
        structured_result=structured_result,
        summary="ECG completed. The tracing shows ST-T wave abnormalities that require interpretation in clinical context."
        if ischemic_case
        else "ECG completed. No clear acute ST elevation pattern is visible on this schematic tracing.",
        visual_payload=ECGVisualPayload(
            type="svg",
            leads=ECG_LEADS,
            svg=build_ecg_svg(st_segments=st_segments),
        ),
    )


def has_ischemic_features(case: ClinicalCase) -> bool:
    case_text = " ".join(
        [
            case.diagnosis,
            *case.symptoms,
            *case.risk_factors,
            *case.exam_findings,
            *(event.description for event in case.progression),
        ]
    ).lower()
    diagnosis_terms = (
        "myocardial infarction",
        "stemi",
        "nstemi",
        "ischemia",
        "coronary",
    )
    symptom_patterns = (
        "chest pressure",
        "chest pain",
        "left arm",
        "st elevation",
        "reciprocal",
    )
    diagnosis = case.diagnosis.lower()
    return any(contains_medical_term(diagnosis, term) for term in diagnosis_terms) or any(
        contains_medical_term(case_text, term) for term in symptom_patterns
    )


def build_troponin_result(case: ClinicalCase) -> InvestigationResult:
    ischemic_case = has_ischemic_features(case)
    mildly_abnormal = has_non_ischemic_troponin_stressors(case)
    if ischemic_case:
        value_ng_ml = 4.8
        flag = "high"
        interpretation = "Markedly elevated cardiac troponin; correlate with symptoms, ECG, and serial change."
    elif mildly_abnormal:
        value_ng_ml = 0.08
        flag = "mildly high"
        interpretation = "Mild troponin elevation; can occur with physiologic stress and needs clinical correlation."
    else:
        value_ng_ml = 0.01
        flag = "normal"
        interpretation = "Troponin is within the reference range on this sample."

    structured_result: dict[str, object] = {
        "test": "high-sensitivity troponin I",
        "value": value_ng_ml,
        "unit": "ng/mL",
        "reference_range": "<0.04 ng/mL",
        "flag": flag,
        "interpretation_hint": interpretation,
        "serial_testing_note": "Repeat testing may be needed if symptoms are early or ongoing.",
    }
    summary = (
        f"Troponin I is {value_ng_ml:.2f} ng/mL "
        f"(reference range <0.04 ng/mL), flagged as {flag}."
    )
    return InvestigationResult(
        investigation="Troponin",
        status="available",
        structured_result=structured_result,
        summary=summary,
        visual_payload=None,
    )


def has_non_ischemic_troponin_stressors(case: ClinicalCase) -> bool:
    case_text = " ".join(
        [
            case.diagnosis,
            *case.symptoms,
            *case.risk_factors,
            *case.exam_findings,
            *(event.description for event in case.progression),
        ]
    ).lower()
    stress_terms = (
        "sepsis",
        "septic",
        "shock",
        "renal",
        "kidney",
        "tachycardia",
        "hypoxia",
        "pulmonary embolism",
        "heart failure",
    )
    return any(term in case_text for term in stress_terms)


def build_ecg_svg(*, st_segments: dict[str, str]) -> str:
    width = 720
    lead_height = 36
    height = 24 + (len(ECG_LEADS) * lead_height)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g stroke="#f3caca" stroke-width="1">',
    ]
    for x in range(0, width, 20):
        rows.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}"/>')
    for y in range(0, height, 20):
        rows.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}"/>')
    rows.append("</g>")
    rows.append('<g fill="#222" font-family="Arial, sans-serif" font-size="12">')
    for index, lead in enumerate(ECG_LEADS):
        y = 24 + index * lead_height
        rows.append(f'<text x="10" y="{y}">{lead}</text>')
        rows.append(
            f'<path d="{lead_wave_path(lead, x_start=60, y_base=y - 4, st_status=st_segments.get(lead))}" '
            'fill="none" stroke="#111" stroke-width="2"/>'
        )
    rows.append("</g></svg>")
    return "".join(rows)


def lead_wave_path(lead: str, *, x_start: int, y_base: int, st_status: str | None) -> str:
    st_offset = -8 if st_status == "elevated" else 8 if st_status == "depressed" else 0
    points: list[tuple[int, int]] = []
    x = x_start
    for beat in range(4):
        offset = beat * 145
        points.extend(
            [
                (x + offset, y_base),
                (x + offset + 18, y_base),
                (x + offset + 26, y_base - 5),
                (x + offset + 34, y_base),
                (x + offset + 48, y_base),
                (x + offset + 54, y_base - 16),
                (x + offset + 60, y_base + 12),
                (x + offset + 68, y_base + st_offset),
                (x + offset + 92, y_base + st_offset),
                (x + offset + 112, y_base - 7 + st_offset),
                (x + offset + 132, y_base),
            ]
        )
    return "M " + " L ".join(f"{px} {py}" for px, py in points)


def unavailable_result(requested_investigation: str) -> InvestigationResult:
    requested = requested_investigation.strip() or "requested investigation"
    return InvestigationResult(
        investigation=requested,
        status="unavailable",
        structured_result={
            "requested": requested,
            "message": "That test is not available yet",
        },
        summary="That test is not available yet",
        visual_payload=None,
    )
