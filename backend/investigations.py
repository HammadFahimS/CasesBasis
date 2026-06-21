from __future__ import annotations

import re
from typing import Any

from agents.investigation_generator import (
    GenericInvestigationAgent,
    InvestigationGenerationError,
    sanitize_requested_investigation,
)
from backend.cxr_service import CXRService
from backend.ecg_service import ECGService
from backend.patient_response import normalize_case
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult

ECG_REQUEST_TERMS = ("ecg", "ekg", "electrocardiogram")
CXR_REQUEST_TERMS = ("cxr", "chest x-ray", "chest xray", "chest x ray", "chest radiograph", "plain film")
TROPONIN_REQUEST_TERMS = ("troponin", "cardiac enzyme", "hs-troponin", "cardiac biomarker")
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
        cxr_service: CXRService | None = None,
    ) -> None:
        self.generic_generator = generic_generator
        self.cxr_service = cxr_service

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
        if is_cxr_request(sanitized_request):
            return build_cxr_result(clinical_case, self._cxr_service())
        try:
            return self._generic_generator().generate(clinical_case, sanitized_request)
        except InvestigationGenerationError:
            return unavailable_result(sanitized_request)

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

    def _cxr_service(self) -> CXRService:
        if self.cxr_service is None:
            self.cxr_service = CXRService()
        return self.cxr_service


def split_investigation_requests(requested_investigations: str) -> list[str]:
    try:
        sanitized = sanitize_requested_investigation(requested_investigations)
    except ValueError:
        return [requested_investigations]
    parts = [
        clean_investigation_request_part(part)
        for part in re.split(r"\s*(?:,|;|\+|&|\band\b|\bplus\b)\s*", sanitized)
    ]
    cleaned = [part for part in parts if part and not is_explanatory_investigation_fragment(part)]
    return cleaned or [sanitized]


def clean_investigation_request_part(requested_investigation: str) -> str:
    cleaned = requested_investigation.strip(" .")
    cleaned = re.sub(r"^(?:order|do|get|check|send|run)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .")


def is_explanatory_investigation_fragment(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower().strip()
    return normalized.startswith(("such as ", "including ", "for example "))


def is_ecg_request(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower()
    return any(term in normalized for term in ECG_REQUEST_TERMS)


def is_troponin_request(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower()
    if any(term in normalized for term in TROPONIN_REQUEST_TERMS):
        return True
    words = set(re.findall(r"\b\w+\b", normalized))
    return {"heart", "attack"}.issubset(words) and bool(words & {"blood", "test"})


def is_cxr_request(requested_investigation: str) -> bool:
    normalized = requested_investigation.lower()
    return any(term in normalized for term in CXR_REQUEST_TERMS)


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


_ecg_service: ECGService | None = None


def _get_ecg_service() -> ECGService:
    global _ecg_service
    if _ecg_service is None:
        _ecg_service = ECGService()
    return _ecg_service


def build_ecg_result(case: ClinicalCase) -> InvestigationResult:
    visual_payload = _get_ecg_service().build_visual_payload(
        case.diagnosis,
        heart_rate=case.vitals.heart_rate,
        age=case.patient.age,
        sex=case.patient.gender,
    )
    ischemic = has_ischemic_features(case)
    if ischemic:
        findings = [
            "ST elevation in the ischaemic territory",
            "reciprocal ST depression in remote leads",
            "hyperacute T wave changes",
        ]
        st_segments: dict[str, str] = {
            "II": "elevated",
            "III": "elevated",
            "aVF": "elevated",
            "aVL": "depressed",
            "V1": "elevated",
            "V2": "elevated",
            "V3": "elevated",
            "V4": "elevated",
        }
        summary = "ECG shows acute ischaemic changes. Interpret in full clinical context."
    else:
        findings = ["rhythm and morphology as shown on tracing"]
        st_segments = {}
        summary = "ECG completed. Review the waveform tracing for abnormalities."
    structured_result: dict[str, object] = {
        "test": "ECG",
        "findings": findings,
        "st_segments": st_segments,
        "interpretation_hint": "Correlate waveform with clinical presentation.",
    }
    return InvestigationResult(
        investigation="ECG",
        status="available",
        structured_result=structured_result,
        summary=summary,
        visual_payload=visual_payload,
    )


def has_ischemic_features(case: ClinicalCase) -> bool:
    case_text = " ".join(
        [
            case.diagnosis,
            *case.symptoms,
            *case.risk_factors,
            *case.exam_findings_flat,
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
    ischemic_context_terms = (
        "st elevation",
        "reciprocal",
        "ischemic",
        "ischemia",
    )
    diagnosis = case.diagnosis.lower()
    return any(contains_medical_term(diagnosis, term) for term in diagnosis_terms) or any(
        contains_medical_term(case_text, term) for term in ischemic_context_terms
    )


def build_troponin_result(case: ClinicalCase) -> InvestigationResult:
    ischemic = has_ischemic_features(case)
    row: dict[str, object] = {
        "name": "high-sensitivity troponin I",
        "value": 4.8 if ischemic else 0.01,
        "unit": "ng/mL",
        "reference_range": "<0.04 ng/mL",
    }
    if ischemic:
        row["flag"] = "high"
    structured_result: dict[str, object] = {
        "test": "Troponin",
        "values": [row],
        "interpretation_hint": "Correlate with clinical presentation and serial measurements.",
    }
    summary = (
        "High-sensitivity troponin markedly elevated. Interpret in full clinical context."
        if ischemic
        else "High-sensitivity troponin within normal limits."
    )
    return InvestigationResult(
        investigation="Troponin",
        status="available",
        structured_result=structured_result,
        summary=summary,
        visual_payload=None,
    )


def build_cxr_result(case: ClinicalCase, service: CXRService) -> InvestigationResult:
    visual_payload = service.build_visual_payload(
        case.diagnosis,
        age=case.patient.age,
        gender=case.patient.gender,
        race=case.patient.race,
    )
    finding = visual_payload.finding if visual_payload else "not available"
    structured_result: dict[str, object] = {
        "test": "Chest X-Ray (PA)",
        "findings": [f"Radiograph consistent with {finding} pattern" if visual_payload else "Image unavailable"],
        "interpretation_hint": "Correlate with clinical presentation and other investigations.",
    }
    summary = (
        f"CXR shows findings consistent with {finding}. Interpret in full clinical context."
        if visual_payload
        else "CXR image unavailable — run scripts/download_cxr_samples.py first."
    )
    return InvestigationResult(
        investigation="Chest X-Ray",
        status="available",
        structured_result=structured_result,
        summary=summary,
        visual_payload=visual_payload,
    )


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
