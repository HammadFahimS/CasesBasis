from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult

GENERIC_INVESTIGATION_SYSTEM_PROMPT = """
You are generating clinical investigation results for a medical simulation.

Core rules:
- Return valid JSON only, matching the InvestigationResult structure exactly
- Never name or directly imply the hidden diagnosis in any output field
- Results must be internally consistent with the case's vitals, symptoms, and progression
- Keep findings realistic and student-interpretable — a skilled student should be able to infer the clinical picture
- Use visual_payload: null for all tests

Blood tests (FBC, U&E, LFTs, troponin, BNP, CRP, lactate, coagulation, etc.):
- Provide specific numeric values — avoid suspiciously round numbers
- Include unit and reference_range for every value
- Flag each value: normal, high, low, mildly high, mildly low, or non-contributory
- For cardiac biomarkers (troponin, CK-MB, BNP/NT-proBNP): reflect the timing and severity
  implied by the case; always add a serial_testing_note in interpretation_hint when elevated
- For FBC in infection/sepsis: raise WBC and neutrophils; in anaemia cases adjust Hb/MCV/MCH

Arterial blood gas (ABG):
- Always return pH, PaO2, PaCO2, HCO3, base excess, and SpO2 as individual named values
- Determine the acid-base disorder (respiratory/metabolic acidosis or alkalosis, compensation)
  from the case context; state the interpretation in interpretation_hint

Imaging (CXR, CT, USS, MRI, echo):
- Return findings as a concise list of radiological observations
- Include clinically important negatives (e.g. "no pneumothorax", "no free air under diaphragm")
- Echo: include relevant chamber/valve findings consistent with the pathology

Microbiology (cultures, sensitivities, swabs):
- Return realistic organisms and sensitivities consistent with the likely source and case context
- Mark as pending if clinically the result would not be available immediately

All tests:
- If unrelated to the case, return plausible normal values — never say the test is unavailable
- Do not include treatment recommendations
""".strip()

GENERIC_INVESTIGATION_PROMPT_TEMPLATE = """
Generated case JSON:
{case_json}

Requested investigation:
{requested_investigation}

Return this JSON structure:
{
  "investigation": "string",
  "status": "available",
  "structured_result": {
    "test": "string",
    "findings": ["string"],
    "values": [
      {
        "name": "string",
        "value": "string or number",
        "unit": "string",
        "reference_range": "string",
        "flag": "normal, high, low, mildly high, mildly low, or non-contributory"
      }
    ],
    "interpretation_hint": "string"
  },
  "summary": "string",
  "visual_payload": null
}

Additional constraints:
- Never include the exact diagnosis name in any output field
- Do not include treatment recommendations
- Do not say the test is unavailable; reasonable low-yield tests should be normal or non-contributory
- Use concise realistic units and reference ranges when numeric values are included

Return JSON only.
""".strip()

MAX_INVESTIGATION_CHARS = 120
MAX_INVESTIGATION_ATTEMPTS = 2


class InvestigationGenerationError(RuntimeError):
    """Raised when generic investigation generation cannot produce valid JSON."""

    pass


def sanitize_requested_investigation(requested_investigation: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", requested_investigation)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_INVESTIGATION_CHARS].strip()
    if not sanitized:
        raise ValueError("requested investigation is required")
    return sanitized


def build_generic_investigation_prompt(
    case: ClinicalCase,
    requested_investigation: str,
) -> str:
    sanitized_request = sanitize_requested_investigation(requested_investigation)
    return GENERIC_INVESTIGATION_PROMPT_TEMPLATE.replace(
        "{case_json}",
        case.model_dump_json(indent=2),
    ).replace(
        "{requested_investigation}",
        sanitized_request,
    )


def result_reveals_diagnosis(result: InvestigationResult, case: ClinicalCase) -> bool:
    diagnosis = case.diagnosis.strip().lower()
    if len(diagnosis) < 4:
        return False
    return diagnosis in result.model_dump_json().lower()


def validate_generic_investigation_result(
    result: InvestigationResult,
    case: ClinicalCase,
) -> InvestigationResult:
    if result.status != "available":
        raise ValueError("generic investigation results must be available")
    if result.visual_payload is not None:
        raise ValueError("generic investigation results must not include a visual payload")
    if result_reveals_diagnosis(result, case):
        raise ValueError("investigation result directly reveals the hidden diagnosis")
    if not result.investigation.strip():
        raise ValueError("investigation name is required")
    if not result.summary.strip():
        raise ValueError("investigation summary is required")
    if not result.structured_result:
        raise ValueError("structured_result is required")
    return result


class GenericInvestigationAgent(OpenAIJsonAgent):
    """Generates structured results for non-deterministic investigation requests."""

    def generate(
        self,
        case: ClinicalCase,
        requested_investigation: str,
    ) -> InvestigationResult:
        prompt = build_generic_investigation_prompt(case, requested_investigation)
        errors: list[Exception] = []

        for _ in range(MAX_INVESTIGATION_ATTEMPTS):
            try:
                result = self.parse_response(
                    system_prompt=GENERIC_INVESTIGATION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    response_model=InvestigationResult,
                )
                return validate_generic_investigation_result(result, case)
            except Exception as exc:
                errors.append(exc)

        raise InvestigationGenerationError(
            f"investigation generation failed after {MAX_INVESTIGATION_ATTEMPTS} attempts"
        ) from errors[-1]
