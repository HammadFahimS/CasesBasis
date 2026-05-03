from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult

GENERIC_INVESTIGATION_SYSTEM_PROMPT = """
You are generating clinical investigation results for a medical simulation.

Rules:
- Return valid JSON only
- Return the InvestigationResult structure exactly
- Do not directly reveal the hidden diagnosis
- Keep results realistic and student-interpretable
- If the test is clinically relevant, abnormal findings should support the case without naming the diagnosis
- If the test is low-yield or unrelated, return realistic normal or non-contributory findings
- Use visual_payload: null for generic non-ECG tests
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
