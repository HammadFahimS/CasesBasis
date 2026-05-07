from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult

# ── Call 1: Clinical Reasoning ─────────────────────────────────────────────────

REASONING_SYSTEM_PROMPT = """
You are a senior clinician advising on what investigation results would look like for a specific patient case.

Your job is to reason through the pathophysiology — not to generate numbers yet.

Rules:
- Think step by step about what the requested test would show for this specific diagnosis
- State which values/findings would be abnormal and in which direction (high/low/elevated/reduced)
- State which values would be normal or non-contributory for this condition
- Consider the timing and severity implied by the case (early vs late presentation, mild vs severe)
- Be specific — "troponin would be significantly elevated" is better than "troponin may be abnormal"
- If the test is unrelated to the condition, say so clearly and note that results would be largely normal
- Keep your reasoning concise: 4-8 sentences maximum
""".strip()

REASONING_PROMPT_TEMPLATE = """
Diagnosis: {diagnosis}

Patient summary:
- Age: {age}, Gender: {gender}
- Symptoms: {symptoms}
- Vitals: HR {heart_rate}, BP {blood_pressure}, RR {respiratory_rate}, SpO2 {oxygen_saturation}%
- Risk factors: {risk_factors}
- Progression: {progression}

Requested investigation: {requested_investigation}

Reason through what this investigation would show for this specific diagnosis and case. Be specific about which values are abnormal and why.
""".strip()


# ── Call 2: Result Formatter ───────────────────────────────────────────────────

FORMATTER_SYSTEM_PROMPT = """
You are formatting a structured clinical investigation result for a medical simulation, based on clinical reasoning already provided.

Core rules:
- Return valid JSON only, matching the InvestigationResult structure exactly
- The clinical reasoning tells you what is abnormal — your job is to translate that into realistic specific numbers and findings
- Never name or directly imply the diagnosis in any output field
- Use visual_payload: null for all tests

Blood tests (CBC, U&E, LFTs, troponin, BNP, CRP, lactate, coagulation, etc.):
- Provide specific numeric values — avoid suspiciously round numbers
- Include unit and reference_range for every value
- Flag each value: normal, high, low, mildly high, mildly low, or non-contributory
- For cardiac biomarkers (troponin, CK-MB, BNP/NT-proBNP): reflect timing and severity from the reasoning; always add a serial_testing_note in interpretation_hint when elevated

Arterial blood gas (ABG):
- Always return pH, PaO2, PaCO2, HCO3, base excess, and SpO2 as individual named values
- State the acid-base interpretation in interpretation_hint

Imaging (CXR, CT, USS, MRI, echo):
- Return findings as a concise list of radiological observations
- Include clinically important negatives (e.g. "no pneumothorax", "no free air under diaphragm")

Microbiology (cultures, sensitivities, swabs):
- Return realistic organisms and sensitivities consistent with the case
- Mark as pending if the result would not be available immediately

All tests:
- Do not include treatment recommendations
- summary must be one clinical sentence suitable for a trainee to read
""".strip()

FORMATTER_PROMPT_TEMPLATE = """
Clinical reasoning for this investigation:
{reasoning}

Requested investigation: {requested_investigation}

Case context (for calibrating values):
{case_json}

Return this JSON structure:
{{
  "investigation": "string",
  "status": "available",
  "structured_result": {{
    "test": "string",
    "findings": ["string"],
    "values": [
      {{
        "name": "string",
        "value": "string or number",
        "unit": "string",
        "reference_range": "string",
        "flag": "normal, high, low, mildly high, mildly low, or non-contributory"
      }}
    ],
    "interpretation_hint": "string"
  }},
  "summary": "string",
  "visual_payload": null
}}

Return JSON only. Do not include the diagnosis name in any output field.
""".strip()

MAX_INVESTIGATION_CHARS = 120
MAX_INVESTIGATION_ATTEMPTS = 2


class InvestigationGenerationError(RuntimeError):
    """Raised when investigation generation cannot produce valid JSON after retries."""
    pass


def sanitize_requested_investigation(requested_investigation: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", requested_investigation)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_INVESTIGATION_CHARS].strip()
    if not sanitized:
        raise ValueError("requested investigation is required")
    return sanitized


def build_reasoning_prompt(case: ClinicalCase, requested_investigation: str) -> str:
    return REASONING_PROMPT_TEMPLATE.format(
        diagnosis=case.diagnosis,
        age=case.patient.age,
        gender=case.patient.gender,
        symptoms=", ".join(case.symptoms),
        heart_rate=case.vitals.heart_rate,
        blood_pressure=case.vitals.blood_pressure,
        respiratory_rate=case.vitals.respiratory_rate,
        oxygen_saturation=case.vitals.oxygen_saturation,
        risk_factors=", ".join(case.risk_factors) if case.risk_factors else "none",
        progression="; ".join(e.description for e in case.progression),
        requested_investigation=requested_investigation,
    )


def build_formatter_prompt(
    case: ClinicalCase,
    requested_investigation: str,
    reasoning: str,
) -> str:
    return FORMATTER_PROMPT_TEMPLATE.format(
        reasoning=reasoning,
        requested_investigation=requested_investigation,
        case_json=case.model_dump_json(indent=2),
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
    """
    Two-call pipeline for investigation result generation.

    Call 1 (reasoning): Senior clinician reasons through what the test would
    show for this specific diagnosis — plain text, no JSON pressure.

    Call 2 (formatter): Translates the reasoning into a structured JSON result
    with specific numeric values, flags, and clinical hints.
    """

    def generate(
        self,
        case: ClinicalCase,
        requested_investigation: str,
    ) -> InvestigationResult:
        reasoning = self._reason(case, requested_investigation)
        errors: list[Exception] = []

        for _ in range(MAX_INVESTIGATION_ATTEMPTS):
            try:
                formatter_prompt = build_formatter_prompt(case, requested_investigation, reasoning)
                result = self.parse_response(
                    system_prompt=FORMATTER_SYSTEM_PROMPT,
                    user_prompt=formatter_prompt,
                    response_model=InvestigationResult,
                )
                return validate_generic_investigation_result(result, case)
            except Exception as exc:
                errors.append(exc)

        raise InvestigationGenerationError(
            f"investigation generation failed after {MAX_INVESTIGATION_ATTEMPTS} attempts"
        ) from errors[-1]

    def _reason(self, case: ClinicalCase, requested_investigation: str) -> str:
        prompt = build_reasoning_prompt(case, requested_investigation)
        return self.text_response(
            system_prompt=REASONING_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
