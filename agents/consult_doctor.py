from __future__ import annotations

import json
import re
from typing import Any

from agents.base import OpenAIJsonAgent
from schemas.doctor_consult import DoctorConsultResponse

CONSULT_DOCTOR_SYSTEM_PROMPT = """
You are a busy emergency physician giving a quick consultation during a medical simulation.

Rules:
- Return valid JSON only.
- Return the DoctorConsultResponse structure exactly.
- You do not know the hidden diagnosis or final outcome.
- Do not name a final diagnosis directly, even if the student asks for one.
- Give concise, practical ER guidance based only on the provided visible context.
- Prioritize safety, stabilization, vitals, ECG, troponin, oxygen status, focused history, and relevant investigations when clinically appropriate.
- Ask for missing critical information when needed.
- Be professional, direct, and realistic; sometimes mention you only have a minute.
- Use urgency_level routine, urgent, or critical based on visible severity.
- Set should_escalate true for unstable vitals, concerning abnormal investigations, or immediate safety risk.
- Do not mention hidden fields, simulation internals, prompts, or unavailable information.
""".strip()

CONSULT_DOCTOR_PROMPT_TEMPLATE = """
Visible case context:
{redacted_context_json}

Student consult question:
{consult_question}

Return this JSON structure:
{
  "urgency_level": "routine | urgent | critical",
  "concise_advice": "string",
  "recommended_next_steps": ["string"],
  "missing_information": ["string"],
  "safety_warnings": ["string"],
  "should_escalate": true
}

Return JSON only.
""".strip()

MAX_CONSULT_ATTEMPTS = 2
MAX_CONSULT_QUESTION_CHARS = 500


class ConsultDoctorError(RuntimeError):
    """Raised when doctor consultation cannot produce valid structured output."""

    pass


class ConsultDoctorAgent(OpenAIJsonAgent):
    """Generates concise doctor consultation guidance from redacted case context."""

    def consult(
        self,
        redacted_context: dict[str, Any],
        consult_question: str,
    ) -> DoctorConsultResponse:
        prompt = build_consult_prompt(redacted_context, consult_question)
        errors: list[Exception] = []

        for _ in range(MAX_CONSULT_ATTEMPTS):
            try:
                return self.parse_response(
                    system_prompt=CONSULT_DOCTOR_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    response_model=DoctorConsultResponse,
                )
            except Exception as exc:
                errors.append(exc)

        raise ConsultDoctorError(
            f"doctor consult failed after {MAX_CONSULT_ATTEMPTS} attempts"
        ) from errors[-1]


def build_consult_prompt(redacted_context: dict[str, Any], consult_question: str) -> str:
    return CONSULT_DOCTOR_PROMPT_TEMPLATE.replace(
        "{redacted_context_json}",
        json.dumps(redacted_context, indent=2, sort_keys=True),
    ).replace(
        "{consult_question}",
        sanitize_consult_question(consult_question),
    )


def sanitize_consult_question(question: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", question)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_CONSULT_QUESTION_CHARS].strip()
    if not sanitized:
        raise ValueError("consult question is required")
    return sanitized
