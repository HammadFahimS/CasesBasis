from __future__ import annotations

import json
import re
from typing import Any

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from schemas.doctor_consult import DoctorConsultResponse

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
        prompt = render(
            load("consult_doctor", "user"),
            redacted_context_json=json.dumps(redacted_context, indent=2, sort_keys=True),
            consult_question=sanitize_consult_question(consult_question),
        )
        errors: list[Exception] = []

        for _ in range(MAX_CONSULT_ATTEMPTS):
            try:
                return self.parse_response(
                    system_prompt=load("consult_doctor", "system"),
                    user_prompt=prompt,
                    response_model=DoctorConsultResponse,
                )
            except Exception as exc:
                errors.append(exc)

        raise ConsultDoctorError(
            f"doctor consult failed after {MAX_CONSULT_ATTEMPTS} attempts"
        ) from errors[-1]


def sanitize_consult_question(question: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", question)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_CONSULT_QUESTION_CHARS].strip()
    if not sanitized:
        raise ValueError("consult question is required")
    return sanitized
