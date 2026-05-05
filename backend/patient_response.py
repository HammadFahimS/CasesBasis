from __future__ import annotations

from typing import Any

from agents.base import build_llm_client, get_llm_provider
from agents.patient import PatientResponseAgent
from schemas.clinical_case import ClinicalCase


class PatientResponseService:
    """Service wrapper for patient role-play responses."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        agent: PatientResponseAgent | None = None,
    ) -> None:
        if agent:
            self.agent = agent
            return
        provider = get_llm_provider()
        shared_client = client or build_llm_client(provider)
        self.agent = PatientResponseAgent(
            client=shared_client,
            model=model,
            provider=provider,
        )

    def answer_question(
        self,
        case: ClinicalCase | dict[str, Any] | str,
        question: str,
        *,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> str:
        clinical_case = normalize_case(case)
        return self.agent.answer(
            clinical_case,
            question,
            conversation_history=conversation_history or [],
        )


def normalize_case(case: ClinicalCase | dict[str, Any] | str) -> ClinicalCase:
    if isinstance(case, ClinicalCase):
        return case
    if isinstance(case, str):
        return ClinicalCase.model_validate_json(case)
    return ClinicalCase.model_validate(case)
