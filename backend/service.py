from __future__ import annotations

from typing import Any

from agents.base import build_llm_client, get_llm_provider, manual_sim_model
from agents.generator import GenerationError, GeneratorAgent
from agents.validator import ValidatorAgent
from backend.medical_validation import MedicalValidationError, validate_medical_realism
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation


class ClinicalCaseSimulator:
    """Coordinates case generation, deterministic validation, and fallback correction."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        generator: GeneratorAgent | None = None,
        validator: ValidatorAgent | None = None,
    ) -> None:
        if generator and validator:
            self.generator = generator
            self.validator = validator
            return
        provider = get_llm_provider()
        shared_client = client or build_llm_client(provider)
        # The Cerebras default (zai-glm-4.7) can't reliably satisfy the strict
        # ClinicalCase/MedicalFacts schemas, so the manual simulator pins the
        # stronger gpt-oss-120b. An explicit `model` arg or a non-Cerebras
        # provider is left untouched.
        gen_model = model if model is not None else manual_sim_model(provider)
        self.generator = generator or GeneratorAgent(
            client=shared_client,
            model=gen_model,
            provider=provider,
        )
        self.validator = validator or ValidatorAgent(
            client=shared_client,
            model=gen_model,
            provider=provider,
        )

    def generate_case(
        self,
        diagnosis: str,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        first_case = self._generate_case_once(
            diagnosis,
            difficulty=difficulty,
            session_history=session_history,
        )
        try:
            validate_medical_realism(first_case)
            return first_case
        except MedicalValidationError as exc:
            corrected_case = self.validator.correct(first_case, failure_reason=str(exc))
            try:
                validate_medical_realism(corrected_case)
                return corrected_case
            except MedicalValidationError as correction_exc:
                strict_case = self.validator.regenerate_strict(
                    diagnosis,
                    failure_reason=str(correction_exc),
                    difficulty=difficulty,
                    session_history=session_history,
                )
                validate_medical_realism(strict_case)
                return strict_case

    def _generate_case_once(
        self,
        diagnosis: str,
        difficulty: str,
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None,
    ) -> ClinicalCase:
        try:
            return self.generator.generate(
                diagnosis,
                difficulty=difficulty,
                session_history=session_history,
            )
        except GenerationError as exc:
            return self.validator.recover(
                diagnosis,
                failure_reason=str(exc),
                difficulty=difficulty,
                session_history=session_history,
            )
