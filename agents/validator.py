from __future__ import annotations

from agents.base import OpenAIJsonAgent
from agents.generator import build_generation_prompt, sanitize_diagnosis
from agents.prompt_loader import load
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation
from schemas.session_context import build_session_context

MAX_VALIDATOR_ATTEMPTS = 2


class ValidatorAgent(OpenAIJsonAgent):
    """Repairs or recovers case JSON after generation or validation failures."""

    def correct(self, case: ClinicalCase, failure_reason: str | None = None) -> ClinicalCase:
        allowed_fields = allowed_correction_fields(failure_reason)
        prompt = f"""
Validation failure: {failure_reason or "medical consistency check failed"}

Only modify these top-level fields if needed: {", ".join(sorted(allowed_fields))}
Leave all other fields exactly as they are.

Generated case JSON:
{case.model_dump_json(indent=2)}

Return the corrected case JSON only.
""".strip()
        proposed_case = self.parse_clinical_case_with_retries(
            system_prompt=load("validator", "correction_system"),
            user_prompt=prompt,
        )
        return merge_allowed_corrections(case, proposed_case, allowed_fields)

    def recover(
        self,
        diagnosis: str,
        failure_reason: str | None = None,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        sanitized_diagnosis = sanitize_diagnosis(diagnosis)
        prompt = build_generation_prompt(
            sanitized_diagnosis,
            difficulty,
            session_context=build_session_context(session_history),
        )
        if failure_reason:
            prompt = f"{prompt}\n\nPrevious generation failure: {failure_reason}"
        generated_case = self.parse_clinical_case_with_retries(
            system_prompt=load("validator", "fallback_system"),
            user_prompt=prompt,
        )
        if generated_case.diagnosis.strip().lower() != sanitized_diagnosis.lower():
            generated_case.diagnosis = sanitized_diagnosis
        return generated_case

    def regenerate_strict(
        self,
        diagnosis: str,
        failure_reason: str,
        difficulty: str = "medium",
        session_history: list[ClinicalEvaluation | dict[str, object] | str] | None = None,
    ) -> ClinicalCase:
        sanitized_diagnosis = sanitize_diagnosis(diagnosis)
        prompt = build_generation_prompt(
            sanitized_diagnosis,
            difficulty,
            session_context=build_session_context(session_history),
        )
        prompt = f"""
{prompt}

STRICT REGENERATION AFTER MEDICAL VALIDATION FAILURE:
- The previous generated or corrected case failed deterministic validation.
- Validation failure reason: {failure_reason}
- Generate a fresh case for the same diagnosis and difficulty.
- Avoid the specific validation failure.
- Do not include unrelated symptom systems unless they are clearly justified by history, risk factors, exam findings, or progression.
- Keep all symptoms, vitals, risk factors, exam findings, and progression medically coherent for the diagnosis.
- Preserve the exact JSON structure and do not add fields.

Return JSON only.
""".strip()
        generated_case = self.parse_response(
            system_prompt=load("validator", "strict_system"),
            user_prompt=prompt,
            response_model=ClinicalCase,
        )
        if generated_case.diagnosis.strip().lower() != sanitized_diagnosis.lower():
            generated_case.diagnosis = sanitized_diagnosis
        return generated_case

    def parse_clinical_case_with_retries(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ClinicalCase:
        errors: list[Exception] = []
        for _ in range(MAX_VALIDATOR_ATTEMPTS):
            try:
                return self.parse_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=ClinicalCase,
                )
            except Exception as exc:
                errors.append(exc)
        raise RuntimeError(
            f"case validation fallback failed after {MAX_VALIDATOR_ATTEMPTS} attempts"
        ) from errors[-1]


def allowed_correction_fields(failure_reason: str | None) -> set[str]:
    reason = (failure_reason or "").lower()
    fields: set[str] = set()
    if "symptom" in reason:
        fields.add("symptoms")
    if "heart_rate" in reason or "heart rate" in reason:
        fields.add("vitals")
    if "blood_pressure" in reason or "blood pressure" in reason or "hypotension" in reason:
        fields.add("vitals")
    if "respiratory_rate" in reason or "respiratory diagnosis" in reason:
        fields.add("vitals")
    if "oxygen" in reason or "vital signs" in reason:
        fields.add("vitals")
    if "shock case" in reason or "shock or sepsis" in reason:
        fields.add("vitals")
    if "risk" in reason:
        fields.add("risk_factors")
    if "exam" in reason:
        fields.add("exam_findings")
    if "progression" in reason:
        fields.add("progression")
    return fields or {"symptoms", "vitals", "progression", "risk_factors", "exam_findings"}


def merge_allowed_corrections(
    original_case: ClinicalCase,
    proposed_case: ClinicalCase,
    allowed_fields: set[str],
) -> ClinicalCase:
    merged = original_case.model_dump()
    proposed = proposed_case.model_dump()
    for field in allowed_fields:
        if field in proposed:
            merged[field] = proposed[field]
    return ClinicalCase.model_validate(merged)
