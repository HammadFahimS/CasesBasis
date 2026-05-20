from __future__ import annotations

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation, ReasoningFeedback


class ReasoningReviewAgent(OpenAIJsonAgent):
    """Optionally refines reasoning feedback without changing scores."""

    def review(
        self,
        *,
        case: ClinicalCase,
        evaluation: ClinicalEvaluation,
        student_questions: list[str],
        patient_responses: list[str],
    ) -> ReasoningFeedback:
        prompt = f"""
Visible case context:
{visible_case_context(case)}

Observed student questions:
{student_questions}

Observed patient responses:
{patient_responses}

Current evaluation:
{evaluation.model_dump_json(indent=2)}

Return improved reasoning_feedback only.
""".strip()
        return self.parse_response(
            system_prompt=load("reasoning_reviewer", "system"),
            user_prompt=prompt,
            response_model=ReasoningFeedback,
        )


def visible_case_context(case: ClinicalCase) -> str:
    return case.model_dump_json(exclude={"diagnosis", "outcome"}, indent=2)
