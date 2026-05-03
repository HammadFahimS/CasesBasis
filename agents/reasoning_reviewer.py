from __future__ import annotations

from agents.base import OpenAIJsonAgent
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation, ReasoningFeedback

REASONING_REVIEW_SYSTEM_PROMPT = """
You are reviewing reasoning feedback for a medical simulation evaluation.

Task:
- refine reasoning_feedback
- improve explanation clarity
- identify subtle reasoning strengths and weaknesses

Rules:
- return only the ReasoningFeedback JSON structure
- do NOT change score
- do NOT change any evaluation fields except reasoning_feedback text
- do NOT expose hidden case data directly
- base feedback only on the observed interaction and provided case context
- keep feedback concise and structured
""".strip()


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
            system_prompt=REASONING_REVIEW_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=ReasoningFeedback,
        )


def visible_case_context(case: ClinicalCase) -> str:
    return case.model_dump_json(exclude={"diagnosis", "outcome"}, indent=2)
