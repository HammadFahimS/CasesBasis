from __future__ import annotations

import json
from typing import Any

from agents.base import OpenAIJsonAgent
from schemas.simulation import StudentAction

STUDENT_AGENT_SYSTEM_PROMPT = """
You are an AI student in a clinical simulation.

Rules:
- Return valid JSON only.
- Choose exactly one next action.
- Use only the visible context provided.
- You do not know the hidden diagnosis, outcome, or full hidden symptom list.
- Act like a trainee, not a perfect physician.
- Do not invent results that have not been revealed.
- On easy difficulty, ask obvious high-yield questions and interpret clear results.
- On medium difficulty, ask reasonable but imperfect questions and occasionally miss subtleties.
- On hard difficulty, you may jump early, miss subtle clues, or choose a lower-yield step.

Allowed actions:
- ask_patient_question
- order_investigation
- submit_interpretation
- submit_diagnosis
- call_doctor
- finish
""".strip()

STUDENT_AGENT_PROMPT_TEMPLATE = """
Difficulty:
{difficulty}

Step number:
{step_number}

Visible context:
{visible_context_json}

Return this JSON structure:
{
  "action": "ask_patient_question | order_investigation | submit_interpretation | submit_diagnosis | call_doctor | finish",
  "patient_question": "string or null",
  "investigation": "string or null",
  "interpretation": "string or null",
  "diagnosis_guess": "string or null",
  "consult_question": "string or null"
}

Return JSON only.
""".strip()


class StudentAgent(OpenAIJsonAgent):
    """Chooses trainee-like next actions from student-visible case context."""

    def next_action(
        self,
        visible_context: dict[str, Any],
        *,
        difficulty: str,
        step_number: int,
    ) -> StudentAction:
        prompt = STUDENT_AGENT_PROMPT_TEMPLATE.replace(
            "{difficulty}",
            difficulty,
        ).replace(
            "{step_number}",
            str(step_number),
        ).replace(
            "{visible_context_json}",
            json.dumps(visible_context, indent=2, sort_keys=True),
        )
        try:
            return self.parse_response(
                system_prompt=STUDENT_AGENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=StudentAction,
            )
        except ValueError:
            return StudentAction(action="finish")
