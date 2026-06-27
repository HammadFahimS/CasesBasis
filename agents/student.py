from __future__ import annotations

import json
from typing import Any

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from schemas.simulation import StudentAction


class StudentAgent(OpenAIJsonAgent):
    """Chooses trainee-like next actions from student-visible case context."""

    def next_action(
        self,
        visible_context: dict[str, Any],
        *,
        difficulty: str,
        step_number: int,
    ) -> StudentAction:
        prompt = render(
            load("student", "user"),
            difficulty=difficulty,
            step_number=str(step_number),
            visible_context_json=json.dumps(visible_context, indent=2, sort_keys=True),
        )
        try:
            return self.parse_response(
                system_prompt=load("student", "system"),
                user_prompt=prompt,
                response_model=StudentAction,
            )
        except ValueError:
            return StudentAction(action="finish")
