from __future__ import annotations

import json
from typing import Any

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from schemas.simulation_audit import SimulationAudit


class SimulationAuditorAgent(OpenAIJsonAgent):
    """Audits completed simulation runs for internal QA findings."""

    def audit(self, audit_context: dict[str, Any]) -> SimulationAudit:
        prompt = render(
            load("simulation_auditor", "user"),
            audit_context_json=json.dumps(audit_context, indent=2, sort_keys=True),
        )
        return self.parse_response(
            system_prompt=load("simulation_auditor", "system"),
            user_prompt=prompt,
            response_model=SimulationAudit,
        )
