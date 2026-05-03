from __future__ import annotations

import json
from typing import Any

from agents.base import OpenAIJsonAgent
from schemas.simulation_audit import SimulationAudit

SIMULATION_AUDITOR_SYSTEM_PROMPT = """
You are an internal QA auditor for a clinical simulation system.

You may see hidden case data because this is internal QA only.
Return valid JSON only using the SimulationAudit schema.

Classify each finding as:
- confirmed_bug
- possible_model_quality_issue
- expected_behavior

Check:
- invalid diagnosis guesses, including symptom-only guesses treated as diagnoses
- hidden diagnosis leakage
- patient revealing too much
- symptom identification accuracy
- diagnosis alias or partial correctness
- scoring fairness
- duplicate feedback or duplicate findings
- contradictory feedback
- investigation display leaks
- investigation interpretation scoring
- doctor consult behavior if used
- doctor consult revealing too much
- usefulness of feedback and recommendations

For each finding, include reasoning and concrete evidence from the run.
Suggest minimal fixes only for confirmed bugs.
Do not modify code.
""".strip()

SIMULATION_AUDITOR_PROMPT_TEMPLATE = """
Full internal simulation context:
{audit_context_json}

Return this JSON structure:
{
  "overall_quality": "pass | warning | fail",
  "findings": [
    {
      "category": "confirmed_bug | possible_model_quality_issue | expected_behavior",
      "severity": "low | medium | high",
      "issue": "string",
      "evidence": "string",
      "suggested_fix": "string"
    }
  ],
  "should_fix": true
}

Return JSON only.
""".strip()


class SimulationAuditorAgent(OpenAIJsonAgent):
    """Audits completed simulation runs for internal QA findings."""

    def audit(self, audit_context: dict[str, Any]) -> SimulationAudit:
        prompt = SIMULATION_AUDITOR_PROMPT_TEMPLATE.replace(
            "{audit_context_json}",
            json.dumps(audit_context, indent=2, sort_keys=True),
        )
        return self.parse_response(
            system_prompt=SIMULATION_AUDITOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=SimulationAudit,
        )
