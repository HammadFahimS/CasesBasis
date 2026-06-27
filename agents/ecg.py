from __future__ import annotations

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load
from schemas.ecg import ALL_LEADS, ECGParameters, LeadFindings


class ECGParameterAgent(OpenAIJsonAgent):
    """Maps a diagnosis string to validated ECGParameters via structured LLM output."""

    def generate(self, diagnosis: str, *, heart_rate: int | None = None) -> ECGParameters:
        user_prompt = f"Generate 12-lead ECG parameters for: {diagnosis}"
        if heart_rate is not None:
            user_prompt += f"\n\nIMPORTANT: The patient's actual heart rate is {heart_rate} bpm. Set heart_rate to exactly {heart_rate} in your output."
        params = self.parse_response(
            system_prompt=load("ecg", "system"),
            user_prompt=user_prompt,
            response_model=ECGParameters,
        )
        for lead in ALL_LEADS:
            if lead not in params.leads:
                params.leads[lead] = LeadFindings()
        # Enforce case-level HR as ground truth regardless of what the LLM chose
        if heart_rate is not None:
            params.heart_rate = heart_rate
        return params
