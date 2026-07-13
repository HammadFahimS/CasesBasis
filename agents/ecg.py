from __future__ import annotations

from typing import Any

from agents.base import CEREBRAS_ECG_MODEL, OpenAIJsonAgent, get_llm_provider
from agents.prompt_loader import load
from schemas.ecg import ALL_LEADS, ECGParameters, LeadFindings


class ECGParameterAgent(OpenAIJsonAgent):
    """Maps a diagnosis string to validated ECGParameters via structured LLM output.

    ECG generation is pinned to gpt-oss-120b on Cerebras (the live sim runs on
    gemma-4-31b; the manual sim also uses gpt-oss-120b). When the provider is
    Cerebras and no model is explicitly given, we force the ECG model so this
    stays true regardless of the ``CEREBRAS_MODEL`` env default."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_provider = provider or get_llm_provider()
        if model is None and resolved_provider == "cerebras":
            model = CEREBRAS_ECG_MODEL
        super().__init__(client=client, model=model, provider=resolved_provider)

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
