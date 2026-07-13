"""Expand an authored (prose) investigation result into structured value lines.

The live sim's cases author each test's result as text; some are already numeric
("pH 7.31, pCO2 55…"), others are prose ("CBC… within normal limits"). To render a
real lab table (and to never show a bare "all normal"), this agent turns that
authored summary into the component value lines a report would list — faithful to
the summary, never inventing a diagnosis or interpretation.

Pinned to gpt-oss-120b on Cerebras (structured output; same rationale as the ECG
agent), regardless of the live-sim ``CEREBRAS_MODEL`` default (gemma-4-31b).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from agents.base import CEREBRAS_MANUAL_MODEL, OpenAIJsonAgent, get_llm_provider
from agents.prompt_loader import load, render


class GeneratedLabValue(BaseModel):
    """One structured result line. Permissive (``extra="ignore"``) so a stray key
    from the model never fails the parse; ``value`` is coerced to a string so a
    numeric literal is accepted."""

    model_config = ConfigDict(extra="ignore")

    name: str
    value: str = ""
    unit: str = ""
    reference_range: str = ""
    flag: str | None = None

    @field_validator("value", "unit", "reference_range", mode="before")
    @classmethod
    def _to_str(cls, v: object) -> str:
        return "" if v is None else str(v)

    @field_validator("flag", mode="before")
    @classmethod
    def _norm_flag(cls, v: object) -> str | None:
        if v in (None, "", "null"):
            return None
        f = str(v).strip().lower()
        return f if f in {"low", "high", "critical", "abnormal", "normal"} else None


class GeneratedLabPanel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    values: list[GeneratedLabValue] = []


class LabValueAgent(OpenAIJsonAgent):
    """Turns a test name + authored result summary into structured value lines."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_provider = provider or get_llm_provider()
        if model is None and resolved_provider == "cerebras":
            model = CEREBRAS_MANUAL_MODEL
        super().__init__(client=client, model=model, provider=resolved_provider)

    def generate(
        self,
        name: str,
        authored_result: str,
        *,
        age: int | None = None,
        sex: str | None = None,
    ) -> list[GeneratedLabValue]:
        user_prompt = render(
            load("lab_values", "user"),
            name=name,
            authored=(authored_result or "").strip() or "(not specified)",
            age=str(age) if age is not None else "adult",
            sex=sex or "unspecified",
        )
        panel = self.parse_response(
            system_prompt=load("lab_values", "system"),
            user_prompt=user_prompt,
            response_model=GeneratedLabPanel,
        )
        # Keep only lines that actually carry a value.
        return [v for v in panel.values if v.name.strip() and v.value.strip()]
