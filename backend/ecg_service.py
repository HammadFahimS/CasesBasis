from __future__ import annotations

import logging

from agents.ecg import ECGParameterAgent
from backend.ecg_synthesizer import synthesize_ecg_svg
from schemas.ecg import ALL_LEADS, ECGParameters
from schemas.investigation import ECGVisualPayload

logger = logging.getLogger(__name__)


class ECGService:
    """Orchestrates ECG parameter generation (LLM) and waveform synthesis (numpy)."""

    def __init__(self, agent: ECGParameterAgent | None = None) -> None:
        self._agent = agent

    def _get_agent(self) -> ECGParameterAgent:
        if self._agent is None:
            self._agent = ECGParameterAgent()
        return self._agent

    def generate(self, diagnosis: str, *, heart_rate: int | None = None) -> tuple[ECGParameters, str]:
        """Return (ECGParameters, svg_string) for the given diagnosis."""
        params = self._get_agent().generate(diagnosis, heart_rate=heart_rate)
        svg = synthesize_ecg_svg(params)
        return params, svg

    def build_visual_payload(self, diagnosis: str, *, heart_rate: int | None = None) -> ECGVisualPayload:
        try:
            _, svg = self.generate(diagnosis, heart_rate=heart_rate)
        except Exception as exc:
            logger.warning("ECG synthesis failed for %r: %s", diagnosis, exc)
            svg = _fallback_svg()
        return ECGVisualPayload(type="svg", leads=ALL_LEADS, svg=svg)


def _fallback_svg() -> str:
    """Minimal placeholder SVG when synthesis fails."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="80">'
        '<rect width="100%" height="100%" fill="#fff8f8"/>'
        '<text x="50%" y="45" text-anchor="middle" font-family="Arial" font-size="13" fill="#888">'
        "ECG generation unavailable"
        "</text>"
        "</svg>"
    )
