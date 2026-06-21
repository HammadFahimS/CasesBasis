from __future__ import annotations

import logging

from agents.ecg import ECGParameterAgent
from backend.ecg_synthesizer import synthesize_ecg_svg
from backend.ptbxl_service import get_ptbxl_service
from schemas.ecg import ALL_LEADS, ECGParameters, LeadFindings
from schemas.investigation import ECGVisualPayload, LeadIIMorphology

logger = logging.getLogger(__name__)

_RHYTHM_LABELS = {
    "sinus": "NORMAL SINUS RHYTHM",
    "sinus_tachycardia": "SINUS TACHYCARDIA",
    "sinus_bradycardia": "SINUS BRADYCARDIA",
    "af": "ATRIAL FIBRILLATION",
    "vt": "VENTRICULAR TACHYCARDIA",
    "vf": "VENTRICULAR FIBRILLATION",
    "svt": "SVT",
    "flutter": "ATRIAL FLUTTER",
    "complete_heart_block": "COMPLETE HEART BLOCK",
}


def _lead_ii_morphology(params: ECGParameters) -> LeadIIMorphology:
    lead_ii = params.leads.get("II") or LeadFindings()
    return LeadIIMorphology(
        hr=params.heart_rate,
        pr_ms=params.pr_interval_ms,
        qrs_ms=params.qrs_duration_ms,
        qt_ms=params.qtc_ms,
        p_absent=lead_ii.p_wave_absent,
        q_path=lead_ii.pathological_q,
        t_inv=lead_ii.t_inverted,
        qrs=lead_ii.qrs_morphology,
        st_mv=lead_ii.st_elevation_mv,
        amp_scale=lead_ii.amplitude_scale,
        rhythm_label=_RHYTHM_LABELS.get(params.rhythm, params.rhythm.upper()),
    )

# Shelf the PTB-XL retrieval path for now — the doctor reviewer prefers the
# LLM+numpy synthetic ECGs (the same ones exposed at /ecg-algorithm) because
# real PTB-XL records are heterogeneous and often show chronic / artifacted
# tracings that don't match the acute case description. The whole PTB-XL
# pipeline (index, FAISS, renderer, source_record toggle) is left untouched
# — flip this flag back to False to re-enable it instantly.
_PTBXL_DISABLED = True


class ECGService:
    """ECG visualization service.

    Primary path (current): LLM-generated `ECGParameters` → numpy synthesizer
    SVG — the same pipeline that powers the `/ecg-algorithm` review page.

    Shelved path (re-enable by setting `_PTBXL_DISABLED = False`):
    real PTB-XL waveform → matplotlib SVG. Indexes, FAISS retrieval, sample-rate
    toggle, and all related plumbing are left intact.
    """

    def __init__(self, agent: ECGParameterAgent | None = None) -> None:
        self._agent = agent

    def _get_agent(self) -> ECGParameterAgent:
        if self._agent is None:
            self._agent = ECGParameterAgent()
        return self._agent

    def generate(self, diagnosis: str, *, heart_rate: int | None = None) -> tuple[ECGParameters, str]:
        """Run the LLM + numpy synthesizer pipeline and return (params, svg)."""
        params = self._get_agent().generate(diagnosis, heart_rate=heart_rate)
        svg = synthesize_ecg_svg(params)
        return params, svg

    def build_visual_payload(
        self,
        diagnosis: str,
        *,
        heart_rate: int | None = None,
        age: int | None = None,
        sex: str | None = None,
    ) -> ECGVisualPayload:
        # ── Shelved PTB-XL path ────────────────────────────────────────────
        if not _PTBXL_DISABLED:
            ptbxl = get_ptbxl_service()
            if ptbxl.enabled:
                try:
                    rendered = ptbxl.render_paper_ecg(
                        diagnosis, heart_rate=heart_rate, age=age, sex=sex,
                    )
                except Exception as exc:
                    logger.warning("PTB-XL render failed for %r: %s", diagnosis, exc)
                    rendered = None
                if rendered is not None:
                    svg, record_id = rendered
                    return ECGVisualPayload(
                        type="svg",
                        leads=ALL_LEADS,
                        svg=svg,
                        source_record=record_id,
                    )

        # ── Current primary path: LLM + numpy synthesizer ─────────────────
        params: ECGParameters | None = None
        try:
            params, svg = self.generate(diagnosis, heart_rate=heart_rate)
        except Exception as exc:
            logger.warning("ECG synthesis failed for %r: %s", diagnosis, exc)
            svg = _fallback_svg()
        morph = _lead_ii_morphology(params) if params is not None else None
        return ECGVisualPayload(
            type="svg",
            leads=ALL_LEADS,
            svg=svg,
            lead_ii_morphology=morph,
        )


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
