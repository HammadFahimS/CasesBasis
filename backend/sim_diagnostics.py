"""Bridge from a simulated-case investigation to a renderable diagnostic card.

The authored ``InvestigationDef.result`` stays the clinical truth (gating and
state-specific values are owned by the engine). This module layers a *real*
visual on top for the diagnostics that have an image generator in the manual
simulator — ECG (``ECGService``) and chest X-ray (``CXRService``) — keyed on the
case's ``visual_dx`` hint plus the live vitals/demographics. Everything else
becomes a plain text report card.

Service calls are guarded: a generator hiccup degrades to a report card rather
than breaking the turn.
"""

from __future__ import annotations

import logging

from schemas.simulated_case import InvestigationDef, PatientProfile, VitalSigns

logger = logging.getLogger(__name__)

_CXR_TERMS = ("chest x-ray", "chest xray", "cxr", "chest radiograph", "chest film")

_ecg_service = None  # lazy singletons — building the FAISS/agent state is costly
_cxr_service = None
_lab_agent = None
# Generated panels are memoised per (test id, authored text, age, sex) so a repeat
# order shows the same numbers and we don't pay a second LLM call.
_lab_value_cache: dict[tuple, list[dict]] = {}

# Categories whose bare-prose result we expand into a structured value table when
# the case didn't author one.
_GENERATABLE = ("lab", "bedside")


def _ecg():
    global _ecg_service
    if _ecg_service is None:
        from backend.ecg_service import ECGService
        _ecg_service = ECGService()
    return _ecg_service


def _lab_values_agent():
    global _lab_agent
    if _lab_agent is None:
        from agents.lab_values import LabValueAgent
        _lab_agent = LabValueAgent()
    return _lab_agent


def _structured_values(inv: "InvestigationDef", authored: str, patient: "PatientProfile") -> list[dict]:
    """The lab table for a card: authored ``values`` if present, else a generated
    panel faithful to the authored prose (cached). Empty for non-lab categories or
    on any failure — the text summary still carries the result."""
    if inv.values:
        return [v.model_dump() for v in inv.values]
    if inv.category not in _GENERATABLE:
        return []
    key = (inv.id, authored, getattr(patient, "age", None), getattr(patient, "gender", None))
    if key in _lab_value_cache:
        return _lab_value_cache[key]
    try:
        gen = _lab_values_agent().generate(
            inv.name, authored, age=patient.age, sex=patient.gender,
        )
        rows = [v.model_dump() for v in gen]
    except Exception as exc:  # never let generation break the turn/card
        logger.warning("lab value generation failed for %s: %s", inv.id, exc)
        rows = []
    _lab_value_cache[key] = rows
    return rows


def _cxr():
    global _cxr_service
    if _cxr_service is None:
        from backend.cxr_service import CXRService
        _cxr_service = CXRService()
    return _cxr_service


def _is_cxr(inv: InvestigationDef) -> bool:
    name = inv.name.lower()
    return any(term in name for term in _CXR_TERMS)


def _dx_hint(inv: InvestigationDef) -> str:
    """The string handed to the image generator as the 'diagnosis'."""
    if inv.visual_dx:
        return inv.visual_dx
    if inv.result:
        return inv.result.split(".")[0].strip() or inv.name
    return inv.name


def build_investigation_card(
    inv: InvestigationDef,
    vitals: VitalSigns,
    patient: PatientProfile,
    authored: str,
) -> dict:
    """A drawer card for one ordered investigation.

    ``authored`` is the engine's revealed result text (kept as the clinical
    content); a real ECG/CXR image is attached when one can be generated."""
    card: dict = {
        "kind": "report",
        "id": inv.id,
        "title": inv.name,
        "category": inv.category,
        "summary": authored,
        # Structured result lines for a proper lab table (authored or generated).
        "values": _structured_values(inv, authored, patient),
        "visual_payload": None,
    }
    try:
        if inv.category == "ecg":
            payload = _ecg().build_visual_payload(
                _dx_hint(inv), heart_rate=vitals.heart_rate,
                age=patient.age, sex=patient.gender,
            )
            card.update(kind="ecg", visual_payload=payload.model_dump())
        elif inv.category == "imaging" and _is_cxr(inv):
            payload = _cxr().build_visual_payload(
                _dx_hint(inv), age=patient.age,
                gender=patient.gender, race=getattr(patient, "race", None),
            )
            if payload is not None:
                card.update(kind="cxr", visual_payload=payload.model_dump())
    except Exception as exc:  # never let a generator break the turn
        logger.warning("diagnostic visual failed for %s: %s", inv.id, exc)
    return card


def build_exam_card(system: str, revealed: str) -> dict:
    """A drawer card for an examined body system (authored findings)."""
    return {
        "kind": "exam",
        "id": system,
        "title": f"{system} exam",
        "category": "exam",
        "summary": revealed,
        "visual_payload": None,
    }


def build_order_card(
    drug: str,
    dose: str | None,
    detail: str = "",
    *,
    status: str = "given",
    confirm_required: bool = False,
    card_id: str = "",
) -> dict:
    """A drawer card for a medication order.

    ``status`` is ``"confirm"`` while the nurse is awaiting a dose on a high-alert
    agent, then ``"given"`` once it's administered. ``card_id`` is a stable slug so
    re-ordering the same drug upserts the existing card rather than duplicating it.
    """
    summary = drug + (f" — {dose}" if dose else "")
    if status == "confirm" and not dose:
        summary = f"{drug} — confirm dose before administration"
    elif detail and detail.lower() not in summary.lower():
        summary += f" ({detail})"
    return {
        "kind": "order",
        "id": card_id or drug,
        "title": drug,
        "category": "orders",
        "summary": summary,
        "dose": dose,
        "status": status,
        "confirm_required": confirm_required,
        "visual_payload": None,
    }


def build_consult_card(consultant: str, advice: str, consult_id: str = "") -> dict:
    """A drawer card for a specialist consult (the consultant's recommendation)."""
    return {
        "kind": "consult",
        "id": consult_id or consultant,
        "title": consultant,
        "category": "consult",
        "summary": advice,
        "visual_payload": None,
    }
