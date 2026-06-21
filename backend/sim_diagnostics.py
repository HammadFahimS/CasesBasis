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


def _ecg():
    global _ecg_service
    if _ecg_service is None:
        from backend.ecg_service import ECGService
        _ecg_service = ECGService()
    return _ecg_service


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
