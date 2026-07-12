"""Shared core for the classic (manual) simulator workbench.

The in-memory session store, the lazily-built service singletons, and the
helpers that turn engine facts into student-facing responses used to live inline
in ``web_app``. They are shared by several route modules — the classic workbench
API (``routers.simulator``), the dashboard's session-detail / pause / resume
endpoints (``routers.dashboard``), and the live voice routes (``routers.voice``)
that reuse ``_get_owned_session`` / ``get_graph``. Extracting them here lets each
router depend on the core without importing ``web_app`` (which would cycle).

A leaf module: it imports only schemas, services, persistence and auth — never a
router or ``web_app``. ``web_app`` re-exports these names so existing call sites
and the external importers (``persistence``, ``simulation_runner``) are
unaffected. The :class:`~backend.simulation_engine.SimulationEngine` remains the
single source of clinical truth; nothing here invents content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from fastapi import HTTPException, Request

from agents.base import manual_sim_model
from agents.physical_exam_agent import PhysicalExamAgent
from backend.auth import get_current_resident_id
from backend.clinical_evaluator import ClinicalEvaluator
from backend.doctor_consult import DoctorConsultService
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.persistence import load_session
from backend.service import ClinicalCaseSimulator
from knowledge.graph import ClinicalKnowledgeGraph
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.investigation import InvestigationResult
from schemas.physical_exam import ExamFinding


@dataclass
class WebSession:
    case: ClinicalCase
    resident_id: str | None = None
    student_questions: list[str] = field(default_factory=list)
    patient_responses: list[str] = field(default_factory=list)
    investigations: list[InvestigationResult] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    doctor_consults: list[DoctorConsultResponse] = field(default_factory=list)
    diagnosis_guess: str | None = None
    physical_exams: list[ExamFinding] = field(default_factory=list)
    revealed_symptoms: list[str] = field(default_factory=list)


SESSIONS: dict[str, WebSession] = {}
CASE_CACHE_TTL_SECONDS = 60
CASE_CACHE: dict[tuple[str, str], tuple[float, ClinicalCase]] = {}
SIMULATOR: ClinicalCaseSimulator | None = None
PATIENT_SERVICE: PatientResponseService | None = None
INVESTIGATION_SERVICE: InvestigationService | None = None
DOCTOR_CONSULT_SERVICE: DoctorConsultService | None = None
EVALUATOR: ClinicalEvaluator | None = None
GRAPH: ClinicalKnowledgeGraph | None = None
PHYSICAL_EXAM_AGENT: PhysicalExamAgent | None = None


def get_session(session_id: str) -> WebSession:
    session = SESSIONS.get(session_id)
    if session is None:
        session = load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        SESSIONS[session_id] = session
    return session


def _get_owned_session(session_id: str, http_request: Request) -> WebSession:
    """Return the session only if it belongs to the requesting resident.

    A session that has an owner may only be accessed by that owner — an
    unauthenticated or mismatched caller is denied. Anonymous sessions (no
    resident_id, e.g. created before login) stay open as a capability keyed by
    the unguessable session id.
    """
    session = get_session(session_id)
    resident_id = get_current_resident_id(http_request)
    if session.resident_id is not None and session.resident_id != resident_id:
        raise HTTPException(status_code=403, detail="access denied")
    return session


def get_simulator() -> ClinicalCaseSimulator:
    global SIMULATOR
    if SIMULATOR is None:
        SIMULATOR = ClinicalCaseSimulator()
    return SIMULATOR


# Every LLM agent the manual simulator drives is pinned to gpt-oss-120b on
# Cerebras (see agents.base.manual_sim_model): the live-sim default zai-glm-4.7
# can't reliably satisfy the manual sim's strict schemas. Non-Cerebras providers
# get None (their own default). The live voice sim uses none of these services.
def get_patient_service() -> PatientResponseService:
    global PATIENT_SERVICE
    if PATIENT_SERVICE is None:
        PATIENT_SERVICE = PatientResponseService(model=manual_sim_model())
    return PATIENT_SERVICE


def get_investigation_service() -> InvestigationService:
    global INVESTIGATION_SERVICE
    if INVESTIGATION_SERVICE is None:
        INVESTIGATION_SERVICE = InvestigationService(model=manual_sim_model())
    return INVESTIGATION_SERVICE


def get_doctor_consult_service() -> DoctorConsultService:
    global DOCTOR_CONSULT_SERVICE
    if DOCTOR_CONSULT_SERVICE is None:
        DOCTOR_CONSULT_SERVICE = DoctorConsultService(model=manual_sim_model())
    return DOCTOR_CONSULT_SERVICE


def get_physical_exam_agent() -> PhysicalExamAgent:
    global PHYSICAL_EXAM_AGENT
    if PHYSICAL_EXAM_AGENT is None:
        PHYSICAL_EXAM_AGENT = PhysicalExamAgent(model=manual_sim_model())
    return PHYSICAL_EXAM_AGENT


def get_evaluator() -> ClinicalEvaluator:
    global EVALUATOR
    if EVALUATOR is None:
        EVALUATOR = ClinicalEvaluator(model=manual_sim_model())
    return EVALUATOR


def get_graph() -> ClinicalKnowledgeGraph | None:
    global GRAPH
    if GRAPH is None:
        GRAPH = ClinicalKnowledgeGraph()
    return GRAPH


def get_or_generate_cached_case(diagnosis: str, difficulty: str) -> ClinicalCase:
    key = cache_key(diagnosis, difficulty)
    cached = CASE_CACHE.get(key)
    now = monotonic()
    if cached and now - cached[0] <= CASE_CACHE_TTL_SECONDS:
        return cached[1].model_copy(deep=True)

    clinical_case = get_simulator().generate_case(
        diagnosis,
        difficulty=difficulty,
        session_history=[],
    )
    # Evict stale entries before writing so the dict stays bounded,
    # but don't clear entries for other keys that are still fresh.
    stale = [k for k, (ts, _) in CASE_CACHE.items() if now - ts > CASE_CACHE_TTL_SECONDS]
    for k in stale:
        del CASE_CACHE[k]
    CASE_CACHE[key] = (now, clinical_case.model_copy(deep=True))
    return clinical_case


def cache_key(diagnosis: str, difficulty: str) -> tuple[str, str]:
    return (" ".join(diagnosis.split()).strip().lower(), difficulty.strip().lower())


def case_preview(case: ClinicalCase) -> dict[str, Any]:
    return {
        "age": case.patient.age,
        "gender": case.patient.gender,
        "name": case.patient.name,
        "race": case.patient.race,
        "occupation": case.patient.occupation,
        "persona_description": case.patient.persona_description,
        "chief_complaint": case.symptoms[0] if case.symptoms else "",
        "vitals": case.vitals.model_dump(),
    }


def _full_state_response(session: WebSession, session_id: str, status: str, duration_seconds: int = 0) -> dict[str, Any]:
    qa_pairs = [
        {"question": q, "answer": a}
        for q, a in zip(session.student_questions, session.patient_responses)
    ]
    return {
        "session_id": session_id,
        "status": status,
        "case_preview": case_preview(session.case),
        "qa_pairs": qa_pairs,
        "investigations": [
            student_facing_investigation_result(inv, session.case)
            for inv in session.investigations
        ],
        "interpretations": session.interpretations,
        "diagnosis_guess": session.diagnosis_guess,
        "duration_seconds": duration_seconds,
    }


def student_facing_investigation_result(
    result: InvestigationResult,
    case: ClinicalCase,
) -> dict[str, Any]:
    if result.status == "unavailable":
        return {
            "investigation": result.investigation,
            "status": result.status,
            "summary": result.summary,
            "report": [],
            "visual_payload": None,
        }
    if is_ecg_display(result):
        return {
            "investigation": result.investigation,
            "status": result.status,
            "summary": "ECG completed",
            "report": [],
            "visual_payload": result.visual_payload.model_dump() if result.visual_payload else None,
        }
    if is_cxr_display(result):
        return {
            "investigation": result.investigation,
            "status": result.status,
            "summary": result.summary,
            "report": [],
            "visual_payload": result.visual_payload.model_dump() if result.visual_payload else None,
        }
    return {
        "investigation": result.investigation,
        "status": result.status,
        "summary": f"{result.investigation} completed",
        "report": raw_investigation_report(result, case),
        "visual_payload": None,
    }


def is_ecg_display(result: InvestigationResult) -> bool:
    return (
        result.visual_payload is not None and hasattr(result.visual_payload, "svg")
    ) or result.investigation.strip().lower() in {"ecg", "ekg"}


def is_cxr_display(result: InvestigationResult) -> bool:
    return (
        result.visual_payload is not None and hasattr(result.visual_payload, "data")
    ) or result.investigation.strip().lower() in {"cxr", "chest x-ray", "chest xray"}


def raw_investigation_report(result: InvestigationResult, case: ClinicalCase) -> list[dict[str, Any]]:
    structured = result.structured_result
    values = structured.get("values")
    if isinstance(values, list):
        return [
            clean_lab_value(value, case)
            for value in values
            if isinstance(value, dict)
        ]

    if all(key in structured for key in ("value", "unit", "reference_range")):
        return [
            clean_lab_value(
                {
                    "name": structured.get("test", result.investigation),
                    "value": structured.get("value"),
                    "unit": structured.get("unit"),
                    "reference_range": structured.get("reference_range"),
                },
                case,
            )
        ]

    findings = structured.get("findings")
    if isinstance(findings, list):
        return [
            {"finding": redact_hidden_terms(str(finding), case)}
            for finding in findings
            if str(finding).strip()
        ]
    return []


def clean_lab_value(value: dict[str, Any], case: ClinicalCase) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in ("name", "value", "unit", "reference_range"):
        if key in value:
            item = value[key]
            cleaned[key] = redact_hidden_terms(str(item), case) if isinstance(item, str) else item
    return cleaned


def redact_hidden_terms(value: str, case: ClinicalCase) -> str:
    diagnosis = case.diagnosis.strip()
    if not diagnosis:
        return value
    return re.sub(re.escape(diagnosis), "the condition", value, flags=re.IGNORECASE)
