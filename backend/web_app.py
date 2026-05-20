from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from uuid import uuid4

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pathlib import Path

from pydantic import BaseModel

from agents.physical_exam_agent import PhysicalExamAgent
from backend.auth import (
    COOKIE_NAME,
    generate_magic_token,
    get_current_resident_id,
    make_session_cookie,
    verify_magic_token,
)
from backend.clinical_evaluator import ClinicalEvaluator
from backend.db import run_migrations
from backend.doctor_consult import DoctorConsultService
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.persistence import (
    consume_magic_link,
    create_resident_with_code,
    create_session,
    delete_session_for_resident,
    find_resident_by_code,
    get_or_create_resident,
    get_resident,
    get_resident_sessions,
    get_resident_stats,
    get_session_evaluation,
    get_session_for_resident,
    load_session,
    log_event,
    get_session_duration,
    get_recent_evaluations_for_coaching,
    pause_session,
    save_evaluation,
    save_session,
    store_magic_link,
)
from backend.service import ClinicalCaseSimulator
from knowledge.graph import ClinicalKnowledgeGraph
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.investigation import InvestigationResult
from schemas.physical_exam import ExamFinding, PhysicalExamRequest

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import gc
    import logging
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_migrations()

    from backend.cxr_service import _faiss_index
    logger = logging.getLogger(__name__)
    logger.info("Warming up CXR FAISS index ...")
    await asyncio.get_event_loop().run_in_executor(None, _faiss_index._load)
    logger.info("CXR FAISS index ready")

    yield

    # clean up torch thread-pool semaphores on shutdown
    _faiss_index._model = None
    gc.collect()


app = FastAPI(title="Clinical Case Simulator", lifespan=lifespan)


@dataclass
class WebSession:
    case: ClinicalCase
    student_questions: list[str] = field(default_factory=list)
    patient_responses: list[str] = field(default_factory=list)
    investigations: list[InvestigationResult] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    doctor_consults: list[DoctorConsultResponse] = field(default_factory=list)
    diagnosis_guess: str | None = None
    physical_exams: list[ExamFinding] = field(default_factory=list)
    revealed_symptoms: list[str] = field(default_factory=list)


class RequestLinkRequest(BaseModel):
    email: str


class ResidentLoginRequest(BaseModel):
    resident_code: str
    last_name: str


class GenerateCaseRequest(BaseModel):
    diagnosis: str
    difficulty: str = "medium"


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class InvestigationRequest(BaseModel):
    session_id: str
    investigation: str


class InterpretationRequest(BaseModel):
    session_id: str
    interpretation: str


class DiagnosisRequest(BaseModel):
    session_id: str
    diagnosis: str


class DoctorConsultRequest(BaseModel):
    session_id: str
    question: str


class EvaluateRequest(BaseModel):
    session_id: str
    elapsed_seconds: int = 0


class PauseRequest(BaseModel):
    elapsed_seconds: int = 0


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


@app.get("/api/graph/{session_id}")
def get_graph_data(session_id: str) -> dict[str, Any]:
    get_session(session_id)
    try:
        import falkordb as _falkordb
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        db = _falkordb.FalkorDB(host=host, port=port)
        graph = db.select_graph(f"session:{session_id}")

        node_res = graph.query("MATCH (n) RETURN id(n), n.name, labels(n)")
        nodes = []
        seen_ids: set[str] = set()
        for row in node_res.result_set or []:
            nid, name, lbls = str(row[0]), row[1], row[2] or []
            if not name or nid in seen_ids:
                continue
            seen_ids.add(nid)
            nodes.append({"id": nid, "name": name, "labels": [l for l in lbls if l]})

        edge_res = graph.query(
            "MATCH (n)-[r]->(m) RETURN id(n), id(m), type(r)"
        )
        edges = []
        for row in edge_res.result_set or []:
            src, tgt, rel = str(row[0]), str(row[1]), row[2] or ""
            edges.append({"source": src, "target": tgt, "fact": rel.replace("_", " ").lower()})

        return {"nodes": nodes, "edges": edges}
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}


@app.get("/body-image.png")
def body_image() -> FileResponse:
    img_path = Path(__file__).parent.parent / "generatedimageofbody.png"
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/static/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(str(Path(__file__).parent / "static" / "favicon.svg"), media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "login.html").read_text()


@app.get("/simulator", response_class=HTMLResponse)
def simulator_page() -> str:
    return HTML_PAGE


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page() -> str:
    return (Path(__file__).parent / "static" / "dashboard.html").read_text()


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/auth/request-link")
def request_magic_link(body: RequestLinkRequest) -> dict[str, str]:
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="valid email required")

    ttl_minutes = int(os.getenv("MAGIC_LINK_TTL_MINUTES", "15"))
    token = generate_magic_token(email)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
    resident_id = get_or_create_resident(email)
    store_magic_link(token, resident_id, expires_at)

    link = f"/api/auth/verify?token={token}"
    _logger.info("Magic link for %s: %s", email, link)

    return {"message": "Check your email for a login link.", "dev_link": link}


@app.get("/api/auth/verify")
def verify_magic_link(token: str) -> RedirectResponse:
    ttl_minutes = int(os.getenv("MAGIC_LINK_TTL_MINUTES", "15"))
    email = verify_magic_token(token, ttl_minutes * 60)
    if email is None:
        raise HTTPException(status_code=400, detail="invalid or expired link")

    resident_id = consume_magic_link(token)
    if resident_id is None:
        raise HTTPException(status_code=400, detail="link already used or not found")

    cookie_value = make_session_cookie(resident_id)
    redirect = RedirectResponse(url="/dashboard", status_code=302)
    redirect.set_cookie(
        COOKIE_NAME,
        cookie_value,
        httponly=True,
        samesite="strict",
        max_age=30 * 24 * 3600,
    )
    return redirect


@app.post("/api/auth/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.post("/api/auth/test-login")
def test_login(response: Response) -> dict[str, Any]:
    resident_id = get_or_create_resident("test@casesbasis.dev")
    _set_auth_cookie(response, resident_id)
    return {"ok": True, "resident_id": resident_id}


@app.post("/api/auth/login")
def login_resident(body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
    resident_id = find_resident_by_code(body.resident_code, body.last_name)
    if resident_id is None:
        raise HTTPException(status_code=404, detail="not_found")
    _set_auth_cookie(response, resident_id)
    return {"ok": True}


@app.post("/api/auth/signup")
def signup_resident(body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
    existing = find_resident_by_code(body.resident_code, body.last_name)
    if existing:
        _set_auth_cookie(response, existing)
        return {"ok": True, "existed": True}
    resident_id = create_resident_with_code(body.resident_code, body.last_name)
    _set_auth_cookie(response, resident_id)
    return {"ok": True, "existed": False}


def _set_auth_cookie(response: Response, resident_id: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        make_session_cookie(resident_id),
        httponly=True,
        samesite="strict",
        max_age=30 * 24 * 3600,
    )


@app.get("/api/auth/me")
def me(request: Request) -> dict[str, str | None]:
    resident_id = get_current_resident_id(request)
    if resident_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    resident = get_resident(resident_id)
    if resident is None:
        raise HTTPException(status_code=401, detail="resident not found")
    return {
        "resident_id": resident.id,
        "email": resident.email,
        "display_name": resident.display_name,
    }


def require_resident(request: Request) -> str:
    resident_id = get_current_resident_id(request)
    if resident_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return resident_id


# ── Dashboard endpoints ────────────────────────────────────────────────────────

@app.get("/api/dashboard/sessions")
def dashboard_sessions(http_request: Request) -> dict[str, Any]:
    resident_id = require_resident(http_request)
    sessions = get_resident_sessions(resident_id)
    return {"sessions": sessions}


@app.get("/api/dashboard/stats")
def dashboard_stats(http_request: Request) -> dict[str, Any]:
    resident_id = require_resident(http_request)
    return get_resident_stats(resident_id)


@app.get("/api/dashboard/coaching-insight")
async def dashboard_coaching_insight(http_request: Request) -> dict[str, Any]:
    import asyncio
    from agents.base import build_llm_client, get_llm_provider, resolve_model

    resident_id = require_resident(http_request)
    evaluations = get_recent_evaluations_for_coaching(resident_id, limit=6)
    if len(evaluations) < 2:
        return {"insight": None, "has_data": False}

    summaries: list[str] = []
    for i, ev in enumerate(evaluations, 1):
        e = ev["evaluation"]
        missed = e.get("key_symptoms_missed", [])
        gaps = (e.get("reasoning_feedback") or {}).get("missed_reasoning_steps", [])
        correct = (e.get("correct_diagnosis_guess") or {}).get("is_correct", False)
        brief = (e.get("brief_feedback") or "")[:180]
        lines = [
            f"Case {i}: {ev['diagnosis']} · score {ev['overall_score']}/100 · diagnosis {'✓' if correct else '✗'}",
        ]
        if missed:
            lines.append(f"  Missed: {', '.join(missed[:5])}")
        if gaps:
            lines.append(f"  Gaps: {'; '.join(gaps[:2])}")
        if brief:
            lines.append(f"  Feedback: {brief}")
        summaries.append("\n".join(lines))

    system_prompt = (
        "You are a clinical education coach reviewing a medical resident's case performance. "
        "Be specific, direct, and actionable. Reference actual patterns from the data. Never give generic advice."
    )
    user_prompt = (
        f"Here are the last {len(evaluations)} evaluated cases for this resident:\n\n"
        + "\n\n".join(summaries)
        + "\n\nIdentify the single most important pattern or recurring gap in their clinical reasoning. "
        "Write a 2–3 sentence coaching insight that names the specific symptom, step, or reasoning failure you observe, "
        "and ends with one concrete action they can take in their next case. "
        "Output only the coaching insight text. No headers, no bullet points, no markdown."
    )

    try:
        provider = get_llm_provider()
        client = build_llm_client(provider)
        model = resolve_model(provider, None)
        insight = await asyncio.to_thread(
            client.text_response,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return {"insight": insight.strip(), "has_data": True}
    except Exception as exc:
        logger.warning("coaching insight LLM failed: %s", exc)
        return {"insight": None, "has_data": False}


@app.get("/api/dashboard/session/{session_id}")
def dashboard_session_detail(session_id: str, http_request: Request) -> dict[str, Any]:
    resident_id = require_resident(http_request)
    row = get_session_for_resident(session_id, resident_id)
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    result = _full_state_response(session, session_id, row["status"], duration_seconds=row.get("duration_seconds") or 0)
    result["evaluation_data"] = get_session_evaluation(session_id)
    return result


@app.delete("/api/dashboard/session/{session_id}")
def dashboard_delete_session(session_id: str, http_request: Request) -> dict[str, bool]:
    resident_id = require_resident(http_request)
    deleted = delete_session_for_resident(session_id, resident_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    SESSIONS.pop(session_id, None)
    log_event(session_id, "session.deleted", resident_id=resident_id)
    return {"ok": True}


# ── Pause / resume endpoints ───────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/pause")
def pause(session_id: str, body: PauseRequest) -> dict[str, bool]:
    get_session(session_id)  # raises 404 if unknown
    graph = get_graph()
    snapshot = graph.get_context(session_id) if graph and graph.enabled else None
    pause_session(session_id, snapshot, body.elapsed_seconds)
    log_event(session_id, "session.paused", {"elapsed_seconds": body.elapsed_seconds})
    return {"ok": True}


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str) -> dict[str, Any]:
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    SESSIONS[session_id] = session
    log_event(session_id, "session.resumed")

    from backend.db import get_db as _get_db
    conn = _get_db()
    try:
        row = conn.execute("SELECT status, duration_seconds FROM web_sessions WHERE id=?", (session_id,)).fetchone()
        status = row["status"] if row else "active"
        duration_seconds = row["duration_seconds"] if row else 0
    finally:
        conn.close()

    return _full_state_response(session, session_id, status, duration_seconds=duration_seconds or 0)


@app.post("/api/cases")
def generate_case(request: GenerateCaseRequest, http_request: Request) -> dict[str, Any]:
    resident_id = get_current_resident_id(http_request)
    clinical_case = get_or_generate_cached_case(request.diagnosis, request.difficulty)
    session_id = str(uuid4())
    SESSIONS[session_id] = WebSession(case=clinical_case)
    create_session(session_id, request.diagnosis, request.difficulty, clinical_case, resident_id=resident_id)
    log_event(session_id, "session.created", {"diagnosis": request.diagnosis, "difficulty": request.difficulty}, resident_id=resident_id)
    graph = get_graph()
    if graph and graph.enabled:
        graph.init_session(session_id, clinical_case)
    return {
        "session_id": session_id,
        "case_preview": case_preview(clinical_case),
    }


@app.post("/api/questions")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    history = list(zip(session.student_questions, session.patient_responses))
    answer = get_patient_service().answer_question(
        session.case, request.question,
        conversation_history=history,
        revealed_symptoms=session.revealed_symptoms,
    )
    session.student_questions.append(request.question)
    session.patient_responses.append(answer)
    # Track which case symptoms the patient has mentioned so far
    for symptom in session.case.symptoms:
        if symptom not in session.revealed_symptoms and symptom.lower() in answer.lower():
            session.revealed_symptoms.append(symptom)
    graph = get_graph()
    if graph and graph.enabled:
        graph.add_turn(request.session_id, request.question, answer)
    save_session(request.session_id, session)
    log_event(request.session_id, "question.asked", {
        "question": request.question,
        "turn_index": len(session.student_questions) - 1,
    })
    return {
        "question": request.question,
        "answer": answer,
    }


@app.post("/api/investigations")
def order_investigation(request: InvestigationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    results = get_investigation_service().get_results(session.case, request.investigation)
    session.investigations.extend(results)
    graph = get_graph()
    if graph and graph.enabled:
        for result in results:
            graph.add_investigation(request.session_id, result)
    save_session(request.session_id, session)
    for result in results:
        log_event(request.session_id, "investigation.ordered", {
            "investigation": result.investigation,
            "status": result.status,
        })
    displays = [
        student_facing_investigation_result(result, session.case)
        for result in results
    ]
    if len(displays) == 1:
        return displays[0]
    return {
        "investigations": displays,
    }


@app.post("/api/interpretations")
def submit_interpretation(request: InterpretationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    session.interpretations.append(request.interpretation)
    save_session(request.session_id, session)
    log_event(request.session_id, "interpretation.submitted", {"interpretation": request.interpretation})
    return {
        "interpretations": session.interpretations,
    }


@app.post("/api/diagnosis")
def submit_diagnosis(request: DiagnosisRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    session.diagnosis_guess = request.diagnosis
    save_session(request.session_id, session)
    log_event(request.session_id, "diagnosis.submitted", {"guess": request.diagnosis})
    return {
        "diagnosis_guess": request.diagnosis,
    }


@app.post("/api/doctor-consult")
def call_doctor(request: DoctorConsultRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    graph = get_graph()
    graph_context = graph.get_context(request.session_id) if graph and graph.enabled else ""
    response = get_doctor_consult_service().consult(
        session.case,
        session.student_questions,
        session.patient_responses,
        session.investigations,
        request.question,
        session_context={"clinical_picture": graph_context} if graph_context else None,
    )
    session.doctor_consults.append(response)
    save_session(request.session_id, session)
    log_event(request.session_id, "consult.called", {"question": request.question})
    return response.model_dump()


@app.post("/api/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    questions = list(session.student_questions)
    responses = list(session.patient_responses)
    if session.diagnosis_guess:
        questions.append(f"Diagnosis guess: {session.diagnosis_guess}")
        responses.append("I'm not sure.")

    graph = get_graph()
    graph_entity_names = graph.get_entity_names(request.session_id) if graph and graph.enabled else []

    evaluation = get_evaluator().evaluate(
        session.case,
        questions,
        responses,
        investigations_ordered=session.investigations,
        student_interpretations=session.interpretations,
        session_history=[],
        graph_entity_names=graph_entity_names or None,
        exams_performed=[e.system for e in session.physical_exams],
    )
    save_evaluation(request.session_id, evaluation, elapsed_seconds=request.elapsed_seconds)
    log_event(request.session_id, "session.evaluated", {"overall_score": getattr(evaluation, "overall_score", None), "elapsed_seconds": request.elapsed_seconds})
    return evaluation.model_dump()


@app.post("/api/physical-exam")
def physical_exam(request: PhysicalExamRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    already_examined = [e.system for e in session.physical_exams]
    new_systems = [s for s in request.systems if s not in already_examined]
    if not new_systems:
        return {"results": [e.model_dump() for e in session.physical_exams]}

    results = get_physical_exam_agent().examine(session.case, new_systems, already_examined)
    session.physical_exams.extend(results)

    graph = get_graph()
    if graph and graph.enabled:
        for finding in results:
            try:
                graph.add_examination(request.session_id, finding)
            except Exception:
                pass

    save_session(request.session_id, session)
    log_event(request.session_id, "exam.performed", {"systems": new_systems})
    return {"results": [e.model_dump() for e in results]}


def get_session(session_id: str) -> WebSession:
    session = SESSIONS.get(session_id)
    if session is None:
        session = load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        SESSIONS[session_id] = session
    return session


def get_simulator() -> ClinicalCaseSimulator:
    global SIMULATOR
    if SIMULATOR is None:
        SIMULATOR = ClinicalCaseSimulator()
    return SIMULATOR


def get_patient_service() -> PatientResponseService:
    global PATIENT_SERVICE
    if PATIENT_SERVICE is None:
        PATIENT_SERVICE = PatientResponseService()
    return PATIENT_SERVICE


def get_investigation_service() -> InvestigationService:
    global INVESTIGATION_SERVICE
    if INVESTIGATION_SERVICE is None:
        INVESTIGATION_SERVICE = InvestigationService()
    return INVESTIGATION_SERVICE


def get_doctor_consult_service() -> DoctorConsultService:
    global DOCTOR_CONSULT_SERVICE
    if DOCTOR_CONSULT_SERVICE is None:
        DOCTOR_CONSULT_SERVICE = DoctorConsultService()
    return DOCTOR_CONSULT_SERVICE


def get_physical_exam_agent() -> PhysicalExamAgent:
    global PHYSICAL_EXAM_AGENT
    if PHYSICAL_EXAM_AGENT is None:
        PHYSICAL_EXAM_AGENT = PhysicalExamAgent()
    return PHYSICAL_EXAM_AGENT


def get_evaluator() -> ClinicalEvaluator:
    global EVALUATOR
    if EVALUATOR is None:
        EVALUATOR = ClinicalEvaluator()
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


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>CasesBasis — Clinical Simulator</title>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Instrument+Serif&display=swap" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    :root{
      --ink-950:#0a0f17;--ink-900:#111824;--ink-800:#1a2332;--ink-700:#273347;
      --ink-600:#3b4a63;--ink-500:#5b6b85;--ink-400:#8696b0;--ink-300:#b3becf;
      --ink-200:#d6dde7;--ink-100:#e9edf2;--ink-50:#f4f6f9;
      --teal-700:#2c8a8a;--teal-600:#358f8f;--teal-500:#46a6a7;--teal-50:#eef8f8;
      --critical:#c43a2c;--critical-bg:#fdecea;
      --caution:#b76e00;--caution-bg:#fdf3e2;
      --normal:#1f7a55;--normal-bg:#e6f4ec;
    }
    .mono{font-family:'JetBrains Mono',ui-monospace,monospace}
    .serif{font-family:'Instrument Serif',Georgia,serif}
    /* ── Topbar ── */
    .cb-topbar{background:linear-gradient(180deg,var(--ink-950) 0%,var(--ink-900) 100%);border-bottom:1px solid #000}
    .cb-topbar-row{display:flex;align-items:center;gap:14px;padding:8px 20px}
    .cb-brand{display:flex;align-items:center;gap:10px;flex-shrink:0}
    .cb-brand-mark{width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--teal-500),var(--teal-700));box-shadow:inset 0 0 0 1px rgba(255,255,255,0.18);position:relative;flex-shrink:0}
    .cb-brand-mark::before,.cb-brand-mark::after{content:'';position:absolute;background:white;border-radius:1px;top:50%;left:50%;transform:translate(-50%,-50%)}
    .cb-brand-mark::before{width:12px;height:2px}
    .cb-brand-mark::after{width:2px;height:12px}
    .cb-brand-name{font-size:14px;font-weight:600;color:white;letter-spacing:-0.01em;line-height:1.2}
    .cb-brand-teal{color:var(--teal-500);font-weight:500}
    .cb-brand-sub{font-size:10px;color:var(--ink-400)}
    .cb-sep{width:1px;height:22px;background:var(--ink-700);flex-shrink:0}
    /* ── Inline case form ── */
    .cb-case-form{flex:1;display:flex;align-items:center;gap:8px;background:var(--ink-800);border:1px solid var(--ink-700);border-radius:8px;padding:4px 4px 4px 12px;min-width:0}
    .cb-form-label{font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-400);font-weight:600;flex-shrink:0;white-space:nowrap}
    .cb-form-sep{width:1px;height:18px;background:var(--ink-700);margin:0 2px;flex-shrink:0}
    .cb-diag-select{flex:1;background:transparent;border:0;outline:0;color:white;font:inherit;font-size:13px;min-width:0;cursor:pointer}
    .cb-diag-select option,.cb-diag-select optgroup{background:var(--ink-900);color:white}
    /* ── Difficulty segment ── */
    .cb-diff-seg{display:flex;gap:1px;background:var(--ink-700);border-radius:5px;padding:2px;flex-shrink:0}
    .cb-diff-btn{background:transparent;color:var(--ink-300);border:0;font:inherit;font-size:11px;font-weight:500;padding:4px 10px;border-radius:4px;cursor:pointer;transition:background .15s,color .15s}
    .cb-diff-btn.cb-diff-active{background:var(--ink-900);color:white;box-shadow:inset 0 0 0 1px var(--ink-600)}
    .cb-diff-btn[data-val="medium"].cb-diff-active{color:#f0b865}
    .cb-diff-btn[data-val="hard"].cb-diff-active{color:#f08080}
    /* ── Generate button ── */
    .cb-gen-btn{background:var(--teal-600);color:white;border:0;font:inherit;font-size:12px;font-weight:600;padding:7px 14px;border-radius:5px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.12);flex-shrink:0;white-space:nowrap;transition:background .15s}
    .cb-gen-btn:hover{background:var(--teal-500)}
    /* ── Topbar right ── */
    .cb-topbar-right{display:flex;align-items:center;gap:12px;flex-shrink:0}
    .cb-session-live{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--ink-300);white-space:nowrap}
    .cb-live-dot{display:inline-block;width:6px;height:6px;background:#2ec27e;border-radius:50%;box-shadow:0 0 0 3px rgba(46,194,126,0.18);animation:cbPulse 2s ease-in-out infinite}
    @keyframes cbPulse{0%,100%{opacity:1}50%{opacity:0.5}}
    .cb-avatar{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#4a5a78,#2d3a52);color:white;font-size:11px;font-weight:600;display:grid;place-items:center;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.1)}
    .cb-topbar-btn{background:var(--ink-700);color:var(--ink-200);border:1px solid var(--ink-600);font:inherit;font-size:11px;font-weight:500;padding:5px 10px;border-radius:5px;cursor:pointer;transition:background .15s,color .15s;white-space:nowrap;text-decoration:none;display:inline-flex;align-items:center;gap:5px;line-height:1}
    .cb-topbar-btn:hover{background:var(--ink-600);color:white}
    /* ── Login modal ── */
    .cb-login-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:300;backdrop-filter:blur(4px);animation:kgFadeIn .18s ease;align-items:center;justify-content:center}
    .cb-login-backdrop.on{display:flex}
    .cb-login-card{background:#fff;border-radius:14px;padding:32px 28px;width:100%;max-width:380px;box-shadow:0 24px 64px rgba(0,0,0,0.22);animation:loginSlideUp .22s cubic-bezier(.34,1.56,.64,1)}
    @keyframes loginSlideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
    .cb-login-logo{display:flex;align-items:center;gap:9px;margin-bottom:20px}
    .cb-login-mark{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,var(--teal-500),var(--teal-700));flex-shrink:0;position:relative}
    .cb-login-mark::before,.cb-login-mark::after{content:'';position:absolute;background:#fff;border-radius:1px;top:50%;left:50%;transform:translate(-50%,-50%)}
    .cb-login-mark::before{width:13px;height:2px}.cb-login-mark::after{width:2px;height:13px}
    .cb-login-title{font-size:18px;font-weight:700;color:var(--ink-900);letter-spacing:-0.02em;line-height:1.2}
    .cb-login-sub{font-size:12px;color:var(--ink-500);margin-bottom:20px;line-height:1.5}
    .cb-login-label{font-size:11px;font-weight:600;color:var(--ink-600);text-transform:uppercase;letter-spacing:0.08em;display:block;margin-bottom:5px}
    .cb-login-input{width:100%;font:inherit;font-size:13px;padding:9px 11px;border:1.5px solid var(--ink-200);border-radius:7px;outline:0;color:var(--ink-900);transition:border-color .15s,box-shadow .15s;box-sizing:border-box}
    .cb-login-input:focus{border-color:var(--teal-500);box-shadow:0 0 0 3px rgba(70,166,167,0.14)}
    .cb-login-btn{width:100%;margin-top:12px;padding:10px;font:inherit;font-size:13px;font-weight:600;background:var(--teal-600);color:#fff;border:0;border-radius:7px;cursor:pointer;transition:background .15s}
    .cb-login-btn:hover{background:var(--teal-500)}
    .cb-login-btn:disabled{opacity:0.55;cursor:not-allowed}
    .cb-login-msg{margin-top:12px;font-size:12px;color:var(--ink-600);line-height:1.5;display:none}
    .cb-login-msg.err{color:var(--critical)}
    .cb-login-devlink{margin-top:10px;padding:10px 12px;background:var(--ink-50);border:1px dashed var(--ink-200);border-radius:7px;font-size:11.5px;line-height:1.55;display:none}
    .cb-login-devlink strong{color:var(--ink-900);font-weight:600;display:block;margin-bottom:4px}
    .cb-login-devlink a{color:var(--teal-600);word-break:break-all;font-family:'JetBrains Mono',monospace;font-size:10.5px}
    .cb-login-close{position:absolute;top:12px;right:14px;background:none;border:0;cursor:pointer;color:var(--ink-400);font-size:18px;line-height:1;padding:4px}
    .cb-login-divider{display:flex;align-items:center;gap:10px;margin:14px 0;color:var(--ink-300);font-size:11px}
    .cb-login-divider::before,.cb-login-divider::after{content:'';flex:1;height:1px;background:var(--ink-100)}
    .cb-login-test-btn{width:100%;padding:10px;font:inherit;font-size:13px;font-weight:600;background:#0d1f14;color:#4ade80;border:1.5px solid #22c55e;border-radius:7px;cursor:pointer;position:relative;overflow:hidden;transition:background .15s}
    .cb-login-test-btn:hover{background:#0f2b1a}
    .cb-login-test-btn::before{content:'';position:absolute;inset:-1px;border-radius:8px;background:transparent;border:1.5px solid #22c55e;box-shadow:0 0 10px 2px rgba(34,197,94,0.35),inset 0 0 10px rgba(34,197,94,0.08);animation:testGlow 2s ease-in-out infinite}
    @keyframes testGlow{0%,100%{box-shadow:0 0 8px 2px rgba(34,197,94,0.3),inset 0 0 8px rgba(34,197,94,0.07)}50%{box-shadow:0 0 18px 5px rgba(34,197,94,0.55),inset 0 0 14px rgba(34,197,94,0.12)}}
    .cb-login-test-label{font-size:10px;color:#86efac;font-weight:500;margin-top:6px;text-align:center;display:block;letter-spacing:0.02em}
    /* ── Stepper ── */
    .cb-stepper{background:var(--ink-900);border-top:1px solid var(--ink-800);padding:0 20px;display:flex;align-items:stretch;overflow-x:auto}
    .cb-stepper::-webkit-scrollbar{height:0}
    .cb-step{display:flex;align-items:center;gap:8px;padding:9px 14px 9px 0;color:var(--ink-500);font-size:11.5px;position:relative;flex-shrink:0;transition:color .2s}
    .cb-step+.cb-step{padding-left:18px}
    .cb-step+.cb-step::before{content:'';position:absolute;left:4px;top:50%;width:7px;height:7px;border-top:1px solid var(--ink-700);border-right:1px solid var(--ink-700);transform:translateY(-50%) rotate(45deg)}
    .cb-step-num{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:10px;font-weight:600;background:var(--ink-800);color:var(--ink-400);box-shadow:inset 0 0 0 1px var(--ink-700);flex-shrink:0;transition:background .2s,color .2s}
    .cb-step-label{font-weight:500}
    .cb-step.done{color:var(--ink-300)}
    .cb-step.done .cb-step-num{background:var(--teal-700);color:white;box-shadow:none}
    .cb-step.active{color:white}
    .cb-step.active .cb-step-num{background:var(--teal-500);color:var(--ink-950);box-shadow:0 0 0 3px rgba(70,166,167,0.25)}
    /* ── Patient strip ── */
    .cb-pt-strip{background:white;border-bottom:1px solid var(--ink-100);padding:18px 24px;display:grid;grid-template-columns:1.4fr 1fr 1.6fr;gap:28px;align-items:stretch}
    .cb-ps-label{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-400);font-weight:600;margin-bottom:8px}
    .cb-cc{font-family:'Instrument Serif',Georgia,serif;font-size:26px;line-height:1.15;letter-spacing:-0.01em;color:var(--ink-900);margin-bottom:10px}
    .cb-cc-quote{color:var(--ink-400)}
    .cb-chips{display:flex;flex-wrap:wrap;gap:6px}
    .cb-chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;padding:3px 9px;border-radius:999px;background:var(--ink-50);color:var(--ink-700);border:1px solid var(--ink-100)}
    .cb-chip .cb-k{color:var(--ink-400);font-weight:500}
    .cb-chip-teal{background:var(--teal-50);color:var(--teal-700);border-color:oklch(0.88 0.06 195)}
    .cb-vitals{display:flex;flex-wrap:nowrap;gap:8px}
    .cb-vital{background:white;border:1px solid var(--ink-100);border-radius:8px;padding:10px 12px;position:relative;overflow:hidden;flex:0 0 auto;min-width:72px}
    .cb-vital::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--ink-200)}
    .cb-vital.critical{background:var(--critical-bg);border-color:#f4c8c2}
    .cb-vital.critical::before{background:var(--critical)}
    .cb-vital.caution{background:var(--caution-bg);border-color:#f1d9a8}
    .cb-vital.caution::before{background:var(--caution)}
    .cb-vital.normal::before{background:var(--normal)}
    .cb-v-row{display:flex;align-items:flex-start}
    .cb-v-label{font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-400);font-weight:600}
    .cb-v-val{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:20px;color:var(--ink-900);line-height:1.1;margin-top:4px;display:flex;align-items:baseline;gap:4px}
    .cb-v-unit{font-size:11px;color:var(--ink-500);font-weight:500}
    .cb-v-flag{margin-left:auto;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px}
    .cb-v-flag.high,.cb-v-flag.low{background:var(--critical);color:white}
    .cb-v-flag.warn{background:var(--caution);color:white}
    .cb-v-meta{font-size:10px;color:var(--ink-500);margin-top:4px}
    .cb-persona-card{background:var(--ink-50);border:1px solid var(--ink-100);border-radius:8px;padding:12px 14px;display:flex;gap:12px;align-items:flex-start;height:100%;box-sizing:border-box}
    .cb-persona-avatar{width:44px;height:44px;border-radius:50%;flex-shrink:0;background:linear-gradient(140deg,#c8b8a0,#8a6e52);box-shadow:inset 0 -10px 16px rgba(0,0,0,0.15),inset 0 0 0 1px rgba(255,255,255,0.2);display:grid;place-items:center;color:white;font-weight:600;font-size:13px}
    .cb-persona-name{font-style:normal;font-weight:600;color:var(--ink-900);margin-bottom:2px;font-size:12px}
    .cb-persona-text{font-size:12px;color:var(--ink-700);line-height:1.5;font-style:italic}
    /* ── Main 3-col grid ── */
    .cb-main{display:grid;grid-template-columns:minmax(320px,1.05fr) minmax(340px,1.25fr) minmax(280px,0.95fr);gap:14px;padding:14px 24px 24px;align-items:start}
    .cb-col{display:flex;flex-direction:column;gap:14px;min-width:0}
    /* ── Panel base ── */
    .cb-panel{background:white;border:1px solid var(--ink-100);border-radius:10px;box-shadow:var(--shadow-sm);display:flex;flex-direction:column;overflow:hidden}
    .cb-panel-head{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--ink-100);flex-shrink:0}
    .cb-panel-head-toggle{cursor:pointer;user-select:none;transition:background .12s}
    .cb-panel-head-toggle:hover{background:var(--ink-50)}
    .cb-panel-body-wrap{display:block}
    .cb-panel.collapsed .cb-panel-body-wrap{display:none}
    .cb-panel.collapsed .cb-panel-head{border-bottom:0}
    .cb-panel-chev{color:var(--ink-400);flex-shrink:0;transition:transform .2s}
    .cb-panel.collapsed .cb-panel-chev{transform:rotate(-90deg)}
    .cb-panel-num{font-size:11px;color:var(--ink-400);font-weight:500;margin-right:2px;font-family:'JetBrains Mono',monospace}
    .cb-panel-title{font-size:11px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;color:var(--ink-700)}
    .cb-panel-actions{margin-left:auto;display:flex;gap:4px}
    .cb-panel-body{padding:14px;display:flex;flex-direction:column;gap:10px}
    .cb-iconbtn{width:24px;height:24px;border-radius:5px;background:transparent;border:0;color:var(--ink-500);display:grid;place-items:center;cursor:pointer}
    .cb-iconbtn:hover{background:var(--ink-50);color:var(--ink-700)}
    /* ── Investigation result cards ── */
    .cb-result{border:1px solid var(--ink-100);border-radius:8px;overflow:hidden;background:white}
    .cb-result-head{display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--ink-50);cursor:pointer;user-select:none;transition:background .12s}
    .cb-result-head:hover{background:var(--ink-100)}
    .cb-result.open .cb-result-head{border-bottom:1px solid var(--ink-100)}
    .cb-result-body{display:none}
    .cb-result.open .cb-result-body{display:block}
    .cb-result-chev{margin-left:4px;flex-shrink:0;color:var(--ink-400);transition:transform .2s}
    .cb-result.open .cb-result-chev{transform:rotate(180deg)}
    .cb-result-icon{min-width:28px;height:22px;padding:0 4px;border-radius:5px;background:white;border:1px solid var(--ink-200);display:grid;place-items:center;color:var(--ink-700);font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace;flex-shrink:0}
    .cb-result-title{font-size:12px;font-weight:600;color:var(--ink-900)}
    .cb-result-sub{font-size:10.5px;color:var(--ink-500)}
    .cb-result-stamp{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--ink-500);white-space:nowrap}
    /* ── CXR PACS viewer ── */
    .cb-cxr-viewer{background:#050708;color:#cdd5e0}
    .cb-cxr-toolbar{display:flex;align-items:center;gap:8px;padding:6px 10px;background:linear-gradient(180deg,#0f1722,#0a1018);border-bottom:1px solid #000;font-size:10px;color:#6b7a92}
    .cb-cxr-tbtn{background:transparent;color:#95a3bb;border:0;font:inherit;font-size:10px;padding:3px 7px;border-radius:3px;cursor:pointer;text-transform:uppercase;letter-spacing:0.06em}
    .cb-cxr-tbtn:hover{background:rgba(255,255,255,0.06);color:white}
    .cb-cxr-tbtn.on{color:var(--teal-500)}
    .cb-cxr-tool-sep{width:1px;height:12px;background:#1c2738;flex-shrink:0}
    .cb-cxr-stage{position:relative;background:#050708;overflow:hidden}
    .cb-cxr-stage img{width:100%;display:block}
    .cb-cxr-overlay{position:absolute;inset:0;pointer-events:none;font-family:'JetBrains Mono',monospace;font-size:9.5px;color:rgba(180,200,220,0.7);text-shadow:0 0 4px rgba(0,0,0,0.9);line-height:1.55}
    .cb-cxr-tl{position:absolute;top:8px;left:10px}
    .cb-cxr-tr{position:absolute;top:8px;right:10px;text-align:right}
    .cb-cxr-br{position:absolute;bottom:8px;right:10px;text-align:right}
    .cb-cxr-window-info{padding:5px 10px;background:#0a1018;border-top:1px solid #000;display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:9px;color:#6b7a92}
    .cb-cxr-side{position:absolute;top:50%;transform:translateY(-50%);font-size:22px;font-weight:700;color:rgba(220,230,240,0.85);font-family:'JetBrains Mono',monospace;pointer-events:none;text-shadow:0 0 8px rgba(0,0,0,0.9)}
    .cb-cxr-side.l{left:12px}
    .cb-cxr-side.r{right:12px}
    .cb-cxr-finding-banner{padding:8px 12px;background:var(--ink-50);border-top:1px solid var(--ink-100);display:flex;gap:10px;align-items:flex-start}
    .cb-cxr-finding-tag{font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;color:var(--caution);background:var(--caution-bg);padding:2px 6px;border-radius:3px;flex-shrink:0}
    .cb-cxr-finding-text{font-size:11.5px;color:var(--ink-700);line-height:1.5}
    /* ── ECG viewer ── */
    .cb-ecg-meta{display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:#0f0606;color:#d4a4a4;font-family:'JetBrains Mono',monospace;font-size:10px;border-top:1px solid #2a1010}
    .cb-ecg-meta strong{color:#ffd9a4;font-weight:600}
    /* ── Lab table ── */
    .cb-lab-table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11.5px}
    .cb-lab-table th{background:var(--ink-50);color:var(--ink-500);text-align:left;font-weight:600;font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;padding:6px 10px;border-bottom:1px solid var(--ink-100)}
    .cb-lab-table td{padding:6px 10px;border-bottom:1px solid var(--ink-100);color:var(--ink-800)}
    .cb-lab-table tr:last-child td{border-bottom:0}
    .cb-lab-name{font-family:'Inter',sans-serif;font-weight:500;color:var(--ink-900)}
    .cb-lab-val{font-weight:600}
    .cb-lab-val.abn{color:var(--critical)}
    .cb-lab-flag{text-align:right;font-weight:700;color:var(--critical)}
    .cb-lab-ref{color:var(--ink-400)}
    .cb-lab-pattern{padding:8px 12px;border-top:1px solid var(--ink-100);background:var(--ink-50);font-size:11.5px;color:var(--ink-700);line-height:1.5}
    /* ── Quick-order chips ── */
    .qchip{font:inherit;font-size:11px;font-weight:500;background:white;border:1px solid var(--ink-200);color:var(--ink-700);padding:4px 10px;border-radius:5px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;transition:border-color .15s,color .15s,background .15s}
    .qchip:hover{border-color:var(--teal-500);color:var(--teal-700)}
    .qchip.ordered{background:var(--teal-50);border-color:oklch(0.85 0.06 195);color:var(--teal-700)}
    .qchip.ordered::before{content:'✓ ';color:var(--teal-600);font-weight:700}
    /* ── Knowledge graph dark stage ── */
    .cb-kg-stage{position:relative;overflow:hidden;height:320px;background:radial-gradient(ellipse at 20% 20%,rgba(70,166,167,0.06),transparent 60%),radial-gradient(ellipse at 80% 80%,rgba(215,122,58,0.05),transparent 60%),var(--ink-900)}
    /* KG expanded overlay */
    #graphPanel.kg-expanded{position:fixed;inset:40px;z-index:200;border-radius:12px;box-shadow:0 24px 80px rgba(0,0,0,0.45);display:flex;flex-direction:column;transition:all .28s cubic-bezier(.4,0,.2,1)}
    #graphPanel.kg-expanded .cb-panel-body-wrap{flex:1;min-height:0;display:flex;flex-direction:column}
    #graphPanel.kg-expanded .cb-kg-stage{flex:1;min-height:0;height:auto}
    .cb-kg-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.45);z-index:199;backdrop-filter:blur(3px);animation:kgFadeIn .2s ease}
    .cb-kg-backdrop.on{display:block}
    @keyframes kgFadeIn{from{opacity:0}to{opacity:1}}
    .cb-kg-stat{position:absolute;top:8px;left:8px;background:rgba(20,30,46,0.75);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:5px 9px;backdrop-filter:blur(6px);font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(255,255,255,0.85);display:flex;gap:12px;pointer-events:none;z-index:2}
    .cb-kg-stat strong{color:var(--teal-500);font-weight:600}
    /* ── Physical exam panel ── */
    .cb-exam-layout{display:grid;grid-template-columns:200px 1fr;gap:0;padding:0}
    .cb-exam-body-col{background:#fff;border-right:1px solid var(--ink-100);padding:6px 4px;display:flex;flex-direction:column;align-items:center}
    .cb-exam-body{position:relative;flex-shrink:0}
    .cb-exam-body img{width:190px;height:310px;object-fit:cover;object-position:center 8%;display:block;flex-shrink:0}
    .cb-exam-pin{position:absolute;transform:translate(-50%,-50%);width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:7.5px;font-weight:700;letter-spacing:-0.3px;color:white;border:1.5px solid rgba(255,255,255,0.9);box-shadow:0 1px 6px rgba(0,0,0,0.3);transition:transform .15s,box-shadow .15s;z-index:2}
    .cb-exam-pin:hover{transform:translate(-50%,-50%) scale(1.2);box-shadow:0 3px 10px rgba(0,0,0,0.35)}
    .cb-exam-pin.normal{background:var(--teal-600)}
    .cb-exam-pin.abnormal{background:var(--critical)}
    .cb-exam-pin-pulse{animation:examPinPop .4s cubic-bezier(.34,1.56,.64,1)}
    @keyframes examPinPop{0%{transform:translate(-50%,-50%) scale(0)}100%{transform:translate(-50%,-50%) scale(1)}}
    .cb-exam-checklist-col{padding:8px 10px;display:flex;flex-direction:column;gap:6px}
    .cb-exam-systems{display:flex;flex-direction:column;gap:0}
    .cb-exam-sys-row{display:flex;align-items:center;gap:6px;padding:5px 5px;border-radius:4px;cursor:pointer;transition:background .12s;border-left:2.5px solid transparent}
    .cb-exam-sys-row:hover{background:var(--ink-50)}
    .cb-exam-sys-row.selected{background:var(--ink-50)}
    .cb-exam-sys-row.done{background:transparent;cursor:default;pointer-events:none;opacity:0.68}
    .cb-exam-sys-row.done .cb-exam-sys-label{color:var(--ink-500)}
    .cb-exam-sys-cb{width:14px;height:14px;border-radius:3px;border:1.5px solid var(--ink-300);flex-shrink:0;display:grid;place-items:center;transition:background .12s,border-color .12s}
    .cb-exam-sys-cb.checked{background:var(--teal-600);border-color:var(--teal-600)}
    .cb-exam-sys-cb.checked::after{content:'';display:block;width:3px;height:6px;border:1.5px solid white;border-top:0;border-left:0;transform:rotate(45deg) translate(-0.5px,-1px)}
    .cb-exam-sys-accent{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .cb-exam-sys-label{font-size:11.5px;font-weight:500;color:var(--ink-800);line-height:1.2}
    .cb-exam-sys-sublabel{font-size:10px;color:var(--ink-400);line-height:1.3;margin-top:1px}
    .cb-exam-results{display:flex;flex-direction:column;gap:8px;padding:0 14px 14px}
    .cb-perf-status{font-size:10px;text-transform:uppercase;letter-spacing:0.08em;padding:4px 8px;border-radius:4px;font-weight:600;display:inline-flex;align-items:center;gap:6px;align-self:flex-start;line-height:1.3}
    .cb-perf-status.in-progress{color:var(--caution);background:var(--caution-bg)}
    .cb-perf-status.complete{color:var(--normal);background:var(--normal-bg)}
    .cb-kg-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.025) 1px,transparent 1px);background-size:24px 24px;pointer-events:none}
    .cb-kg-legend{position:absolute;bottom:8px;left:8px;background:rgba(20,30,46,0.75);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:5px 8px;backdrop-filter:blur(6px);display:flex;gap:10px;pointer-events:none}
    .cb-kg-legend-item{display:flex;align-items:center;gap:5px;font-size:10px;color:rgba(255,255,255,0.8)}
    .cb-kg-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
    /* ── Doctor card ── */
    .cb-doctor-head{display:flex;align-items:center;gap:10px;padding:8px 12px;background:linear-gradient(180deg,#1f3357,#182843);color:white;flex-shrink:0}
    .cb-doctor-avatar{width:28px;height:28px;border-radius:50%;background:linear-gradient(140deg,#d8d0c2,#8a7f6e);box-shadow:inset 0 0 0 2px white,0 0 0 1px rgba(0,0,0,0.2);flex-shrink:0}
    .cb-doctor-name{font-size:12px;font-weight:600;line-height:1.2}
    .cb-doctor-role{font-size:10px;color:rgba(255,255,255,0.6)}
    .cb-doctor-status{margin-left:auto;font-size:10px;color:#6ad6a8;display:inline-flex;align-items:center;gap:5px;white-space:nowrap;flex-shrink:0}
    .cb-doctor-status::before{content:'';width:6px;height:6px;background:#2ec27e;border-radius:50%;box-shadow:0 0 0 3px rgba(46,194,126,0.2);animation:cbPulse 2s ease-in-out infinite}
    .cb-doctor-response{padding:10px 12px;background:var(--ink-50);border-bottom:1px solid var(--ink-100);font-size:12px;line-height:1.55;color:var(--ink-800);font-style:italic;position:relative}
    .cb-doctor-response::before{content:'"';position:absolute;top:-2px;left:4px;font-family:'Instrument Serif',serif;font-size:28px;color:var(--ink-300);line-height:1}
    .cb-doctor-response p{margin:0;padding-left:12px}
    /* ── Field inputs ── */
    .cb-field{width:100%;font:inherit;font-size:12.5px;line-height:1.55;padding:8px 10px;border:1px solid var(--ink-200);border-radius:7px;background:white;color:var(--ink-900);outline:0;box-sizing:border-box}
    .cb-field:focus{border-color:var(--teal-500);box-shadow:0 0 0 3px rgba(70,166,167,0.12)}
    textarea.cb-field{resize:vertical}
    /* ── Performance score donut ── */
    .cb-score-ring{width:90px;height:90px;flex-shrink:0;position:relative}
    .cb-score-ring svg{width:100%;height:100%;transform:rotate(-90deg)}
    .cb-score-num{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center}
    .cb-score-val{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:var(--ink-900);line-height:1}
    .cb-score-of{font-size:9px;color:var(--ink-400);text-transform:uppercase;letter-spacing:0.08em;margin-top:2px}
    /* ── Performance metrics ── */
    .cb-perf-metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .cb-perf-metric{border:1px solid var(--ink-100);border-radius:7px;padding:8px 10px}
    .cb-perf-label{font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-400);font-weight:600}
    .cb-perf-bar{height:3px;background:var(--ink-100);border-radius:2px;overflow:hidden;margin-top:5px}
    .cb-perf-bar-fill{height:100%;border-radius:2px;transition:width .5s ease}
    /* ── Feedback items ── */
    .cb-feedback-item{display:flex;gap:8px;align-items:flex-start;font-size:12px;line-height:1.5;color:var(--ink-700)}
    .cb-feedback-bullet{width:16px;height:16px;border-radius:50%;flex-shrink:0;display:grid;place-items:center;font-size:10px;font-weight:700;margin-top:1px}
    .cb-section-div{font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-500);font-weight:600;display:flex;align-items:center;gap:6px;margin:2px 0}
    .cb-section-div::after{content:'';flex:1;height:1px;background:var(--ink-100)}
    /* ── Status bar ── */
    .cb-statusbar{background:var(--ink-900);color:var(--ink-500);font-family:'JetBrains Mono',monospace;font-size:10.5px;padding:5px 24px;display:flex;gap:20px;align-items:center;border-top:1px solid var(--ink-800);flex-shrink:0;position:sticky;bottom:0;z-index:40}
    .cb-statusbar strong{color:var(--ink-200);font-weight:500}
    .cb-statusbar .cb-sb-dot{width:5px;height:5px;border-radius:50%;background:#2ec27e;box-shadow:0 0 0 2px rgba(46,194,126,0.18);display:inline-block;margin-right:5px;animation:cbPulse 2s ease-in-out infinite}
    /* ── Chat message system ── */
    .cb-chat-body{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth;background:var(--ink-50);min-height:400px;max-height:520px}
    .cb-chat-body::-webkit-scrollbar{width:4px}
    .cb-chat-body::-webkit-scrollbar-thumb{background:var(--ink-200);border-radius:2px}
    .cb-chat-divider{text-align:center;font-size:10px;color:var(--ink-400);text-transform:uppercase;letter-spacing:0.1em;display:flex;align-items:center;gap:10px;margin:4px 0}
    .cb-chat-divider::before,.cb-chat-divider::after{content:'';flex:1;height:1px;background:var(--ink-100)}
    .cb-msg{display:flex;gap:10px;max-width:88%}
    .cb-msg.s{margin-left:auto;flex-direction:row-reverse}
    .cb-msg-av{width:26px;height:26px;border-radius:50%;flex-shrink:0;display:grid;place-items:center;font-size:10px;font-weight:600;color:white}
    .cb-msg.p .cb-msg-av{background:linear-gradient(140deg,#c8b8a0,#8a6e52)}
    .cb-msg.s .cb-msg-av{background:var(--ink-700)}
    .cb-msg-bubble{background:white;border:1px solid var(--ink-100);border-radius:10px;padding:8px 12px;font-size:13px;line-height:1.5;color:var(--ink-800)}
    .cb-msg.s .cb-msg-bubble{background:var(--teal-700);color:white;border-color:var(--teal-700)}
    .cb-msg.p .cb-msg-bubble{border-top-left-radius:3px}
    .cb-msg.s .cb-msg-bubble{border-top-right-radius:3px}
    .cb-msg-meta{font-size:10px;color:var(--ink-400);margin-top:4px;display:flex;gap:8px;align-items:center}
    .cb-msg.s .cb-msg-meta{justify-content:flex-end}
    .cb-msg-tag{font-size:9.5px;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;padding:1px 5px;border-radius:3px}
    .cb-tag-r{background:var(--normal-bg);color:var(--normal)}
    .cb-tag-e{color:var(--ink-500);font-style:italic;font-weight:400;text-transform:none;letter-spacing:0;font-size:10px}
    .cb-chat-typing{display:flex;gap:8px;align-items:center;color:var(--ink-400);font-size:11px;font-style:italic;padding-left:36px}
    .cb-typing-dots{display:inline-flex;gap:3px}
    .cb-typing-dots span{width:4px;height:4px;border-radius:50%;background:var(--ink-400);animation:tdots 1.2s infinite}
    .cb-typing-dots span:nth-child(2){animation-delay:.2s}
    .cb-typing-dots span:nth-child(3){animation-delay:.4s}
    @keyframes tdots{0%,60%,100%{opacity:0.3}30%{opacity:1}}
    .cb-chat-foot{border-top:1px solid var(--ink-100);padding:10px 12px}
    .cb-chat-ctrl{display:flex;gap:6px;margin-bottom:8px;font-size:11px;color:var(--ink-500);align-items:center}
    .cb-toggle{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
    .cb-toggle-track{width:26px;height:14px;background:var(--teal-600);border-radius:999px;position:relative;flex-shrink:0;transition:background .2s}
    .cb-toggle-track::after{content:'';position:absolute;right:2px;top:2px;width:10px;height:10px;background:white;border-radius:50%;box-shadow:0 1px 2px rgba(0,0,0,.2);transition:right .15s,left .15s}
    .cb-toggle-track.off{background:var(--ink-200)}
    .cb-toggle-track.off::after{left:2px;right:auto}
    .cb-chat-input{display:flex;align-items:center;gap:6px;background:white;border:1px solid var(--ink-200);border-radius:8px;padding:4px 4px 4px 12px}
    .cb-chat-input input{flex:1;background:transparent;border:0;outline:0;font:inherit;font-size:13px;padding:7px 0;color:var(--ink-900);min-width:0}
    .cb-chat-input input::placeholder{color:var(--ink-400)}
    .cb-chat-mic{width:30px;height:30px;border-radius:6px;background:var(--ink-50);border:1px solid var(--ink-200);display:grid;place-items:center;cursor:pointer;color:var(--ink-600);flex-shrink:0}
    .cb-chat-send{width:30px;height:30px;border-radius:6px;background:var(--teal-600);border:0;color:white;display:grid;place-items:center;cursor:pointer;flex-shrink:0}
    *{font-family:'Inter',sans-serif}
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
    .fade-in{animation:fadeUp .3s ease-out}
    .spinning{animation:spin .7s linear infinite}
    .dot{animation:bounce 1s ease-in-out infinite}
    .dot:nth-child(2){animation-delay:.15s}
    .dot:nth-child(3){animation-delay:.3s}
    .bubble-you{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-radius:18px 18px 4px 18px;margin-left:auto}
    .bubble-patient{background:#fff;border:1px solid #e2e8f0;border-radius:18px 18px 18px 4px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
    .chat-area{height:420px;overflow-y:auto;scroll-behavior:smooth}
    .chat-area::-webkit-scrollbar{width:4px}
    .chat-area::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:2px}
    input,textarea,select{transition:border-color .15s,box-shadow .15s}
    input:focus,textarea:focus,select:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
    .ecg-wrap{background:#fff9f9;border-radius:8px;border:1px solid #fecaca}
    .ecg-wrap svg{display:block;width:100%;height:auto}
    .score-ring{width:110px;height:110px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.2rem;font-weight:800;flex-shrink:0}
    .chip{cursor:pointer;transition:background .15s,color .15s}
    .urgency-routine{background:#d1fae5;color:#065f46}
    .urgency-urgent{background:#fef3c7;color:#92400e}
    .urgency-critical{background:#fee2e2;color:#991b1b}
    [hidden]{display:none!important}
    #graphSvg{background:transparent}
    .graph-tooltip{position:absolute;background:rgba(15,23,42,.88);color:#e2e8f0;font-size:11px;padding:6px 10px;border-radius:8px;pointer-events:none;max-width:260px;line-height:1.4;backdrop-filter:blur(4px);z-index:100}
    .node-circle{cursor:grab;transition:opacity .2s}
    .node-circle:active{cursor:grabbing}
    .edge-label{paint-order:stroke;stroke:rgba(255,255,255,0.85);stroke-width:3px;stroke-linejoin:round;transition:opacity .2s}
    @keyframes micPulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.5)}70%{box-shadow:0 0 0 7px rgba(239,68,68,0)}}
    .mic-recording{animation:micPulse 1.1s ease-out infinite;background:#fee2e2!important;border-color:#ef4444!important;color:#dc2626!important}
    .replay-btn{opacity:0;transition:opacity .15s}
    .bubble-patient:hover .replay-btn{opacity:1}
    /* ── Pause Modal ── */
    .cb-pause-backdrop{display:none;position:fixed;inset:0;z-index:400;backdrop-filter:blur(6px) saturate(0.7) brightness(0.55);background:radial-gradient(ellipse at 50% 50%, rgba(10,15,23,0.45), rgba(10,15,23,0.82) 70%);align-items:center;justify-content:center;padding:24px}
    .cb-pause-backdrop.on{display:flex;animation:kgFadeIn .2s ease}
    .cb-pause-modal{width:560px;max-width:100%;background:radial-gradient(ellipse at 50% -20%, oklch(0.22 0.04 230) 0%, transparent 60%),linear-gradient(180deg,#121a28 0%,#0d1420 100%);color:#e9edf2;border:1px solid rgba(255,255,255,0.08);border-radius:18px;box-shadow:0 30px 80px rgba(8,12,20,0.55),0 4px 12px rgba(8,12,20,0.35),inset 0 1px 0 rgba(255,255,255,0.06);position:relative;overflow:hidden;animation:cbPauseRise .36s cubic-bezier(.2,.7,.1,1) both}
    @keyframes cbPauseRise{from{opacity:0;transform:translateY(14px) scale(0.985)}to{opacity:1;transform:translateY(0) scale(1)}}
    .cb-pause-close{position:absolute;top:14px;right:14px;width:30px;height:30px;border-radius:8px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);color:rgba(255,255,255,0.6);display:grid;place-items:center;cursor:pointer;z-index:2;transition:background .15s,color .15s}
    .cb-pause-close:hover{background:rgba(255,255,255,0.10);color:white}
    .cb-pause-ecg{position:relative;height:104px;background:radial-gradient(ellipse at 50% 100%,rgba(40,180,120,0.10),transparent 60%),linear-gradient(180deg,#0a1119,#0c1320);border-bottom:1px solid rgba(255,255,255,0.05);overflow:hidden}
    .cb-pause-ecg-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(50,180,120,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(50,180,120,0.06) 1px,transparent 1px),linear-gradient(rgba(50,180,120,0.10) 1px,transparent 1px),linear-gradient(90deg,rgba(50,180,120,0.10) 1px,transparent 1px);background-size:10px 10px,10px 10px,50px 50px,50px 50px;mask-image:linear-gradient(90deg,transparent,black 8%,black 92%,transparent);-webkit-mask-image:linear-gradient(90deg,transparent,black 8%,black 92%,transparent)}
    .cb-pause-ecg-meta{position:absolute;left:16px;top:12px;display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;color:rgba(120,255,180,0.78);font-size:10.5px;letter-spacing:0.08em;text-transform:uppercase}
    .cb-pause-ecg-lead{color:rgba(255,255,255,0.45);border:1px solid rgba(120,255,180,0.25);padding:1px 6px;border-radius:3px;font-size:9.5px}
    .cb-pause-ecg-bpm{position:absolute;right:16px;top:12px;text-align:right;font-family:'JetBrains Mono',monospace;color:rgba(120,255,180,0.85)}
    .cb-pause-ecg-bpm-val{font-size:22px;font-weight:700;line-height:1;text-shadow:0 0 8px rgba(120,255,180,0.5)}
    .cb-pause-ecg-bpm-lbl{font-size:9px;letter-spacing:0.12em;color:rgba(255,255,255,0.45);text-transform:uppercase;margin-top:2px;font-weight:600}
    .cb-pause-ecg-line{position:absolute;inset:0;overflow:hidden;mask-image:linear-gradient(90deg,transparent 0%,black 7%,black 93%,transparent 100%);-webkit-mask-image:linear-gradient(90deg,transparent 0%,black 7%,black 93%,transparent 100%)}
    .cb-pause-ecg-track{position:absolute;left:0;top:0;height:100%;width:200%;animation:cbEcgScroll 2.2s linear infinite}
    @keyframes cbEcgScroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
    .cb-pause-ecg-track svg{width:100%;height:100%;display:block}
    .cb-pause-trace{fill:none;stroke:oklch(0.78 0.22 145);stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px oklch(0.88 0.22 145)) drop-shadow(0 0 10px rgba(80,220,140,0.45))}
    .cb-pause-ecg-cursor{position:absolute;right:7%;top:50%;width:10px;height:10px;border-radius:50%;background:oklch(0.88 0.22 145);box-shadow:0 0 0 2px rgba(120,255,180,0.25),0 0 14px 4px rgba(120,255,180,0.55),0 0 30px 10px rgba(80,220,140,0.18);transform:translate(50%,-50%);animation:cbHeart 0.92s ease-in-out infinite}
    @keyframes cbHeart{0%{transform:translate(50%,-50%) scale(1);opacity:.85}10%{transform:translate(50%,-50%) scale(1.45);opacity:1}25%{transform:translate(50%,-50%) scale(1);opacity:.85}40%{transform:translate(50%,-50%) scale(1.25);opacity:.95}55%,100%{transform:translate(50%,-50%) scale(1);opacity:.85}}
    .cb-pause-body{padding:20px 28px 22px;text-align:center;position:relative}
    .cb-pause-eyebrow{display:inline-flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,255,255,0.55);font-weight:600;padding:5px 12px;border-radius:99px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08)}
    .cb-pause-bars{display:inline-flex;gap:2.5px;align-items:center}
    .cb-pause-bars i{display:block;width:3px;height:10px;background:oklch(0.78 0.22 145);border-radius:1px;box-shadow:0 0 6px rgba(120,255,180,0.5)}
    .cb-pause-title{font-family:'Instrument Serif',Georgia,serif;font-size:30px;font-weight:400;letter-spacing:-0.01em;line-height:1.1;color:white;margin:12px 0 4px}
    .cb-pause-title em{font-style:italic;color:var(--teal-500)}
    .cb-pause-subtitle{font-size:13.5px;color:rgba(255,255,255,0.6);line-height:1.5;max-width:380px;margin:0 auto}
    .cb-pause-ctx{margin:14px auto 0;display:flex;align-items:stretch;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:10px;overflow:hidden}
    .cb-pause-seg{padding:8px 14px;border-right:1px solid rgba(255,255,255,0.06);text-align:left;flex:1;min-width:0}
    .cb-pause-seg:first-child{flex:1.25}
    .cb-pause-seg:last-child{border-right:0}
    .cb-pause-seg-k{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(255,255,255,0.4);font-weight:600;display:block;margin-bottom:1px}
    .cb-pause-seg-v{font-family:'JetBrains Mono',monospace;font-size:12px;color:rgba(255,255,255,0.92);font-weight:500;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .cb-pause-timer-wrap{margin-top:18px;display:flex;flex-direction:column;align-items:center;gap:6px}
    .cb-pause-timer-label{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;color:var(--caution);background:rgba(212,154,71,0.12);border:1px solid rgba(212,154,71,0.30);padding:4px 11px 4px 9px;border-radius:99px;font-weight:600}
    .cb-pause-timer{font-family:'JetBrains Mono',monospace;font-size:52px;font-weight:600;letter-spacing:-0.02em;color:white;line-height:1;display:flex;align-items:baseline;gap:2px;text-shadow:0 2px 30px rgba(255,255,255,0.06)}
    .cb-pause-sep{color:rgba(255,255,255,0.35);margin:0 4px;animation:cbSepBlink 1.6s steps(2,end) infinite}
    @keyframes cbSepBlink{0%,50%{opacity:1}50.01%,100%{opacity:.35}}
    .cb-pause-timer-foot{font-size:11.5px;color:rgba(255,255,255,0.45);font-family:'JetBrains Mono',monospace;letter-spacing:0.04em}
    .cb-pause-timer-foot strong{color:rgba(255,255,255,0.75);font-weight:600}
    .cb-pause-saved{margin:14px auto 0;display:inline-flex;align-items:center;gap:8px;font-size:12px;color:rgba(120,255,180,0.88);background:rgba(40,180,120,0.08);border:1px solid rgba(40,180,120,0.22);padding:6px 12px;border-radius:99px}
    .cb-pause-actions{margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .cb-pause-btn{font:inherit;font-size:14px;font-weight:600;padding:11px 16px;border-radius:10px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;gap:9px;transition:transform .08s ease,background .15s ease;letter-spacing:-.005em}
    .cb-pause-btn:active{transform:translateY(1px)}
    .cb-pause-primary{background:var(--teal-600);color:white;border:1px solid var(--teal-500);box-shadow:inset 0 1px 0 rgba(255,255,255,0.18),0 6px 14px rgba(46,141,143,0.35)}
    .cb-pause-primary:hover{background:var(--teal-500)}
    .cb-pause-ghost{background:rgba(255,255,255,0.04);color:rgba(255,255,255,0.88);border:1px solid rgba(255,255,255,0.12)}
    .cb-pause-ghost:hover{background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.22)}
    .cb-pause-foot{margin-top:14px;padding-top:12px;border-top:1px dashed rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10.5px;color:rgba(255,255,255,0.4)}
    .cb-pause-kbd b{font-family:'JetBrains Mono',monospace;font-weight:600;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);color:rgba(255,255,255,0.7);font-size:10px}
  </style>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800">

<!-- HEADER -->
<header class="cb-topbar sticky top-0 z-50">
  <div class="cb-topbar-row">
    <div class="cb-brand">
      <div class="cb-brand-mark" aria-hidden="true"></div>
      <div>
        <div class="cb-brand-name">Cases<span class="cb-brand-teal">Basis</span></div>
        <div class="cb-brand-sub">Clinical Simulation</div>
      </div>
    </div>
    <div class="cb-sep"></div>
    <div class="cb-case-form">
      <span class="cb-form-label">Diagnosis</span>
      <select id="diagnosis" class="cb-diag-select">
        <optgroup label="Cardiac">
          <option value="anterior STEMI">Anterior STEMI</option>
          <option value="NSTEMI">NSTEMI</option>
          <option value="atrial fibrillation" selected>Atrial Fibrillation</option>
          <option value="acute decompensated heart failure">Acute Heart Failure</option>
          <option value="hypertensive emergency">Hypertensive Emergency</option>
        </optgroup>
        <optgroup label="Respiratory">
          <option value="acute severe asthma">Acute Severe Asthma</option>
          <option value="COPD exacerbation">COPD Exacerbation</option>
          <option value="pulmonary embolism">Pulmonary Embolism</option>
          <option value="pneumothorax">Pneumothorax</option>
          <option value="community-acquired pneumonia">Pneumonia</option>
        </optgroup>
        <optgroup label="Metabolic">
          <option value="diabetic ketoacidosis">Diabetic Ketoacidosis (DKA)</option>
          <option value="hypoglycaemia">Hypoglycaemia</option>
          <option value="hyperkalaemia">Hyperkalaemia</option>
        </optgroup>
        <optgroup label="Neurological">
          <option value="ischaemic stroke">Ischaemic Stroke</option>
          <option value="subarachnoid haemorrhage">Subarachnoid Haemorrhage</option>
          <option value="bacterial meningitis">Bacterial Meningitis</option>
        </optgroup>
        <optgroup label="Emergency">
          <option value="septic shock">Septic Shock</option>
          <option value="anaphylaxis">Anaphylaxis</option>
        </optgroup>
        <optgroup label="GI / Surgical">
          <option value="upper GI bleed">Upper GI Bleed</option>
          <option value="acute pancreatitis">Acute Pancreatitis</option>
          <option value="appendicitis">Appendicitis</option>
        </optgroup>
      </select>
      <span class="cb-form-sep"></span>
      <span class="cb-form-label">Difficulty</span>
      <div class="cb-diff-seg">
        <button onclick="setDifficulty('easy')" class="cb-diff-btn" data-val="easy">Easy</button>
        <button onclick="setDifficulty('medium')" class="cb-diff-btn cb-diff-active" data-val="medium">Medium</button>
        <button onclick="setDifficulty('hard')" class="cb-diff-btn" data-val="hard">Hard</button>
      </div>
      <input type="hidden" id="difficulty" value="medium"/>
      <button id="generateBtn" onclick="generateCase()" class="cb-gen-btn">
        <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M8 1v14M1 8h14"/></svg>
        Generate
      </button>
    </div>
    <div class="cb-topbar-right">
      <div id="sessionBadge" hidden class="cb-session-live">
        <span class="cb-live-dot"></span>Session active
      </div>
      <button id="pauseBtn" hidden onclick="pauseSession()" class="cb-topbar-btn" title="Pause and save this session">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="4" height="12" rx="1"/><rect x="9" y="2" width="4" height="12" rx="1"/></svg>Pause
      </button>
      <a href="/dashboard" id="dashLink" hidden class="cb-topbar-btn">Dashboard</a>
      <div id="userAvatar" hidden class="cb-avatar" title="" style="cursor:default;background:linear-gradient(135deg,var(--teal-700),var(--teal-500))"></div>
      <button id="logoutBtn" hidden onclick="doLogout()" class="cb-topbar-btn">Log out</button>
      <a href="/" id="loginLink" class="cb-topbar-btn">Log in</a>
    </div>
  </div>
  <!-- Workflow stepper -->
  <nav class="cb-stepper" aria-label="Workflow steps">
    <div class="cb-step active" id="step1"><span class="cb-step-num" id="sn1">1</span><span class="cb-step-label">Generate Case</span></div>
    <div class="cb-step" id="step2"><span class="cb-step-num" id="sn2">2</span><span class="cb-step-label">Patient Overview</span></div>
    <div class="cb-step" id="step3"><span class="cb-step-num" id="sn3">3</span><span class="cb-step-label">Patient Interview</span></div>
    <div class="cb-step" id="step4"><span class="cb-step-num" id="sn4">4</span><span class="cb-step-label">Order Investigations</span></div>
    <div class="cb-step" id="step5"><span class="cb-step-num" id="sn5">5</span><span class="cb-step-label">Interpret Results</span></div>
    <div class="cb-step" id="step6"><span class="cb-step-num" id="sn6">6</span><span class="cb-step-label">Submit Diagnosis</span></div>
    <div class="cb-step" id="step7"><span class="cb-step-num" id="sn7">7</span><span class="cb-step-label">Evaluate</span></div>
  </nav>
</header>

<main class="max-w-screen-2xl mx-auto">

  <!-- PATIENT OVERVIEW STRIP (shown after case generated) -->
  <section id="vitalsStrip" hidden aria-label="Patient overview" class="cb-pt-strip fade-in">
    <!-- Chief complaint + chips -->
    <div>
      <div class="cb-ps-label">Chief Complaint</div>
      <div class="cb-cc"><span class="cb-cc-quote">&ldquo;</span><span id="pvCC"></span><span class="cb-cc-quote">&rdquo;</span></div>
      <div class="cb-chips">
        <span class="cb-chip" id="pvAgeChip"><span class="cb-k">Age</span> <span id="pvAge"></span></span>
        <span class="cb-chip" id="pvGenderChip"><span class="cb-k">Sex</span> <span id="pvGender"></span></span>
        <span class="cb-chip" id="pvRaceChip" hidden><span class="cb-k">Race</span> <span id="pvRace"></span></span>
        <span class="cb-chip cb-chip-teal" id="pvOccChip" hidden><span id="pvOccupation"></span></span>
      </div>
    </div>
    <!-- Vital tiles -->
    <div>
      <div class="cb-ps-label">Vitals on Presentation</div>
      <div class="cb-vitals">
        <div class="cb-vital" id="vtHR">
          <div class="cb-v-row"><span class="cb-v-label">HR</span><span class="cb-v-flag" id="vfHR" hidden></span></div>
          <div class="cb-v-val"><span id="pvHR"></span><span class="cb-v-unit">bpm</span></div>
        </div>
        <div class="cb-vital" id="vtBP">
          <div class="cb-v-row"><span class="cb-v-label">BP</span><span class="cb-v-flag" id="vfBP" hidden></span></div>
          <div class="cb-v-val"><span id="pvBP"></span><span class="cb-v-unit">mmHg</span></div>
        </div>
        <div class="cb-vital" id="vtSPO2">
          <div class="cb-v-row"><span class="cb-v-label">SpO₂</span><span class="cb-v-flag" id="vfSPO2" hidden></span></div>
          <div class="cb-v-val"><span id="pvSPO2"></span><span class="cb-v-unit">%</span></div>
        </div>
        <div class="cb-vital" id="vtRR">
          <div class="cb-v-row"><span class="cb-v-label">RR</span><span class="cb-v-flag" id="vfRR" hidden></span></div>
          <div class="cb-v-val"><span id="pvRR"></span><span class="cb-v-unit">/min</span></div>
        </div>
      </div>
    </div>
    <!-- Persona card -->
    <div>
      <div class="cb-ps-label">Persona</div>
      <div class="cb-persona-card" id="pvPersonaCard">
        <div class="cb-persona-avatar" id="pvPersonaAvatar">?</div>
        <div>
          <div class="cb-persona-name" id="pvPersonaName"></div>
          <div class="cb-persona-text" id="pvPersona"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- MAIN GRID: 3 columns -->
  <div class="cb-main">

    <!-- ── COL 1: Patient chat ── -->
    <div class="cb-col">
      <div class="cb-panel">
        <div class="cb-panel-head">
          <span class="cb-panel-num">03</span>
          <span class="cb-panel-title">Patient Interview</span>
          <span id="chatTurnBadge" hidden class="cb-chip" style="background:var(--teal-50);color:var(--teal-700);border-color:oklch(0.85 0.06 195);font-size:10px;padding:1px 7px"></span>
        </div>
        <div id="chatArea" class="cb-chat-body">
          <div class="cb-chat-divider">Generate a case to begin</div>
        </div>
        <div class="cb-chat-foot">
          <div class="cb-chat-ctrl">
            <label class="cb-toggle" onclick="toggleTts()">
              <span id="ttsTrack" class="cb-toggle-track"></span>
              <span>Read patient responses aloud</span>
            </label>
            <span id="ttsVoiceLabel" style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--ink-400)">EN-US · Auto</span>
          </div>
          <div class="cb-chat-input">
            <input id="question" placeholder="Ask the patient a question…"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQuestion()}"/>
            <button id="micBtn" onclick="toggleMic()" title="Speak your question" class="cb-chat-mic">
              <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
              </svg>
            </button>
            <button id="askBtn" onclick="askQuestion()" class="cb-chat-send" title="Send">
              <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 8l12-6-4 14-3-6-5-2z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>

      <!-- KG backdrop for expanded view -->
      <div class="cb-kg-backdrop" id="kgBackdrop" onclick="collapseKg()"></div>

      <!-- Login modal -->
      <div class="cb-login-backdrop" id="loginBackdrop" onclick="if(event.target===this)hideLoginModal()">
        <div class="cb-login-card" style="position:relative">
          <button class="cb-login-close" onclick="hideLoginModal()" aria-label="Close">&times;</button>
          <div class="cb-login-logo">
            <div class="cb-login-mark"></div>
            <div>
              <div class="cb-login-title">Cases<span style="color:var(--teal-600)">Basis</span></div>
            </div>
          </div>
          <p class="cb-login-sub">Enter your email and we'll send you a magic sign-in link — no password needed.</p>
          <label class="cb-login-label" for="loginEmail">Email address</label>
          <input id="loginEmail" class="cb-login-input" type="email" placeholder="you@example.com"
            onkeydown="if(event.key==='Enter')doRequestLink()"/>
          <button id="loginSubmitBtn" class="cb-login-btn" onclick="doRequestLink()">Send sign-in link</button>
          <div id="loginMsg" class="cb-login-msg"></div>
          <div id="loginDevLink" class="cb-login-devlink">
            <strong>Dev mode — click your magic link:</strong>
            <a id="loginDevAnchor" href="#"></a>
          </div>
          <div class="cb-login-divider">or</div>
          <button class="cb-login-test-btn" onclick="doTestLogin()">
            ⚡ Test Resident
          </button>
          <span class="cb-login-test-label">Instant login · no account needed · dev only</span>
        </div>
      </div>

      <!-- Knowledge Graph -->
      <div id="graphPanel" hidden class="cb-panel fade-in">
        <div class="cb-panel-head cb-panel-head-toggle" onclick="togglePanel('graphPanel')">
          <span class="cb-panel-num">KG</span>
          <span class="cb-panel-title">Knowledge Graph</span>
          <span id="graphStatus" class="text-xs italic" style="color:var(--ink-400);margin-left:6px">Waiting…</span>
          <span id="graphBadge" hidden class="cb-chip" style="font-size:10px;padding:1px 7px;margin-left:6px"></span>
          <div style="margin-left:auto;display:flex;align-items:center;gap:4px">
            <div id="graphControls" hidden style="display:flex;align-items:center;gap:2px">
              <button onclick="event.stopPropagation();graphZoomIn()" class="cb-iconbtn" style="font-size:14px;font-weight:700">+</button>
              <button onclick="event.stopPropagation();graphZoomOut()" class="cb-iconbtn" style="font-size:14px;font-weight:700">−</button>
              <button onclick="event.stopPropagation();graphFit()" class="cb-iconbtn" style="font-size:10px;padding:0 4px">Fit</button>
            </div>
            <button onclick="event.stopPropagation();toggleKgExpand()" id="kgExpandBtn" class="cb-iconbtn" title="Expand">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
                <path d="M1 4.5V1h3.5M7.5 1H11v3.5M11 7.5V11H7.5M4.5 11H1V7.5"/>
              </svg>
            </button>
            <svg class="cb-panel-chev" width="14" height="14" viewBox="0 0 14 14" fill="none" style="margin-left:2px"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          </div>
        </div>
        <div class="cb-panel-body-wrap">
        <div class="cb-kg-stage">
          <div class="cb-kg-grid"></div>
          <div id="kgStatBar" hidden class="cb-kg-stat">
            <span><strong id="kgNodeCount">0</strong> nodes</span>
            <span><strong id="kgEdgeCount">0</strong> edges</span>
            <span id="kgRecentSpan" hidden><strong id="kgRecentCount">+0</strong> last min</span>
          </div>
          <div id="graphEmpty" class="absolute inset-0 flex flex-col items-center justify-center gap-2 pointer-events-none">
            <svg class="w-10 h-10" fill="none" stroke="rgba(255,255,255,0.18)" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"/>
            </svg>
            <p class="text-xs" style="color:rgba(255,255,255,0.3)">Ask questions to build the graph</p>
          </div>
          <svg id="graphSvg" style="position:absolute;inset:0;width:100%;height:100%;cursor:grab"></svg>
          <div class="cb-kg-legend">
            <div class="cb-kg-legend-item"><div class="cb-kg-dot" style="background:#f97316"></div>Symptom</div>
            <div class="cb-kg-legend-item"><div class="cb-kg-dot" style="background:#f59e0b"></div>Risk</div>
            <div class="cb-kg-legend-item"><div class="cb-kg-dot" style="background:#6366f1"></div>Finding</div>
            <div class="cb-kg-legend-item"><div class="cb-kg-dot" style="background:#60a5fa"></div>Medication</div>
            <div class="cb-kg-legend-item"><div class="cb-kg-dot" style="background:#10b981"></div>Diagnosis</div>
          </div>
          <div id="graphTooltip" class="graph-tooltip" style="display:none"></div>
        </div>
        </div>
      </div>
    </div>

    <!-- ── COL 2: Investigations ── -->
    <div class="cb-col">

      <!-- Order form -->
      <div class="cb-panel">
        <div class="cb-panel-head">
          <span class="cb-panel-num">04</span>
          <span class="cb-panel-title">Investigations</span>
          <span id="invCountBadge" hidden class="cb-chip" style="background:var(--ink-50);color:var(--ink-600);font-size:10px;padding:1px 7px;margin-left:auto"></span>
        </div>
        <div style="padding:12px 14px;display:flex;flex-direction:column;gap:10px">
          <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
            <span style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-400);font-weight:600;margin-right:2px;flex-shrink:0">Quick order</span>
            <button class="qchip" data-inv="cxr"      onclick="quickOrder('CXR')">CXR</button>
            <button class="qchip" data-inv="abg"      onclick="quickOrder('ABG')">ABG</button>
            <button class="qchip" data-inv="cbc"      onclick="quickOrder('CBC')">CBC</button>
            <button class="qchip" data-inv="ecg"      onclick="quickOrder('ECG')">ECG</button>
            <button class="qchip" data-inv="troponin" onclick="quickOrder('Troponin')">Troponin</button>
            <button class="qchip" data-inv="ddimer"   onclick="quickOrder('D-dimer')">D-dimer</button>
            <button class="qchip" data-inv="ue"       onclick="quickOrder('U&amp;E')">U&amp;E</button>
            <button class="qchip" data-inv="peakflow" onclick="quickOrder('Peak Flow')">Peak Flow</button>
          </div>
          <div style="display:flex;gap:6px">
            <input id="investigation" value="" placeholder="Order any investigation… e.g. 'serum tryptase'"
              style="flex:1;min-width:0;padding:6px 10px;font:inherit;font-size:12px;border:1px dashed var(--ink-200);border-radius:6px;background:var(--ink-50);color:var(--ink-800);outline:0"
              onkeydown="if(event.key==='Enter')orderInvestigation()"/>
            <button id="invBtn" onclick="orderInvestigation()"
              style="font:inherit;font-size:11px;font-weight:500;padding:6px 10px;border-radius:6px;background:white;border:1px solid var(--ink-200);color:var(--ink-700);cursor:pointer;flex-shrink:0">
              Order
            </button>
          </div>
          <div id="invResults" style="display:flex;flex-direction:column;gap:10px"></div>
        </div>
      </div>

      <!-- Physical Exam panel -->
      <div id="examPanel" class="cb-panel collapsed">
        <div class="cb-panel-head cb-panel-head-toggle" onclick="togglePanel('examPanel')">
          <span class="cb-panel-num">PE</span>
          <span class="cb-panel-title">Physical Examination</span>
          <span id="examDoneBadge" hidden class="cb-chip" style="background:var(--teal-50);color:var(--teal-700);border-color:oklch(0.85 0.06 195);font-size:10px;padding:1px 7px;margin-left:6px"></span>
          <svg class="cb-panel-chev" width="14" height="14" viewBox="0 0 14 14" fill="none" style="margin-left:auto"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        </div>
        <div class="cb-panel-body-wrap">
          <div class="cb-exam-layout">
            <!-- Body image -->
            <div class="cb-exam-body cb-exam-body-col" id="examBodyWrap">
              <img src="/body-image.png" alt="Body diagram" draggable="false"/>
              <!-- Pins injected by JS -->
            </div>
            <!-- System checklist -->
            <div class="cb-exam-checklist-col">
              <div class="cb-exam-systems" id="examSystems">
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'General')">
                  <div class="cb-exam-sys-accent" style="background:#64748b"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">General</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Cardiovascular')">
                  <div class="cb-exam-sys-accent" style="background:#ef4444"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Cardiovascular</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Respiratory')">
                  <div class="cb-exam-sys-accent" style="background:#06b6d4"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Respiratory</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Abdominal')">
                  <div class="cb-exam-sys-accent" style="background:#f97316"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Abdominal</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Neurological')">
                  <div class="cb-exam-sys-accent" style="background:#8b5cf6"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Neurological</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Musculoskeletal')">
                  <div class="cb-exam-sys-accent" style="background:#10b981"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Musculoskeletal</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Dermatological')">
                  <div class="cb-exam-sys-accent" style="background:#ec4899"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Dermatological</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'ENT/Head &amp; Neck')">
                  <div class="cb-exam-sys-accent" style="background:#f59e0b"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">ENT / Head &amp; Neck</div>
                </div>
                <div class="cb-exam-sys-row" onclick="toggleExamSys(this,'Peripheral Vascular')">
                  <div class="cb-exam-sys-accent" style="background:#6366f1"></div>
                  <div class="cb-exam-sys-cb"></div>
                  <div class="cb-exam-sys-label">Peripheral Vascular</div>
                </div>
              </div>
              <button id="examBtn" onclick="performExam()"
                style="font:inherit;font-size:11px;font-weight:600;padding:7px 12px;border-radius:6px;background:var(--ink-900);color:white;border:0;cursor:pointer;transition:background .15s"
                onmouseover="this.style.background='var(--ink-700)'" onmouseout="this.style.background='var(--ink-900)'">
                Examine selected
              </button>
            </div>
          </div>
          <div id="examResults" class="cb-exam-results"></div>
        </div>
      </div>

      <!-- ECG panel -->
      <div id="ecgPanel" hidden class="cb-panel fade-in">
        <div class="cb-panel-head cb-panel-head-toggle" onclick="togglePanel('ecgPanel')">
          <span class="cb-panel-num">ECG</span>
          <span class="cb-panel-title">Electrocardiogram · 12-lead</span>
          <svg class="cb-panel-chev" width="14" height="14" viewBox="0 0 14 14" fill="none" style="margin-left:auto"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        </div>
        <div class="cb-panel-body-wrap">
          <div id="ecgContent" class="ecg-wrap"></div>
          <div id="ecgMeta" hidden class="cb-ecg-meta">
            <span id="ecgMetaLeft">—</span>
            <span id="ecgMetaRight" style="text-align:right;max-width:60%"></span>
          </div>
        </div>
      </div>

      <!-- CXR panel -->
      <div id="cxrPanel" hidden class="cb-panel fade-in">
        <div class="cb-panel-head cb-panel-head-toggle" onclick="togglePanel('cxrPanel')">
          <span class="cb-panel-num">CXR</span>
          <span class="cb-panel-title">Chest Radiograph · PA</span>
          <svg class="cb-panel-chev" width="14" height="14" viewBox="0 0 14 14" fill="none" style="margin-left:auto"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        </div>
        <div class="cb-panel-body-wrap">
          <div class="cb-cxr-viewer">
            <div class="cb-cxr-toolbar">
              <div style="display:flex;gap:2px">
                <button class="cb-cxr-tbtn on" data-mode="wl"   onclick="setCxrMode('wl')">W/L</button>
                <button class="cb-cxr-tbtn"    data-mode="zoom" onclick="setCxrMode('zoom')">Zoom</button>
                <button class="cb-cxr-tbtn"    data-mode="pan"  onclick="setCxrMode('pan')">Pan</button>
              </div>
              <span class="cb-cxr-tool-sep"></span>
              <div style="display:flex;gap:2px"><button class="cb-cxr-tbtn" data-mode="invert" onclick="toggleCxrInvert()">Invert</button></div>
              <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b7a92">PA · SYNTH</span>
            </div>
            <div class="cb-cxr-stage">
              <img id="cxrImage" src="" alt="Chest X-Ray"/>
              <div class="cb-cxr-side l">L</div>
              <div class="cb-cxr-side r">R</div>
              <div class="cb-cxr-overlay">
                <div class="cb-cxr-tl" id="cxrOverlayPatient"></div>
                <div class="cb-cxr-tr">PA UPRIGHT<br/>STAT · PORTABLE<br/>RoentGen-v2</div>
                <div class="cb-cxr-br">SYNTH · AI<br/><span id="cxrOverlayStudy">----</span></div>
              </div>
            </div>
            <div class="cb-cxr-window-info">
              <span>W: 1500 &nbsp;&nbsp; L: -600</span>
              <span>ZOOM 100% · PA</span>
              <span id="cxrFindingLabel">RoentGen-v2 · synth</span>
            </div>
            <div id="cxrFindingBanner" hidden class="cb-cxr-finding-banner">
              <span class="cb-cxr-finding-tag">AI · Provisional</span>
              <div id="cxrFindingText" class="cb-cxr-finding-text"></div>
            </div>
          </div>
        </div>
      </div>

    </div>

    <!-- ── COL 3: Side rail ── -->
    <div class="cb-col">

      <!-- 05–06: Reasoning & Diagnosis (merged panel) -->
      <div class="cb-panel">
        <div class="cb-panel-head">
          <span class="cb-panel-num">05 – 06</span>
          <span class="cb-panel-title">Reasoning &amp; Diagnosis</span>
        </div>
        <div class="cb-panel-body" style="gap:14px">

          <!-- Clinical Interpretation -->
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-500);font-weight:600">Clinical Interpretation</span>
              <span id="interpCharCount" style="font-size:10.5px;color:var(--ink-400)">0 / 1000</span>
            </div>
            <textarea id="interpretation" rows="4" placeholder="Your interpretation of ordered tests…"
              class="cb-field" oninput="updateInterpCount(this)" style="resize:vertical;width:100%"></textarea>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-top:6px;gap:8px">
              <ul id="interpList" style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:3px;flex:1;font-size:11.5px;color:var(--ink-700)"></ul>
              <button id="interpBtn" onclick="submitInterpretation()"
                style="background:var(--ink-900);color:white;border:0;font:inherit;font-size:11.5px;font-weight:600;padding:7px 12px;border-radius:6px;cursor:pointer;flex-shrink:0"
                onmouseover="this.style.background='var(--ink-800)'" onmouseout="this.style.background='var(--ink-900)'">
                Save
              </button>
            </div>
          </div>

          <div class="cb-section-div">Working Diagnosis</div>

          <!-- Working Diagnosis -->
          <div>
            <div style="display:flex;gap:6px">
              <input id="diagnosisGuess" placeholder="Your working diagnosis…"
                class="cb-field" style="flex:1;min-width:0"
                onkeydown="if(event.key==='Enter')submitDiagnosis()"/>
              <button id="diagnosisBtn" onclick="submitDiagnosis()"
                style="background:var(--teal-600);color:white;border:0;font:inherit;font-size:11.5px;font-weight:600;padding:8px 12px;border-radius:6px;cursor:pointer;flex-shrink:0;white-space:nowrap;display:inline-flex;align-items:center;gap:5px"
                onmouseover="this.style.background='var(--teal-500)'" onmouseout="this.style.background='var(--teal-600)'">
                Submit
                <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M3 8l3 3 7-7"/></svg>
              </button>
            </div>
            <div id="diagnosisResult" hidden style="margin-top:8px;background:var(--teal-50);border:1px solid oklch(0.85 0.06 195);border-radius:7px;padding:8px 10px;font-size:12px;color:var(--teal-700)" class="fade-in"></div>
          </div>

          <div class="cb-section-div">Senior Doctor</div>

          <!-- Doctor Consult (inline card) -->
          <div style="border:1px solid var(--ink-100);border-radius:8px;overflow:hidden">
            <div class="cb-doctor-head">
              <div class="cb-doctor-avatar"></div>
              <div>
                <div class="cb-doctor-name">Dr. Hamza</div>
                <div class="cb-doctor-role">Senior ER Physician</div>
              </div>
              <div class="cb-doctor-status">On call</div>
            </div>
            <div id="doctorConsults"></div>
            <div style="display:flex;gap:6px;padding:8px 10px;align-items:flex-end">
              <textarea id="doctorQuestion" rows="2" placeholder="Ask Dr. Hamza for guidance…"
                class="cb-field" style="flex:1;resize:none"></textarea>
              <button id="doctorBtn" onclick="callDoctor()"
                style="background:var(--ink-900);color:white;border:0;font:inherit;font-size:11.5px;font-weight:600;padding:8px 12px;border-radius:6px;cursor:pointer;flex-shrink:0;white-space:nowrap"
                onmouseover="this.style.background='var(--ink-800)'" onmouseout="this.style.background='var(--ink-900)'">
                Ask
              </button>
            </div>
          </div>

          <div class="cb-section-div" style="margin-top:2px">Final Evaluation</div>

          <!-- Evaluate button -->
          <button id="evaluateBtn" onclick="evaluateCase()"
            style="width:100%;background:var(--ink-900);color:white;border:0;font:inherit;font-size:12px;font-weight:600;padding:9px;border-radius:7px;cursor:pointer;letter-spacing:0.02em;display:inline-flex;align-items:center;justify-content:center;gap:6px"
            onmouseover="this.style.background='var(--ink-800)'" onmouseout="this.style.background='var(--ink-900)'">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
              <path d="M3 8l3 3 7-7"/>
            </svg>
            Evaluate Final Performance
          </button>

        </div>
      </div>

      <!-- Evaluation Results -->
      <div id="evalResults" hidden class="cb-panel fade-in">
        <div class="cb-panel-head">
          <span class="cb-panel-num">★</span>
          <span class="cb-panel-title">Performance Report</span>
        </div>
        <div class="cb-panel-body">
          <div id="perfStatusBadge" class="cb-perf-status in-progress" hidden>
            <svg width="7" height="7" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg>
            In Progress · partial report
          </div>
          <!-- Score donut + brief feedback -->
          <div style="display:flex;align-items:flex-start;gap:12px">
            <div id="scoreCircle" class="cb-score-ring">
              <svg viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="36" stroke="var(--ink-100)" stroke-width="8" fill="none"/>
                <circle id="scoreArc" cx="50" cy="50" r="36" stroke="var(--teal-500)" stroke-width="8" fill="none"
                  stroke-dasharray="226.2" stroke-dashoffset="226.2" stroke-linecap="round" style="transition:stroke-dashoffset .6s ease,stroke .4s"/>
              </svg>
              <div class="cb-score-num">
                <div class="cb-score-val" id="scoreVal">—</div>
                <div class="cb-score-of">/ 100</div>
              </div>
            </div>
            <div style="flex:1;min-width:0;padding-top:4px">
              <div class="cb-ps-label">Overall Score</div>
              <p id="scoreFeedback" style="font-size:12px;line-height:1.55;color:var(--ink-700);margin:0"></p>
            </div>
          </div>
          <!-- Metrics grid -->
          <div class="cb-perf-metrics">
            <div class="cb-perf-metric">
              <div class="cb-perf-label">Diagnosis</div>
              <div id="evalDx" style="font-size:12px;margin-top:3px;color:var(--ink-900)">—</div>
            </div>
            <div class="cb-perf-metric">
              <div class="cb-perf-label">Symptoms</div>
              <div id="evalSymptoms" style="font-size:12px;margin-top:3px">—</div>
            </div>
            <div class="cb-perf-metric">
              <div class="cb-perf-label">Investigations</div>
              <div id="evalInvestigations" style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600;color:var(--ink-900);margin-top:2px;display:flex;align-items:baseline;gap:3px">—</div>
              <div class="cb-perf-bar"><div id="evalInvBar" class="cb-perf-bar-fill" style="width:0%;background:var(--teal-500)"></div></div>
            </div>
            <div class="cb-perf-metric">
              <div class="cb-perf-label">Time on Task</div>
              <div id="evalTimeOnTask" style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600;color:var(--ink-900);margin-top:2px">—</div>
              <div class="cb-perf-bar"><div id="evalTimeBar" class="cb-perf-bar-fill" style="width:0%;background:var(--ink-400)"></div></div>
            </div>
            <div class="cb-perf-metric" style="grid-column:1/-1">
              <div class="cb-perf-label">Question Quality</div>
              <div id="evalQuality" style="font-size:12px;margin-top:3px;color:var(--ink-700)">—</div>
              <div class="cb-perf-bar"><div id="evalQualityBar" class="cb-perf-bar-fill" style="width:0%;background:var(--teal-500)"></div></div>
            </div>
          </div>
          <!-- Reasoning -->
          <div>
            <div class="cb-section-div">Reasoning</div>
            <div id="evalReasoning" style="display:flex;flex-direction:column;gap:6px"></div>
          </div>
          <!-- Learning -->
          <div>
            <div class="cb-section-div">Learning</div>
            <div id="evalLearning" style="display:flex;flex-direction:column;gap:6px"></div>
          </div>
        </div>
      </div>

    </div>
  </div>

</main>

<!-- STATUS BAR -->
<footer class="cb-statusbar" aria-label="Status bar">
  <span id="sbCase"><strong>CASE</strong> —</span>
  <span id="sbSession"><strong>SESSION</strong> inactive</span>
  <span id="sbDiag"><strong>Dx</strong> —</span>
  <span id="sbAutosave" hidden><span class="cb-sb-dot"></span><span id="sbSaveTime">—</span></span>
  <span id="sbTimer" hidden><strong>TIME</strong> <span id="sbTimerVal">00:00</span></span>
  <span style="margin-left:auto">CasesBasis · clinical simulation</span>
</footer>

<!-- PAUSE MODAL -->
<div id="pauseBackdrop" class="cb-pause-backdrop" role="dialog" aria-modal="true" aria-labelledby="pauseModalTitle">
  <div class="cb-pause-modal">
    <button class="cb-pause-close" onclick="hidePauseModal()" aria-label="Resume case">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
    </button>
    <!-- ECG heartbeat strip -->
    <div class="cb-pause-ecg" aria-hidden="true">
      <div class="cb-pause-ecg-grid"></div>
      <div class="cb-pause-ecg-meta">
        <span class="cb-pause-ecg-lead">LEAD II</span>
        <span>SINUS · NORMAL</span>
      </div>
      <div class="cb-pause-ecg-bpm">
        <div class="cb-pause-ecg-bpm-val">72</div>
        <div class="cb-pause-ecg-bpm-lbl">BPM</div>
      </div>
      <div class="cb-pause-ecg-line">
        <div class="cb-pause-ecg-track">
          <svg viewBox="0 0 800 104" preserveAspectRatio="none">
            <path class="cb-pause-trace" d="M 0 52 L 30 52 Q 45 52 52 46 Q 60 40 68 46 Q 76 52 84 52 L 110 52 L 118 58 L 124 10 L 132 94 L 140 46 L 150 52 L 180 52 Q 200 52 215 42 Q 232 32 248 52 L 300 52 L 400 52 M 400 52 L 430 52 Q 445 52 452 46 Q 460 40 468 46 Q 476 52 484 52 L 510 52 L 518 58 L 524 10 L 532 94 L 540 46 L 550 52 L 580 52 Q 600 52 615 42 Q 632 32 648 52 L 700 52 L 800 52"/>
          </svg>
        </div>
      </div>
      <div class="cb-pause-ecg-cursor"></div>
    </div>
    <!-- Body -->
    <div class="cb-pause-body">
      <div class="cb-pause-eyebrow">
        <span class="cb-pause-bars"><i></i><i style="height:6px"></i><i style="height:12px"></i><i style="height:6px"></i><i></i></span>
        Case paused
      </div>
      <h2 id="pauseModalTitle" class="cb-pause-title">Take a breath, <em id="pauseDrName">Doctor</em>.</h2>
      <p class="cb-pause-subtitle">Your case is on hold. The clock has stopped — pick up exactly where you left off whenever you're ready.</p>
      <div class="cb-pause-ctx">
        <div class="cb-pause-seg">
          <span class="cb-pause-seg-k">Case</span>
          <span class="cb-pause-seg-v" id="pauseCtxCase">—</span>
        </div>
        <div class="cb-pause-seg">
          <span class="cb-pause-seg-k">Phase</span>
          <span class="cb-pause-seg-v" id="pauseCtxPhase">—</span>
        </div>
        <div class="cb-pause-seg">
          <span class="cb-pause-seg-k">Coverage</span>
          <span class="cb-pause-seg-v" id="pauseCtxCoverage">— / 7</span>
        </div>
      </div>
      <div class="cb-pause-timer-wrap">
        <div class="cb-pause-timer-label">
          <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="2" width="3" height="8" rx="0.6"/><rect x="7" y="2" width="3" height="8" rx="0.6"/></svg>
          Timer paused
        </div>
        <div class="cb-pause-timer" aria-live="off">
          <span id="pauseHH">00</span><span class="cb-pause-sep">:</span><span id="pauseMM">00</span><span class="cb-pause-sep">:</span><span id="pauseSS">00</span>
        </div>
        <div class="cb-pause-timer-foot">Paused at <strong id="pauseAtTime">—</strong></div>
      </div>
      <div class="cb-pause-saved">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3 3 7-7"/></svg>
        All progress and notes saved
      </div>
      <div class="cb-pause-actions">
        <button class="cb-pause-btn cb-pause-ghost" onclick="location.href='/dashboard'">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 7.5L8 2.5l6 5"/><path d="M3.5 7v6.5h9V7"/></svg>
          Back to dashboard
        </button>
        <button class="cb-pause-btn cb-pause-primary" onclick="resumeFromPause()">
          <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M4 3l9 5-9 5z"/></svg>
          Resume case
        </button>
      </div>
      <div class="cb-pause-foot">
        <span>Auto-resume disabled while paused</span>
        <span class="cb-pause-kbd">Resume <b>↵</b> · Dashboard <b>esc</b></span>
      </div>
    </div>
  </div>
</div>

<!-- TOAST -->
<div id="toast" hidden class="fixed bottom-5 right-5 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-2xl fade-in"></div>

<script>
  let sessionId = null;
  let _currentResident = null;
  let _currentStep = 1;
  const _STEP_LABELS = ['','Generate Case','Patient Overview','Patient Interview','Order Investigations','Interpret Results','Submit Diagnosis','Evaluate'];

  /* ── Auth ── */
  async function checkAuth() {
    try {
      const r = await fetch('/api/auth/me');
      if (!r.ok) return;
      _currentResident = await r.json();
      const el = id => document.getElementById(id);
      el('dashLink').hidden = false;
      const av = el('userAvatar');
      av.hidden = false;
      av.textContent = _nameInitials(_currentResident.display_name);
      av.title = _currentResident.display_name;
      el('logoutBtn').hidden = false;
      el('loginLink').hidden = true;
    } catch(e) {}
  }
  function _nameInitials(name) {
    return (name||'').split(' ').filter(Boolean).map(w=>w[0]).join('').toUpperCase().slice(0,2)||'?';
  }

  function showPauseModal(frozenSecs) {
    const first = (_currentResident && _currentResident.display_name)
      ? _currentResident.display_name.trim().split(/\s+/)[0] : 'Doctor';
    document.getElementById('pauseDrName').textContent = first;
    document.getElementById('pauseCtxCase').textContent = document.getElementById('diagnosis').value || '—';
    document.getElementById('pauseCtxPhase').textContent = _STEP_LABELS[_currentStep] || '—';
    document.getElementById('pauseCtxCoverage').textContent = `${_currentStep} / 7 steps`;
    const hh = Math.floor(frozenSecs / 3600).toString().padStart(2,'0');
    const mm = Math.floor((frozenSecs % 3600) / 60).toString().padStart(2,'0');
    const ss = (frozenSecs % 60).toString().padStart(2,'0');
    document.getElementById('pauseHH').textContent = hh;
    document.getElementById('pauseMM').textContent = mm;
    document.getElementById('pauseSS').textContent = ss;
    document.getElementById('pauseAtTime').textContent = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});
    document.getElementById('pauseBackdrop').classList.add('on');
  }

  function hidePauseModal() {
    document.getElementById('pauseBackdrop').classList.remove('on');
  }

  async function resumeFromPause() {
    hidePauseModal();
    if (!sessionId) return;
    try {
      const data = await postJson(`/api/sessions/${sessionId}/resume`, {});
      _elapsedCarry = data.duration_seconds || 0;
      _sessionStart = Date.now();
      _startTimer();
      toast('Session resumed', 'ok');
    } catch(e) {
      _sessionStart = Date.now();
      _startTimer();
    }
  }

  async function pauseSession() {
    if (!sessionId) return;
    if (!_currentResident) {
      toast('Log in to save your session — click "Log in" above', 'err');
      return;
    }
    spin('pauseBtn', true);
    try {
      const frozenSecs = _elapsedTotal();
      // Send only the NEW segment (not carry — DB already has carry from prior pauses)
      const segSecs = _sessionStart ? Math.round((Date.now() - _sessionStart) / 1000) : 0;
      await postJson(`/api/sessions/${sessionId}/pause`, { elapsed_seconds: segSecs });
      _stopTimer();
      _sessionStart = null;
      _elapsedCarry = 0;
      showPauseModal(frozenSecs);
    } catch(e) { toast(e.message); } finally { spin('pauseBtn', false); }
  }
  async function doLogout() {
    await fetch('/api/auth/logout', {method:'POST'});
    location.href = '/';
  }
  function showLoginModal() {
    document.getElementById('loginBackdrop').classList.add('on');
    setTimeout(() => document.getElementById('loginEmail').focus(), 50);
  }
  function hideLoginModal() {
    document.getElementById('loginBackdrop').classList.remove('on');
    document.getElementById('loginMsg').style.display = 'none';
    document.getElementById('loginDevLink').style.display = 'none';
    document.getElementById('loginEmail').value = '';
    document.getElementById('loginSubmitBtn').disabled = false;
    document.getElementById('loginSubmitBtn').textContent = 'Send sign-in link';
  }
  async function doTestLogin() {
    try {
      const r = await fetch('/api/auth/test-login', {method:'POST'});
      if (!r.ok) throw new Error('Test login failed');
      hideLoginModal();
      await checkAuth();
      toast('Logged in as Test Resident', 'ok');
    } catch(e) { toast(e.message); }
  }
  async function doRequestLink() {
    const email = document.getElementById('loginEmail').value.trim();
    if (!email) return;
    const btn = document.getElementById('loginSubmitBtn');
    const msg = document.getElementById('loginMsg');
    btn.disabled = true;
    btn.textContent = 'Sending…';
    msg.style.display = 'none';
    document.getElementById('loginDevLink').style.display = 'none';
    try {
      const r = await fetch('/api/auth/request-link', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({email}),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'Request failed');
      msg.textContent = data.message;
      msg.className = 'cb-login-msg';
      msg.style.display = 'block';
      if (data.dev_link) {
        const anchor = document.getElementById('loginDevAnchor');
        anchor.href = data.dev_link;
        anchor.textContent = data.dev_link;
        document.getElementById('loginDevLink').style.display = 'block';
      }
      btn.textContent = 'Resend link';
      btn.disabled = false;
    } catch(e) {
      msg.textContent = e.message;
      msg.className = 'cb-login-msg err';
      msg.style.display = 'block';
      btn.textContent = 'Send sign-in link';
      btn.disabled = false;
    }
  }
  async function resumeCase(sid) {
    try {
      const data = await postJson(`/api/sessions/${sid}/resume`, {});
      sessionId = data.session_id;
      const cp = data.case_preview || {};
      _patientGender = (cp.gender||'female').toLowerCase();
      _ttsEnabled = true;
      document.getElementById('sessionBadge').hidden = false;
      document.getElementById('pauseBtn').hidden = false;

      // Case preview panels
      document.getElementById('pvCC').textContent = cp.chief_complaint||'—';
      document.getElementById('pvAge').textContent = cp.age||'—';
      document.getElementById('pvGender').textContent = cp.gender||'—';
      const race = cp.race||''; document.getElementById('pvRace').textContent = race;
      document.getElementById('pvRaceChip').hidden = !race;
      const occ = cp.occupation||''; document.getElementById('pvOccupation').textContent = occ;
      document.getElementById('pvOccChip').hidden = !occ;
      const v = cp.vitals||{};
      document.getElementById('pvHR').textContent = v.heart_rate||'—';
      document.getElementById('pvBP').textContent = v.blood_pressure||'—';
      document.getElementById('pvSPO2').textContent = v.oxygen_saturation||'—';
      document.getElementById('pvRR').textContent = v.respiratory_rate||'—';
      const gender = _patientGender;
      document.getElementById('pvPersonaAvatar').textContent = gender.startsWith('f')?'F':gender.startsWith('m')?'M':'?';
      const nameEl = document.getElementById('pvPersonaName');
      nameEl.textContent = occ||(gender.startsWith('f')?'Female patient':'Male patient');
      document.getElementById('pvPersona').textContent = cp.persona_description||'';
      _patientInitials = occ.trim().split(/\s+/).filter(Boolean).slice(0,2).map(w=>w[0]).join('').toUpperCase()||'PT';
      document.getElementById('vitalsStrip').hidden = false;

      // Stepper
      const qa = data.qa_pairs||[], invs = data.investigations||[], dx = data.diagnosis_guess;
      let step = 2;
      if (qa.length) step = 3;
      if (invs.length) step = 5;
      if (dx) step = 6;
      setStepperStep(step);

      // Status bar
      document.getElementById('sbCase').innerHTML = `<strong>CASE</strong> ${sid}`;
      document.getElementById('sbSession').innerHTML = `<strong>SESSION</strong> resumed`;
      const sbAuto = document.getElementById('sbAutosave');
      if (sbAuto) { sbAuto.hidden = false; document.getElementById('sbSaveTime').textContent = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'}); }

      // Chat transcript
      document.getElementById('chatArea').innerHTML = '<div class="cb-chat-divider">Session resumed · continue your assessment</div>';
      _chatTurnCount = 0;
      qa.forEach(({question, answer}) => { addBubble('you', question); addBubble('patient', answer); });

      // Investigations
      _orderedSet = new Set(); _orderedCount = 0;
      document.getElementById('invResults').innerHTML = '';
      invs.forEach(inv => renderInv(inv));

      // Interpretations
      document.getElementById('interpList').innerHTML = '';
      (data.interpretations||[]).forEach(txt => {
        const el = document.createElement('div');
        el.className = 'cb-interp-item';
        el.textContent = txt;
        document.getElementById('interpList').appendChild(el);
      });

      // Diagnosis
      if (dx) document.getElementById('diagnosisGuess').value = dx;

      // Eval panel
      const evalRes = document.getElementById('evalResults');
      if (evalRes) { evalRes.hidden = false; }
      document.getElementById('perfStatusBadge').hidden = false;
      document.getElementById('perfStatusBadge').className = 'cb-perf-status in-progress';
      document.getElementById('perfStatusBadge').innerHTML = '<svg width="7" height="7" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg> Resumed · complete your assessment';
      _elapsedCarry = data.duration_seconds || 0;
      _sessionStart = Date.now();
      _startTimer();
      toast('Session resumed', 'ok');

      // Remove ?resume= from URL without reload
      const url = new URL(window.location);
      url.searchParams.delete('resume');
      history.replaceState({}, '', url);
    } catch(e) { toast('Could not resume: ' + (e.message||'unknown error')); }
  }

  /* ── Session timer ── */
  let _elapsedCarry = 0;   // seconds accumulated from prior paused segments
  let _timerInterval = null;

  function _elapsedTotal() {
    const seg = _sessionStart ? Math.round((Date.now() - _sessionStart) / 1000) : 0;
    return _elapsedCarry + seg;
  }

  function _fmtTime(s) {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return m + ':' + sec;
  }

  function _startTimer() {
    const el = document.getElementById('sbTimer');
    if (el) el.hidden = false;
    if (_timerInterval) clearInterval(_timerInterval);
    _timerInterval = setInterval(() => {
      const tv = document.getElementById('sbTimerVal');
      if (tv) tv.textContent = _fmtTime(_elapsedTotal());
    }, 1000);
  }

  function _stopTimer() {
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
  }

  /* ── voice I/O ── */
  let _ttsEnabled = true;
  let _patientGender = 'female';
  let _patientInitials = 'PT';
  let _studentInitials = 'HF';
  let _chatTurnCount = 0;
  let _sessionStart = null;
  let _recognition = null;
  let _isRecording = false;

  function setupSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { document.getElementById('micBtn').hidden = true; return; }
    _recognition = new SR();
    _recognition.continuous = false;
    _recognition.interimResults = true;
    _recognition.lang = 'en-US';
    _recognition.onresult = ev => {
      const t = Array.from(ev.results).map(r => r[0].transcript).join('');
      document.getElementById('question').value = t;
      if (ev.results[ev.results.length - 1].isFinal) { _stopRecording(); askQuestion(); }
    };
    _recognition.onerror = ev => {
      _stopRecording();
      const msgs = {
        'not-allowed':       'Mic access denied — enable it in browser/OS settings.',
        'service-not-allowed':'Speech recognition blocked on this browser or platform.',
        'network':           'Speech service unreachable — check your connection.',
        'no-speech':         null, // user just didn't speak, no toast needed
      };
      const msg = ev.error in msgs ? msgs[ev.error] : 'Mic error: ' + ev.error;
      if (msg) toast(msg);
    };
    _recognition.onend = _stopRecording;
  }

  function toggleMic() {
    if (!sessionId) { toast('Generate a case first.'); return; }
    if (!_recognition) { toast('Voice input not supported on this browser. Use the text field.'); return; }
    _isRecording ? _stopRecording() : _startRecording();
  }
  async function _startRecording() {
    // Explicitly request mic permission first so mobile browsers show the prompt
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop()); // release immediately — SpeechRecognition manages its own
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? 'Mic access denied — enable it in your browser/OS settings.'
        : 'Microphone unavailable: ' + err.message;
      toast(msg);
      return;
    }
    window.speechSynthesis.cancel();
    _isRecording = true;
    document.getElementById('micBtn').classList.add('mic-recording');
    document.getElementById('question').value = '';
    document.getElementById('question').placeholder = 'Listening…';
    try { _recognition.start(); } catch(_) {}
  }
  function _stopRecording() {
    _isRecording = false;
    document.getElementById('micBtn').classList.remove('mic-recording');
    document.getElementById('question').placeholder = 'Ask the patient a question…';
    try { _recognition.stop(); } catch(_) {}
  }

  function speakText(text) {
    if (!_ttsEnabled || !window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const malePrefs   = ['Daniel', 'Alex', 'Google UK English Male', 'Microsoft David', 'Aaron'];
    const femalePrefs = ['Samantha', 'Karen', 'Google UK English Female', 'Microsoft Zira', 'Moira'];
    const prefs = _patientGender === 'male' ? malePrefs : femalePrefs;
    const pick = voices.find(v => prefs.some(p => v.name.includes(p)))
              || voices.find(v => v.lang.startsWith('en'))
              || null;
    if (pick) utt.voice = pick;
    utt.rate  = 0.93;
    utt.pitch = _patientGender === 'male' ? 0.88 : 1.05;
    window.speechSynthesis.speak(utt);
  }

  function toggleTts() {
    _ttsEnabled = !_ttsEnabled;
    window.speechSynthesis.cancel();
    const track = document.getElementById('ttsTrack');
    if (track) track.classList.toggle('off', !_ttsEnabled);
  }

  // Preload voices (Chrome loads them async)
  if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  /* ── helpers ── */
  async function postJson(url, body) {
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const txt = await r.text();
    let data; try{data=JSON.parse(txt)}catch{data={detail:txt}};
    if(!r.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
    return data;
  }
  function req() { if(!sessionId) throw new Error('Generate a case first.'); }
  function toast(msg, type='error') {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = `fixed bottom-5 right-5 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-2xl fade-in ${type==='ok'?'bg-emerald-600 text-white':'bg-red-600 text-white'}`;
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(()=>el.hidden=true, 3500);
  }
  function spin(id, on) {
    const b = document.getElementById(id); if(!b) return;
    if(on){b._orig=b.innerHTML; b.disabled=true; b.innerHTML='<svg class="w-4 h-4 spinning inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>';}
    else{b.innerHTML=b._orig||b.innerHTML; b.disabled=false;}
  }
  let _orderedSet = new Set();
  let _orderedCount = 0;

  function _invKey(name) {
    return name.toLowerCase().replace(/[^a-z0-9]/g, '');
  }
  function _markChipOrdered(name) {
    const key = _invKey(name);
    if (_orderedSet.has(key)) return;
    _orderedSet.add(key);
    const chip = document.querySelector(`.qchip[data-inv="${key}"]`);
    if (chip) chip.classList.add('ordered');
    _orderedCount++;
    const badge = document.getElementById('invCountBadge');
    if (badge) { badge.textContent = _orderedCount + ' ordered'; badge.hidden = false; }
  }
  function quickOrder(name) {
    if (!sessionId) { toast('Generate a case first.'); return; }
    document.getElementById('investigation').value = name;
    orderInvestigation();
  }

  function tickAutosave() {
    const el = document.getElementById('sbSaveTime');
    if(el) el.textContent = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  }

  function setDifficulty(val) {
    document.getElementById('difficulty').value = val;
    document.querySelectorAll('.cb-diff-btn').forEach(b => {
      b.classList.toggle('cb-diff-active', b.dataset.val === val);
    });
  }

  function setStepperStep(doneUpTo) {
    _currentStep = doneUpTo;
    for (let i = 1; i <= 7; i++) {
      const el = document.getElementById('step'+i);
      if (!el) continue;
      el.className = 'cb-step' + (i < doneUpTo ? ' done' : i === doneUpTo ? ' active' : '');
      const num = document.getElementById('sn'+i);
      if (num) num.textContent = (i < doneUpTo) ? '✓' : String(i);
    }
  }

  /* ── generateCase ── */
  async function generateCase() {
    spin('generateBtn',true);
    try {
      const data = await postJson('/api/cases', {
        diagnosis: document.getElementById('diagnosis').value,
        difficulty: document.getElementById('difficulty').value
      });
      sessionId = data.session_id;
      document.getElementById('pauseBtn').hidden = false;
      _patientGender = (data.case_preview.gender||'female').toLowerCase();
      window.speechSynthesis && window.speechSynthesis.cancel();
      _ttsEnabled = true;
      const track = document.getElementById('ttsTrack');
      if (track) track.classList.remove('off');
      const voiceLbl = document.getElementById('ttsVoiceLabel');
      if (voiceLbl) voiceLbl.textContent = 'EN-US · ' + (_patientGender==='male'?'Adult Male':_patientGender==='female'?'Adult Female':'Auto');
      document.getElementById('sessionBadge').hidden = false;
      setStepperStep(2);

      // Status bar
      const diagLabel = document.getElementById('diagnosis').value||'—';
      document.getElementById('sbCase').innerHTML = `<strong>CASE</strong> ${data.session_id||'—'}`;
      document.getElementById('sbSession').innerHTML = `<strong>SESSION</strong> active`;
      document.getElementById('sbDiag').innerHTML = `<strong>Dx</strong> ${diagLabel}`;
      const sbAuto = document.getElementById('sbAutosave');
      sbAuto.hidden = false;
      document.getElementById('sbSaveTime').textContent = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});

      const v = data.case_preview.vitals||{};
      const cp = data.case_preview;

      // Chief complaint + chips
      document.getElementById('pvCC').textContent = cp.chief_complaint||'—';
      document.getElementById('pvAge').textContent = cp.age||'—';
      document.getElementById('pvGender').textContent = (cp.gender||'—');
      const race = cp.race||'';
      document.getElementById('pvRace').textContent = race;
      document.getElementById('pvRaceChip').hidden = !race;
      const occ = cp.occupation||'';
      document.getElementById('pvOccupation').textContent = occ;
      document.getElementById('pvOccChip').hidden = !occ;

      // Vital tiles with color-coding
      function vitalTile(tileId, flagId, value, {critLow, cautLow, cautHigh, critHigh} = {}) {
        const tile = document.getElementById(tileId);
        const flag = document.getElementById(flagId);
        if (!tile || value == null) return;
        let cls = 'normal', flagTxt = '', flagCls = '';
        if (critHigh != null && value >= critHigh) { cls='critical'; flagTxt='HIGH'; flagCls='high'; }
        else if (critLow != null && value <= critLow) { cls='critical'; flagTxt='LOW'; flagCls='low'; }
        else if (cautHigh != null && value >= cautHigh) { cls='caution'; flagTxt='↑'; flagCls='warn'; }
        else if (cautLow != null && value <= cautLow) { cls='caution'; flagTxt='↓'; flagCls='warn'; }
        tile.className = 'cb-vital ' + cls;
        if (flag) { flag.textContent = flagTxt; flag.className = 'cb-v-flag ' + flagCls; flag.hidden = !flagTxt; }
      }
      const hr = v.heart_rate, spo2 = v.oxygen_saturation, rr = v.respiratory_rate;
      const bpParts = (v.blood_pressure||'').split('/');
      const sbp = parseInt(bpParts[0])||null;
      document.getElementById('pvHR').textContent  = hr||'—';
      document.getElementById('pvBP').textContent  = v.blood_pressure||'—';
      document.getElementById('pvSPO2').textContent= spo2||'—';
      document.getElementById('pvRR').textContent  = rr||'—';
      vitalTile('vtHR',   'vfHR',   hr,   {critLow:40, cautLow:50, cautHigh:100, critHigh:140});
      vitalTile('vtBP',   'vfBP',   sbp,  {critLow:80, cautLow:100, cautHigh:160, critHigh:180});
      vitalTile('vtSPO2', 'vfSPO2', spo2, {critLow:88, cautLow:94});
      vitalTile('vtRR',   'vfRR',   rr,   {cautHigh:22, critHigh:35});

      // Persona card
      const persona = cp.persona_description||'';
      document.getElementById('pvPersona').textContent = persona;
      const gender = (cp.gender||'').toLowerCase();
      const initials = gender.startsWith('f') ? 'F' : gender.startsWith('m') ? 'M' : '?';
      document.getElementById('pvPersonaAvatar').textContent = initials;
      const nameEl = document.getElementById('pvPersonaName');
      nameEl.textContent = occ ? occ : (gender.startsWith('f') ? 'Female patient' : gender.startsWith('m') ? 'Male patient' : 'Patient');

      // Derive patient initials from occupation words (e.g. "College Student" → "CS")
      const occWords = occ.trim().split(/\s+/).filter(Boolean);
      if (occWords.length >= 2) {
        _patientInitials = (occWords[0][0] + occWords[occWords.length-1][0]).toUpperCase();
      } else if (occWords.length === 1) {
        _patientInitials = occWords[0].substring(0,2).toUpperCase();
      } else {
        _patientInitials = gender.startsWith('m') ? 'MP' : 'FP';
      }
      _chatTurnCount = 0;
      const turnBadge = document.getElementById('chatTurnBadge');
      if (turnBadge) turnBadge.hidden = true;

      document.getElementById('vitalsStrip').hidden = false;

      document.getElementById('chatArea').innerHTML = '<div class="cb-chat-divider">Session started · begin your assessment</div>';
      _orderedSet = new Set();
      _orderedCount = 0;
      document.querySelectorAll('.qchip').forEach(c => c.classList.remove('ordered'));
      const invBadge = document.getElementById('invCountBadge');
      if (invBadge) invBadge.hidden = true;
      document.getElementById('invResults').innerHTML = '';
      // reset physical exam
      _examSelected = new Set(); _examDone = new Set(); _examResultMap = {};
      document.getElementById('examResults').innerHTML = '';
      document.querySelectorAll('.cb-exam-pin').forEach(p => p.remove());
      document.querySelectorAll('.cb-exam-sys-row').forEach(r => {
        r.classList.remove('done');
        r.style.borderLeftColor = 'transparent';
        const cb = r.querySelector('.cb-exam-sys-cb');
        cb.classList.remove('checked');
      });
      const examBadge = document.getElementById('examDoneBadge');
      if (examBadge) examBadge.hidden = true;
      document.getElementById('ecgContent').innerHTML = '';
      document.getElementById('ecgPanel').hidden = true;
      document.getElementById('cxrImage').src = '';
      document.getElementById('cxrPanel').hidden = true;
      document.getElementById('interpList').innerHTML = '';
      document.getElementById('interpretation').value = '';
      const interpCounter = document.getElementById('interpCharCount');
      if (interpCounter) interpCounter.textContent = '0 / 1000';
      document.getElementById('diagnosisGuess').value = '';
      document.getElementById('doctorConsults').innerHTML = '';
      document.getElementById('doctorQuestion').value = '';
      document.getElementById('diagnosisResult').hidden = true;
      _elapsedCarry = 0;
      _sessionStart = Date.now();
      _startTimer();
      // Show eval panel in "In Progress" state immediately
      const evalRes = document.getElementById('evalResults');
      evalRes.hidden = false;
      document.getElementById('perfStatusBadge').hidden = false;
      document.getElementById('perfStatusBadge').className = 'cb-perf-status in-progress';
      document.getElementById('perfStatusBadge').innerHTML = '<svg width="7" height="7" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg> In Progress · partial report';
      // Reset donut to empty
      const arc0 = document.getElementById('scoreArc');
      arc0.style.stroke = 'var(--teal-500)';
      arc0.style.strokeDashoffset = '226.2';
      document.getElementById('scoreVal').textContent = '—';
      document.getElementById('scoreFeedback').textContent = 'Complete your assessment, then click Evaluate.';
      document.getElementById('evalDx').innerHTML = '—';
      document.getElementById('evalSymptoms').innerHTML = '—';
      document.getElementById('evalQuality').innerHTML = '—';
      document.getElementById('evalQualityBar').style.width = '0%';
      document.getElementById('evalInvestigations').innerHTML = '—';
      document.getElementById('evalInvBar').style.width = '0%';
      document.getElementById('evalTimeOnTask').innerHTML = '—';
      document.getElementById('evalTimeBar').style.width = '0%';
      document.getElementById('evalReasoning').innerHTML = '';
      document.getElementById('evalLearning').innerHTML = '';
      startGraphPolling();
      toast('Case generated', 'ok');
    } catch(e){ toast(e.message); } finally { spin('generateBtn',false); }
  }

  /* ── askQuestion ── */
  async function askQuestion() {
    try {
      req();
      const q = document.getElementById('question').value.trim(); if(!q) return;
      addBubble('you', q);
      document.getElementById('question').value = '';
      const typId = addTyping();
      spin('askBtn',true);
      const data = await postJson('/api/questions', {session_id:sessionId, question:q});
      document.getElementById(typId)?.remove();
      addBubble('patient', data.answer);
      tickAutosave();
      speakText(data.answer);
      setTimeout(refreshGraph, 12000);
    } catch(e){ toast(e.message); } finally { spin('askBtn',false); }
  }
  function tagQuestion(text) {
    const t = text.toLowerCase();
    if (/\b(tell me|describe|explain|what (happened|brings)|can you tell)\b/.test(t)) return {label:'Open question', cls:'cb-tag-r'};
    if (/\b(history|previous|before|past|ever had|has this happened)\b/.test(t)) return {label:'PMHx', cls:'cb-tag-r'};
    if (/\b(medication|medicine|drug|inhaler|pill|tablet|take any|prescribed)\b/.test(t)) return {label:'Medication hx', cls:'cb-tag-r'};
    if (/\b(chest pain|fever|severe|worst|blood|faint|collapse|lose consciousness)\b/.test(t)) return {label:'Red flags', cls:'cb-tag-r'};
    if (/\b(when|where|how long|how bad|scale|rate your|started|onset)\b/.test(t)) return {label:'Symptom hx', cls:'cb-tag-r'};
    return null;
  }

  function addBubble(role, text) {
    const area = document.getElementById('chatArea');
    const isStudent = role === 'you';
    const w = document.createElement('div');
    w.className = 'cb-msg fade-in ' + (isStudent ? 's' : 'p');
    // Avatar
    const av = document.createElement('div');
    av.className = 'cb-msg-av';
    av.textContent = isStudent ? _studentInitials : _patientInitials;
    // Content wrapper
    const cnt = document.createElement('div');
    // Bubble
    const bbl = document.createElement('div');
    bbl.className = 'cb-msg-bubble';
    bbl.textContent = text;
    cnt.appendChild(bbl);
    // Meta row
    const meta = document.createElement('div');
    meta.className = 'cb-msg-meta';
    const ts = new Date().toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit'});
    meta.appendChild(Object.assign(document.createElement('span'), {textContent: ts}));
    if (isStudent) {
      _chatTurnCount++;
      const badge = document.getElementById('chatTurnBadge');
      if (badge) { badge.textContent = _chatTurnCount + ' turn' + (_chatTurnCount===1?'':'s'); badge.hidden = false; }
      const tag = tagQuestion(text);
      if (tag) {
        const tagEl = Object.assign(document.createElement('span'), {className:'cb-msg-tag '+tag.cls, textContent:tag.label});
        meta.appendChild(tagEl);
      }
    } else {
      const rpl = document.createElement('button');
      rpl.style.cssText = 'background:transparent;border:0;padding:0;cursor:pointer;color:var(--ink-400);display:flex;align-items:center;gap:2px;margin-left:auto';
      rpl.title = 'Replay';
      rpl.innerHTML = '<svg width="10" height="10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072M12 6v12m-3.536-9.536a5 5 0 000 7.072"/></svg>';
      rpl.onclick = () => speakText(text);
      meta.appendChild(rpl);
    }
    cnt.appendChild(meta);
    w.appendChild(av);
    w.appendChild(cnt);
    area.appendChild(w);
    area.scrollTop = area.scrollHeight;
  }

  function addTyping() {
    const area = document.getElementById('chatArea');
    const id = 'typ'+Date.now();
    const w = document.createElement('div');
    w.id = id;
    w.className = 'cb-chat-typing fade-in';
    w.innerHTML = '<span>Patient is thinking</span><span class="cb-typing-dots"><span></span><span></span><span></span></span>';
    area.appendChild(w);
    area.scrollTop = area.scrollHeight;
    return id;
  }

  /* ── investigations ── */
  async function orderInvestigation() {
    try {
      req();
      const inv = document.getElementById('investigation').value.trim(); if(!inv) return;
      spin('invBtn',true);
      const data = await postJson('/api/investigations', {session_id:sessionId, investigation:inv});
      _markChipOrdered(inv);
      // also mark any sub-investigations returned
      if (data.investigations) data.investigations.forEach(r => _markChipOrdered(r.investigation||inv));
      renderInv(data);
      tickAutosave();
      setTimeout(refreshGraph, 12000);
    } catch(e){ toast(e.message); } finally { spin('invBtn',false); }
  }
  function renderInv(result) {
    if(result.investigations){result.investigations.forEach(renderInv);return;}

    // ECG with SVG → render in the ECG panel
    if(result.visual_payload&&result.visual_payload.svg){
      const panel = document.getElementById('ecgPanel');
      document.getElementById('ecgContent').innerHTML = result.visual_payload.svg;
      const hr = document.getElementById('pvHR').textContent.replace(' bpm','').trim();
      const meta = document.getElementById('ecgMeta');
      const ecgNow = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
      document.getElementById('ecgMetaLeft').innerHTML = hr
        ? `RATE <strong>${hr}</strong> bpm &nbsp;·&nbsp; 25 mm/s &nbsp;·&nbsp; 10 mm/mV`
        : '25 mm/s &nbsp;·&nbsp; 10 mm/mV';
      document.getElementById('ecgMetaRight').innerHTML =
        `${(result.summary||'REVIEW TRACING FOR ABNORMALITIES').replace(/\.\s*$/, '').toUpperCase()} &nbsp;·&nbsp; ${ecgNow}`;
      meta.hidden = false;
      panel.hidden = false;

      const c = document.getElementById('invResults');
      const badge = document.createElement('div');
      badge.className='fade-in cb-result';
      badge.innerHTML=`<div class="cb-result-head" onclick="toggleInvCard(this)">
        <div class="cb-result-icon">ECG</div>
        <div><div class="cb-result-title">12-Lead ECG</div><div class="cb-result-sub">${result.summary||'Review tracing'}</div></div>
        <span class="cb-result-stamp">available</span>
        <svg class="cb-result-chev" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </div>
      <div class="cb-result-body" style="padding:10px 12px;font-size:12px;color:var(--ink-600);line-height:1.5">
        ${result.summary||'12-Lead ECG rendered.'} <span style="color:var(--ink-400)">Scroll up to view the tracing.</span>
      </div>`;
      c.appendChild(badge);
      return;
    }

    // CXR with base64 → render in CXR panel
    if(result.visual_payload&&result.visual_payload.data){
      const panel = document.getElementById('cxrPanel');
      document.getElementById('cxrImage').src = `data:image/jpeg;base64,${result.visual_payload.data}`;

      const finding = result.visual_payload.finding||'';
      document.getElementById('cxrFindingLabel').textContent = finding ? `Finding: ${finding}` : 'RoentGen-v2 · synth';

      const age = document.getElementById('pvAge').textContent;
      const gender = (document.getElementById('pvGender').textContent||'').charAt(0).toUpperCase();
      const ovEl = document.getElementById('cxrOverlayPatient');
      if(ovEl) ovEl.innerHTML = `PATIENT<br/>MRN ----<br/>${gender} · ${age}`;

      const banner = document.getElementById('cxrFindingBanner');
      const textEl = document.getElementById('cxrFindingText');
      if(textEl) textEl.textContent = result.summary || (finding ? `Radiograph consistent with ${finding} pattern.` : '');
      if(banner) banner.hidden = !result.summary && !finding;

      panel.hidden = false;

      const c = document.getElementById('invResults');
      const badge = document.createElement('div');
      badge.className='fade-in cb-result';
      badge.innerHTML=`<div class="cb-result-head" onclick="toggleInvCard(this)">
        <div class="cb-result-icon">CXR</div>
        <div><div class="cb-result-title">Chest Radiograph · PA</div><div class="cb-result-sub">${finding ? 'Finding: '+finding : 'Synthetic radiograph'}</div></div>
        <span class="cb-result-stamp">available</span>
        <svg class="cb-result-chev" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </div>
      <div class="cb-result-body" style="padding:10px 12px;font-size:12px;color:var(--ink-600);line-height:1.5">
        ${result.summary||'Chest radiograph rendered.'}${finding ? '' : ''} <span style="color:var(--ink-400)">Scroll up to view the image.</span>
      </div>`;
      c.appendChild(badge);
      return;
    }

    // Lab / text result
    const c = document.getElementById('invResults');
    const card = document.createElement('div');
    card.className='fade-in cb-result';

    const abbr = result.investigation.substring(0,3).toUpperCase();
    // Count abnormal values for the sub-label
    let abnCount = 0;
    if (result.report) result.report.forEach(item => {
      if (!item.name || item.value === undefined) return;
      const num = parseFloat(String(item.value));
      const m = (item.reference_range||'').match(/^([\d.]+)\s*[–\-]\s*([\d.]+)/);
      if (!isNaN(num) && m && (num < parseFloat(m[1]) || num > parseFloat(m[2]))) abnCount++;
    });
    const subLabel = abnCount > 0 ? `${abnCount} abnormal value${abnCount>1?'s':''}` : (result.summary||result.status||'');
    card.innerHTML=`<div class="cb-result-head" onclick="toggleInvCard(this)">
      <div class="cb-result-icon">${abbr}</div>
      <div><div class="cb-result-title">${result.investigation}</div><div class="cb-result-sub">${subLabel}</div></div>
      <span class="cb-result-stamp">${result.status}</span>
      <svg class="cb-result-chev" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
    </div>`;

    const body = document.createElement('div');
    body.className = 'cb-result-body';

    if(result.report&&result.report.length){
      const hasValues = result.report.some(i=>i.name&&i.value!==undefined);
      if(hasValues){
        const tbl = document.createElement('table');
        tbl.className='cb-lab-table';
        tbl.innerHTML='<thead><tr><th>Analyte</th><th>Result</th><th>Ref. range</th><th style="text-align:right">Flag</th></tr></thead>';
        const tbody=document.createElement('tbody');
        result.report.forEach(item=>{
          if(item.finding){
            const tr=document.createElement('tr');
            tr.innerHTML=`<td colspan="4" style="padding:7px 10px;font-size:12px;color:var(--ink-700)">${item.finding}</td>`;
            tbody.appendChild(tr);return;
          }
          const val=String(item.value||'');
          const ref=item.reference_range||'';
          let abn='', flag='—';
          const num=parseFloat(val);
          const m=ref.match(/^([\d.]+)\s*[–\-]\s*([\d.]+)/);
          if(!isNaN(num)&&m){
            if(num<parseFloat(m[1])){ abn='abn'; flag='L'; }
            else if(num>parseFloat(m[2])){ abn='abn'; flag='H'; }
          }
          const tr=document.createElement('tr');
          tr.innerHTML=`<td class="cb-lab-name">${item.name||''}</td>
            <td><span class="cb-lab-val ${abn}">${val}</span>${item.unit?' <span class="cb-lab-ref">'+item.unit+'</span>':''}</td>
            <td class="cb-lab-ref">${ref}</td>
            <td class="cb-lab-flag" style="text-align:right">${flag}</td>`;
          tbody.appendChild(tr);
        });
        tbl.appendChild(tbody);
        body.appendChild(tbl);
        if (result.summary && result.summary !== result.investigation + ' completed') {
          const pat = document.createElement('div');
          pat.className = 'cb-lab-pattern';
          pat.innerHTML = `<strong>Pattern:</strong> ${result.summary}`;
          body.appendChild(pat);
        }
      } else {
        const prose=document.createElement('div');
        prose.style.cssText='padding:10px 12px;display:flex;flex-direction:column;gap:4px';
        result.report.forEach(item=>{
          if(item.finding){
            const p=document.createElement('p');
            p.style.cssText='font-size:12px;color:var(--ink-700);margin:0;line-height:1.5';
            p.textContent=item.finding;
            prose.appendChild(p);
          }
        });
        body.appendChild(prose);
      }
    }
    card.appendChild(body);
    c.appendChild(card);
  }

  function toggleInvCard(headEl) {
    headEl.closest('.cb-result').classList.toggle('open');
  }
  function togglePanel(id) {
    document.getElementById(id).classList.toggle('collapsed');
    if (id === 'graphPanel') setTimeout(graphFit, 300);
  }

  let _kgExpanded = false;
  function toggleKgExpand() {
    _kgExpanded ? collapseKg() : expandKg();
  }
  function _kgRerender() {
    if (_graphData.nodes && _graphData.nodes.length) {
      renderGraph(_graphData.nodes, _graphData.edges||[]);
    }
  }
  function expandKg() {
    _kgExpanded = true;
    document.getElementById('graphPanel').classList.remove('collapsed');
    document.getElementById('graphPanel').classList.add('kg-expanded');
    document.getElementById('kgBackdrop').classList.add('on');
    const btn = document.getElementById('kgExpandBtn');
    btn.title = 'Collapse';
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4.5 1v3.5H1M11 4.5H7.5V1M7.5 11V7.5H11M1 7.5h3.5V11"/></svg>`;
    setTimeout(_kgRerender, 320);
  }
  function collapseKg() {
    _kgExpanded = false;
    document.getElementById('graphPanel').classList.remove('kg-expanded');
    document.getElementById('kgBackdrop').classList.remove('on');
    const btn = document.getElementById('kgExpandBtn');
    btn.title = 'Expand';
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M1 4.5V1h3.5M7.5 1H11v3.5M11 7.5V11H7.5M4.5 11H1V7.5"/></svg>`;
    setTimeout(_kgRerender, 320);
  }

  function updateInterpCount(el) {
    const n = (el.value||'').length;
    const counter = document.getElementById('interpCharCount');
    if (counter) counter.textContent = n + ' / 1000';
  }

  /* ── physical exam ── */
  // Percentage coordinates (0–100) in the displayed image container.
  // Calibrated for 190×310 object-fit:cover of the 1024×1024 body PNG
  // (cover scale = 310/1024 = 0.303; horizontal crop = (310–190)/2 = 60px on each side).
  const EXAM_PINS = {
    'Neurological':        {x:30, y:10},
    'General':             {x:70, y:10},
    'ENT/Head & Neck':     {x:50, y:22},
    'Cardiovascular':      {x:28, y:37},
    'Respiratory':         {x:72, y:37},
    'Abdominal':           {x:50, y:51},
    'Musculoskeletal':     {x:12, y:46},
    'Dermatological':      {x:88, y:46},
    'Peripheral Vascular': {x:50, y:87},
  };
  const EXAM_COLORS = {
    'General':             '#64748b',
    'Cardiovascular':      '#ef4444',
    'Respiratory':         '#06b6d4',
    'Abdominal':           '#f97316',
    'Neurological':        '#8b5cf6',
    'Musculoskeletal':     '#10b981',
    'Dermatological':      '#ec4899',
    'ENT/Head & Neck':     '#f59e0b',
    'Peripheral Vascular': '#6366f1',
  };
  const EXAM_ABBR = {
    'General': 'GN', 'Cardiovascular': 'CV', 'Respiratory': 'RS',
    'Abdominal': 'AB', 'Neurological': 'NE', 'Musculoskeletal': 'MS',
    'Dermatological': 'DM', 'ENT/Head & Neck': 'EN', 'Peripheral Vascular': 'PV',
  };
  let _examSelected = new Set();
  let _examDone = new Set();
  let _examResultMap = {};

  function toggleExamSys(row, system) {
    if (row.classList.contains('done')) return;
    const cb = row.querySelector('.cb-exam-sys-cb');
    const color = EXAM_COLORS[system] || 'var(--teal-600)';
    if (_examSelected.has(system)) {
      _examSelected.delete(system);
      cb.classList.remove('checked');
      row.style.borderLeftColor = 'transparent';
    } else {
      _examSelected.add(system);
      cb.classList.add('checked');
      row.style.borderLeftColor = color;
    }
  }

  async function performExam() {
    req();
    const systems = [..._examSelected].filter(s => !_examDone.has(s));
    if (!systems.length) { toast('Select at least one system to examine'); return; }
    spin('examBtn', true);
    try {
      const data = await postJson('/api/physical-exam', {session_id: sessionId, systems});
      data.results.forEach(r => renderExamResult(r));
      // clear checkboxes
      _examSelected.clear();
      document.querySelectorAll('.cb-exam-sys-cb.checked').forEach(el => el.classList.remove('checked'));
    } catch(e) { toast(e.message); } finally { spin('examBtn', false); }
  }

  function renderExamResult(result) {
    const system = result.system;
    _examDone.add(system);
    _examResultMap[system] = result;

    // Mark row as done with system color
    const sysColor = EXAM_COLORS[system] || 'var(--teal-600)';
    document.querySelectorAll('.cb-exam-sys-row').forEach(row => {
      if (row.getAttribute('onclick') && row.getAttribute('onclick').includes(`'${system}'`)) {
        row.classList.add('done');
        row.style.borderLeftColor = sysColor;
        const cb = row.querySelector('.cb-exam-sys-cb');
        cb.classList.add('checked');
      }
    });

    // Update done badge
    const badge = document.getElementById('examDoneBadge');
    badge.textContent = _examDone.size + ' examined';
    badge.hidden = false;

    // Add pin to body SVG
    addExamPin(system, result.abnormal);

    // Add result card
    const container = document.getElementById('examResults');
    const card = document.createElement('div');
    card.className = 'fade-in cb-result';
    card.id = 'examCard_' + system.replace(/\W/g,'_');
    const abbr = EXAM_ABBR[system] || system.substring(0,2).toUpperCase();
    const flagColor = result.abnormal ? 'var(--critical)' : 'var(--normal)';
    const flagLabel = result.abnormal ? 'Abnormal' : 'Normal';
    card.innerHTML = `<div class="cb-result-head" onclick="toggleInvCard(this)">
      <div class="cb-result-icon" style="border-color:${sysColor};color:${sysColor}">${abbr}</div>
      <div><div class="cb-result-title">${system}</div><div class="cb-result-sub">${result.summary}</div></div>
      <span class="cb-result-stamp" style="color:${flagColor}">${flagLabel}</span>
      <svg class="cb-result-chev" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
    </div>`;
    const body = document.createElement('div');
    body.className = 'cb-result-body';
    const ul = document.createElement('ul');
    ul.style.cssText = 'margin:0;padding:10px 12px 10px 28px;display:flex;flex-direction:column;gap:4px';
    result.findings.forEach(f => {
      const li = document.createElement('li');
      li.style.cssText = 'font-size:12px;color:var(--ink-700);line-height:1.5';
      li.textContent = f;
      ul.appendChild(li);
    });
    body.appendChild(ul);
    card.appendChild(body);
    container.appendChild(card);
  }

  function addExamPin(system, abnormal) {
    const wrap = document.getElementById('examBodyWrap');
    if (!wrap) return;
    const pos = EXAM_PINS[system];
    if (!pos) return;

    const svgW = 100, svgH = 100; // pins use % coords; scale = rect.px / 100
    const wrapRect = wrap.getBoundingClientRect();
    const imgEl = wrap.querySelector('img');
    if (!imgEl) return;
    const rect = imgEl.getBoundingClientRect();
    const scaleX = rect.width / svgW;
    const scaleY = rect.height / svgH;
    // offset of SVG within the wrap div (accounts for centering / padding)
    const svgOffX = rect.left - wrapRect.left;
    const svgOffY = rect.top - wrapRect.top;
    const sysColor = EXAM_COLORS[system] || '#64748b';

    const pin = document.createElement('div');
    pin.className = 'cb-exam-pin cb-exam-pin-pulse ' + (abnormal ? 'abnormal' : 'normal');
    pin.title = system;
    pin.textContent = EXAM_ABBR[system] || system.substring(0,2).toUpperCase();
    pin.style.left = (svgOffX + pos.x * scaleX) + 'px';
    pin.style.top  = (svgOffY + pos.y * scaleY) + 'px';
    pin.style.outline = `2px solid ${sysColor}`;
    pin.style.outlineOffset = '2px';
    pin.onclick = () => {
      const cardId = 'examCard_' + system.replace(/\W/g,'_');
      const card = document.getElementById(cardId);
      if (card) {
        card.classList.add('open');
        card.scrollIntoView({behavior:'smooth', block:'nearest'});
      }
    };
    wrap.appendChild(pin);
  }

  /* ── interpretations ── */
  async function submitInterpretation() {
    try {
      req();
      const txt = document.getElementById('interpretation').value.trim(); if(!txt) return;
      spin('interpBtn',true);
      const data = await postJson('/api/interpretations', {session_id:sessionId, interpretation:txt});
      const ul = document.getElementById('interpList'); ul.innerHTML='';
      data.interpretations.forEach((item,i)=>{
        const li = document.createElement('li');
        li.className = 'fade-in';
        li.style.cssText = 'display:flex;gap:6px;align-items:flex-start';
        li.innerHTML = `<span style="color:var(--teal-600);font-weight:700;flex-shrink:0;font-size:11px">${i+1}.</span><span>${item}</span>`;
        ul.appendChild(li);
      });
      document.getElementById('interpretation').value='';
      const counter = document.getElementById('interpCharCount');
      if (counter) counter.textContent = '0 / 1000';
    } catch(e){ toast(e.message); } finally { spin('interpBtn',false); }
  }

  /* ── diagnosis ── */
  async function submitDiagnosis() {
    try {
      req();
      const dx = document.getElementById('diagnosisGuess').value.trim(); if(!dx) return;
      spin('diagnosisBtn',true);
      const data = await postJson('/api/diagnosis', {session_id:sessionId, diagnosis:dx});
      const el = document.getElementById('diagnosisResult');
      el.innerHTML=`<strong style="font-weight:600">Submitted:</strong> ${data.diagnosis_guess}`;
      el.hidden=false;
    } catch(e){ toast(e.message); } finally { spin('diagnosisBtn',false); }
  }

  /* ── doctor consult ── */
  async function callDoctor() {
    try {
      req();
      const q = document.getElementById('doctorQuestion').value.trim(); if(!q) return;
      spin('doctorBtn',true);
      const data = await postJson('/api/doctor-consult', {session_id:sessionId, question:q});
      const card=document.createElement('div');
      card.className='fade-in cb-doctor-response';
      let html='';
      if(data.concise_advice) html+=`<p>${data.concise_advice}</p>`;
      if(data.recommended_next_steps&&data.recommended_next_steps.length){
        html+=`<ul style="margin:6px 0 0;padding-left:12px;font-size:11.5px;color:var(--ink-600);list-style:none">`;
        data.recommended_next_steps.forEach(s=>html+=`<li style="margin-top:2px">→ ${s}</li>`);
        html+=`</ul>`;
      }
      if(data.safety_warnings&&data.safety_warnings.length){
        html+=`<div style="margin:6px 0 0;background:var(--critical-bg);border-radius:5px;padding:6px 8px;font-size:11px;color:var(--critical)">`;
        data.safety_warnings.forEach(w=>html+=`<p style="margin:0">⚠ ${w}</p>`);
        html+=`</div>`;
      }
      card.innerHTML=html;
      document.getElementById('doctorConsults').appendChild(card);
      document.getElementById('doctorQuestion').value='';
    } catch(e){ toast(e.message); } finally { spin('doctorBtn',false); }
  }

  /* ── knowledge graph ── */
  let _graphSim = null, _graphData = {nodes:[], edges:[]}, _pollTimer = null;
  let _graphZoom = null, _graphZoomBehavior = null;
  let _liveNodes = [];  // simulation-mutated node objects (have real x,y)

  const NODE_COLORS = {
    symptom:'#f97316',    condition:'#f97316',    sign:'#f97316',
    riskfactor:'#f59e0b', risk:'#f59e0b',
    investigation:'#f97316',
    finding:'#6366f1',    result:'#6366f1',
    medication:'#60a5fa', drug:'#60a5fa',          treatment:'#60a5fa',
    diagnosis:'#10b981',
    time:'#a78bfa',       temporal:'#a78bfa',      duration:'#a78bfa',
    person:'#8b5cf6',     patient:'#8b5cf6',
    bodypart:'#94a3b8',   location:'#94a3b8',
    action:'#0ea5e9',     measurement:'#0ea5e9',
  };
  function nodeColor(d) {
    for (const lbl of (d.labels||[])) {
      const key = lbl.toLowerCase();
      for (const [k,v] of Object.entries(NODE_COLORS)) { if(key.includes(k)) return v; }
    }
    return '#94a3b8';
  }
  function isDiagnosis(d) {
    return (d.labels||[]).some(l => l.toLowerCase().includes('diagnos'));
  }

  function graphZoomIn()  { if(_graphZoomBehavior) d3.select('#graphSvg').transition().duration(250).call(_graphZoomBehavior.scaleBy, 1.4); }
  function graphZoomOut() { if(_graphZoomBehavior) d3.select('#graphSvg').transition().duration(250).call(_graphZoomBehavior.scaleBy, 0.7); }
  function graphFit() {
    if (!_graphZoomBehavior || !_liveNodes.length) return;
    const svg = document.getElementById('graphSvg');
    const W = svg.clientWidth || 900, H = svg.clientHeight || 400;
    const xs = _liveNodes.map(n => n.x || W/2);
    const ys = _liveNodes.map(n => n.y || H/2);
    const x0 = Math.min(...xs) - 40, x1 = Math.max(...xs) + 40;
    const y0 = Math.min(...ys) - 40, y1 = Math.max(...ys) + 40;
    const scale = Math.min(0.92 * W / (x1 - x0), 0.92 * H / (y1 - y0), 2);
    const tx = W / 2 - scale * (x0 + x1) / 2;
    const ty = H / 2 - scale * (y0 + y1) / 2;
    d3.select('#graphSvg').transition().duration(400)
      .call(_graphZoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  let _selectedId = null;

  function _nodeR(n) { return isDiagnosis(n) ? 28 : 22; }

  function resetHighlight(linkSel, edgeLblSel, nodeSel) {
    _selectedId = null;
    linkSel.attr('stroke','rgba(255,255,255,0.18)').attr('stroke-width',1.5).attr('stroke-opacity',1);
    edgeLblSel.attr('opacity',0);
    nodeSel.select('circle').attr('r',_nodeR).attr('fill-opacity',1).attr('stroke-opacity',0.25);
    nodeSel.selectAll('text').attr('opacity',1);
  }

  function highlightNode(d, links, linkSel, edgeLblSel, nodeSel) {
    if (_selectedId === d.id) { resetHighlight(linkSel, edgeLblSel, nodeSel); return; }
    _selectedId = d.id;

    const adjIds = new Set([d.id]);
    links.forEach(lk => {
      if (lk.source.id===d.id) adjIds.add(lk.target.id);
      if (lk.target.id===d.id) adjIds.add(lk.source.id);
    });
    const isAdj = lk => lk.source.id===d.id || lk.target.id===d.id;

    linkSel
      .attr('stroke', lk => isAdj(lk) ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.08)')
      .attr('stroke-width', lk => isAdj(lk) ? 2 : 1.5)
      .attr('stroke-opacity', 1);
    edgeLblSel.attr('opacity', lk => isAdj(lk) ? 1 : 0);
    nodeSel.select('circle')
      .attr('r', n => n.id===d.id ? _nodeR(n)+4 : _nodeR(n))
      .attr('fill-opacity', n => adjIds.has(n.id) ? 1 : 0.18)
      .attr('stroke-opacity', n => adjIds.has(n.id) ? 0.3 : 0.05);
    nodeSel.selectAll('text').attr('opacity', n => adjIds.has(n.id) ? 1 : 0.15);
  }

  function renderGraph(nodes, edges) {
    const svgEl = document.getElementById('graphSvg');
    const W = svgEl.clientWidth || 900, H = svgEl.clientHeight || 400;
    const tooltip = document.getElementById('graphTooltip');
    document.getElementById('graphEmpty').style.display = nodes.length ? 'none' : 'flex';
    if (!nodes.length) return;

    if (_graphSim) _graphSim.stop();
    _selectedId = null;
    _liveNodes = nodes;
    const s = d3.select('#graphSvg');
    s.selectAll('*').remove();

    // Defs: arrowhead markers per color
    const defs = s.append('defs');
    defs.append('marker')
      .attr('id','arr').attr('viewBox','-0 -5 10 10')
      .attr('refX',30).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-5L10,0L0,5').attr('fill','rgba(255,255,255,0.35)');
    defs.append('marker')
      .attr('id','arr-hi').attr('viewBox','-0 -5 10 10')
      .attr('refX',30).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-5L10,0L0,5').attr('fill','rgba(255,255,255,0.7)');

    const byId = Object.fromEntries(nodes.map(n=>([n.id,n])));
    const links = edges.filter(e=>byId[e.source]&&byId[e.target]);

    // Detect connected components via union-find
    const _uf = {};
    nodes.forEach(n=>{ _uf[n.id]=n.id; });
    function _find(x){ return _uf[x]===x?x:(_uf[x]=_find(_uf[x])); }
    edges.forEach(e=>{
      const s=typeof e.source==='object'?e.source.id:e.source;
      const t=typeof e.target==='object'?e.target.id:e.target;
      if(_uf[s]!==undefined&&_uf[t]!==undefined) _uf[_find(s)]=_find(t);
    });
    const _compMap={};
    nodes.forEach(n=>{ const r=_find(n.id); (_compMap[r]=_compMap[r]||[]).push(n); });
    const _comps=Object.values(_compMap);

    // Assign a grid center to each component, spread across the canvas
    const _cols=Math.ceil(Math.sqrt(_comps.length));
    const _rows=Math.ceil(_comps.length/_cols);
    _comps.forEach((comp,i)=>{
      const cx=((i%_cols)+0.5)*(W/_cols);
      const cy=(Math.floor(i/_cols)+0.5)*(H/_rows);
      comp.forEach(n=>{ n._cx=cx; n._cy=cy; });
    });

    // Scatter initial positions near each component's assigned center
    nodes.forEach(n=>{ if(n.x==null){n.x=n._cx+(Math.random()-.5)*60;n.y=n._cy+(Math.random()-.5)*60;} });

    // Zoom/pan
    const root = s.append('g');
    _graphZoomBehavior = d3.zoom()
      .scaleExtent([0.15, 4])
      .on('zoom', e => { root.attr('transform', e.transform); });
    s.call(_graphZoomBehavior).on('dblclick.zoom', null);

    // Click background → reset highlight
    s.on('click', () => resetHighlight(linkSel, edgeLblSel, nodeSel));

    // ── Edge layer ────────────────────────────────────────────────────────────
    const linkSel = root.append('g').attr('class','links').selectAll('line')
      .data(links).enter().append('line')
      .attr('stroke','rgba(255,255,255,0.18)').attr('stroke-width',1.5)
      .attr('marker-end','url(#arr)')
      .on('mouseenter',(ev,d)=>{ if(d.fact){ tooltip.textContent=d.fact; tooltip.style.display='block'; } })
      .on('mousemove',ev=>{ const r=svgEl.getBoundingClientRect(); tooltip.style.left=(ev.clientX-r.left+14)+'px'; tooltip.style.top=(ev.clientY-r.top-10)+'px'; })
      .on('mouseleave',()=>tooltip.style.display='none');

    // ── Edge labels — hidden by default, shown only on node click ───────────────
    const edgeLblSel = root.append('g').selectAll('text')
      .data(links.filter(e=>e.fact)).enter().append('text')
      .attr('class','edge-label')
      .attr('text-anchor','middle').attr('font-size','9px').attr('font-weight','500')
      .attr('fill','rgba(255,255,255,0.75)').attr('opacity',0).attr('pointer-events','none')
      .text(d=>{ const f=d.fact||''; return f.length>14?f.slice(0,14)+'…':f; });

    // ── Node layer ─────────────────────────────────────────────────────────────
    const nodeSel = root.append('g').attr('class','nodes').selectAll('g')
      .data(nodes).enter().append('g').attr('class','node-circle')
      .call(d3.drag()
        .on('start',(ev,d)=>{ ev.sourceEvent.stopPropagation(); if(!ev.active)_graphSim.alphaTarget(.3).restart(); d.fx=d.x;d.fy=d.y; })
        .on('drag', (ev,d)=>{ d.fx=ev.x;d.fy=ev.y; })
        .on('end',  (ev,d)=>{ if(!ev.active)_graphSim.alphaTarget(0); d.fx=null;d.fy=null; })
      )
      .on('click',(ev,d)=>{ ev.stopPropagation(); highlightNode(d,links,linkSel,edgeLblSel,nodeSel); })
      .on('mouseenter',(ev,d)=>{
        const type = d.labels&&d.labels.length ? d.labels[0] : '';
        tooltip.innerHTML = `<strong>${d.name}</strong>${type?` <span style="opacity:.7;font-size:10px">[${type}]</span>`:''}`;
        tooltip.style.display='block';
      })
      .on('mousemove',ev=>{ const r=svgEl.getBoundingClientRect(); tooltip.style.left=(ev.clientX-r.left+14)+'px'; tooltip.style.top=(ev.clientY-r.top-10)+'px'; })
      .on('mouseleave',()=>tooltip.style.display='none');

    // Node circle — solid fill
    nodeSel.append('circle')
      .attr('r', _nodeR)
      .attr('fill', d=>nodeColor(d))
      .attr('fill-opacity', 1)
      .attr('stroke', 'rgba(255,255,255,0.25)')
      .attr('stroke-width', 1.5);

    // Node name — white, clipped to radius
    nodeSel.append('text')
      .attr('class','node-name')
      .attr('text-anchor','middle').attr('dy', d => isDiagnosis(d) ? '-.25em' : '.35em')
      .attr('font-size', d => isDiagnosis(d) ? '10px' : '9px').attr('font-weight','700')
      .attr('fill','white').attr('pointer-events','none')
      .text(d=>{ const n=d.name||''; return n.length>11?n.slice(0,11)+'…':n; });

    // Sub-label: "working dx" only for diagnosis nodes
    nodeSel.append('text')
      .attr('class','node-type')
      .attr('text-anchor','middle').attr('dy','1em')
      .attr('font-size','7.5px').attr('font-weight','400')
      .attr('fill','rgba(255,255,255,0.7)').attr('pointer-events','none')
      .text(d=> isDiagnosis(d) ? 'working dx' : '');

    // ── Simulation ─────────────────────────────────────────────────────────────
    // Per-component cluster force: pulls each node toward its subgraph's assigned center
    function _clusterForce(alpha) {
      nodes.forEach(n=>{
        n.vx += (n._cx - n.x) * 0.12 * alpha;
        n.vy += (n._cy - n.y) * 0.12 * alpha;
      });
    }
    _graphSim = d3.forceSimulation(nodes)
      .force('link',    d3.forceLink(links).id(d=>d.id).distance(80).strength(0.9))
      .force('charge',  d3.forceManyBody().strength(-160))
      .force('collide', d3.forceCollide(d => _nodeR(d) + 14))
      .force('cluster', _clusterForce)
      .alphaDecay(0.03)
      .on('tick',()=>{
        // Clamp nodes within SVG bounds so they can never drift off screen
        nodes.forEach(d => {
          d.x = Math.max(30, Math.min(W - 30, d.x));
          d.y = Math.max(30, Math.min(H - 30, d.y));
        });
        linkSel
          .attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
        edgeLblSel
          .attr('x',d=>(d.source.x+d.target.x)/2)
          .attr('y',d=>(d.source.y+d.target.y)/2-5);
        nodeSel.attr('transform',d=>`translate(${d.x},${d.y})`);
      })
      .on('end', graphFit);
  }

  async function refreshGraph() {
    if (!sessionId) return;
    try {
      const r = await fetch('/api/graph/'+sessionId);
      if (!r.ok) return;
      const data = await r.json();
      const nPrev = _graphData.nodes.length;
      _graphData = data;
      if (data.nodes && (data.nodes.length !== nPrev || nPrev === 0)) {
        renderGraph(data.nodes||[], data.edges||[]);
      }
      const badge = document.getElementById('graphBadge');
      const status = document.getElementById('graphStatus');
      if (data.nodes && data.nodes.length) {
        badge.textContent = data.nodes.length+' entities · '+data.edges.length+' relations';
        badge.hidden = false;
        status.textContent = 'Live';
        status.className = 'text-xs text-emerald-300 font-medium';
        document.getElementById('graphControls').hidden = false;
        // KG stat bar
        const statBar = document.getElementById('kgStatBar');
        if (statBar) {
          document.getElementById('kgNodeCount').textContent = data.nodes.length;
          document.getElementById('kgEdgeCount').textContent = data.edges.length;
          const newNodes = data.nodes.length - nPrev;
          const recentSpan = document.getElementById('kgRecentSpan');
          if (recentSpan && newNodes > 0) {
            document.getElementById('kgRecentCount').textContent = '+' + newNodes;
            recentSpan.hidden = false;
          }
          statBar.hidden = false;
        }
      }
    } catch(_) {}
  }

  function startGraphPolling() {
    document.getElementById('graphPanel').hidden = false;
    document.getElementById('graphEmpty').style.display = 'flex';
    document.getElementById('graphStatus').textContent = 'Waiting for interactions…';
    document.getElementById('graphStatus').className = 'text-xs text-indigo-300 italic';
    document.getElementById('graphBadge').hidden = true;
    _graphData = {nodes:[], edges:[]};
    _liveNodes = [];
    _graphZoom = null; _graphZoomBehavior = null;
    if (_graphSim) { _graphSim.stop(); _graphSim=null; }
    d3.select('#graphSvg').selectAll('*').remove();
    document.getElementById('graphControls').hidden = true;
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(refreshGraph, 8000);
  }

  /* ── evaluate ── */
  async function evaluateCase() {
    try {
      req();
      spin('evaluateBtn',true);
      const _segSecs = _sessionStart ? Math.round((Date.now() - _sessionStart) / 1000) : 0;
      const data = await postJson('/api/evaluate', {session_id:sessionId, elapsed_seconds:_segSecs});
      const score = data.overall_score||0;
      // Donut ring
      const arc = document.getElementById('scoreArc');
      const arcColor = score>=75 ? 'var(--normal)' : score>=50 ? 'var(--caution)' : 'var(--critical)';
      arc.style.stroke = arcColor;
      arc.style.strokeDashoffset = String(226.2*(1-score/100));
      document.getElementById('scoreVal').textContent = score;
      document.getElementById('scoreFeedback').textContent = data.brief_feedback||'';

      const dx=data.correct_diagnosis_guess||{};
      document.getElementById('evalDx').innerHTML = dx.is_correct
        ?`<span style="color:var(--normal);font-weight:600">✓ Correct</span> <span style="color:var(--ink-400)">— ${dx.guess||''}</span>`
        :`<span style="color:var(--critical);font-weight:600">✗ Incorrect</span> <span style="color:var(--ink-400)">— ${dx.guess||'not submitted'}</span>`;

      const found=data.key_symptoms_identified||[], missed=data.key_symptoms_missed||[];
      document.getElementById('evalSymptoms').innerHTML=`<span style="color:var(--normal);font-weight:600">${found.length} found</span>`+(missed.length?` <span style="color:var(--critical)">${missed.length} missed</span>`:'')+(missed.length?`<p style="color:var(--ink-400);font-size:11px;margin:2px 0 0">${missed.slice(0,3).join(', ')}</p>`:'');

      const rel=data.relevance_of_questions||{};
      const tot=(rel.relevant_count||0)+(rel.irrelevant_count||0);
      const pct=tot>0?Math.round((rel.relevant_count||0)/tot*100):0;
      document.getElementById('evalQuality').innerHTML = rel.relevant_count!==undefined
        ?`<span style="color:var(--normal);font-weight:600">${rel.relevant_count}</span> relevant · <span style="color:var(--ink-400)">${rel.irrelevant_count}</span> off-target`:'—';
      document.getElementById('evalQualityBar').style.width = pct+'%';
      document.getElementById('evalQualityBar').style.background = pct>=70?'var(--normal)':pct>=40?'var(--caution)':'var(--critical)';

      function fbItem(icon, iconBg, iconColor, text, bold) {
        const d=document.createElement('div'); d.className='cb-feedback-item';
        d.innerHTML=`<div class="cb-feedback-bullet" style="background:${iconBg};color:${iconColor}">${icon}</div><span>${bold?`<strong>${text}</strong>`:text}</span>`;
        return d;
      }
      const rf=data.reasoning_feedback||{}, rEl=document.getElementById('evalReasoning'); rEl.innerHTML='';
      if(rf.diagnostic_path) rEl.appendChild(fbItem('→','var(--teal-50)','var(--teal-700)',rf.diagnostic_path,false));
      (rf.missed_reasoning_steps||[]).forEach(s=>rEl.appendChild(fbItem('!','var(--caution-bg)','var(--caution)',s,false)));

      const lr=data.learning_recommendations||{}, lEl=document.getElementById('evalLearning'); lEl.innerHTML='';
      if(lr.focus_next) lEl.appendChild(fbItem('★','var(--teal-50)','var(--teal-700)',lr.focus_next,true));
      (lr.action_items||[]).forEach(item=>lEl.appendChild(fbItem('·','var(--ink-50)','var(--ink-500)',item,false)));

      // Investigations metric
      document.getElementById('evalInvestigations').innerHTML =
        `<span>${_orderedCount}</span><small style="font-size:11px;color:var(--ink-400);font-weight:500"> ordered</small>`;
      const invPct = Math.min(_orderedCount * 20, 100);
      document.getElementById('evalInvBar').style.width = invPct + '%';

      // Time on Task metric
      const _totalElapsed = _elapsedTotal();
      if (_totalElapsed > 0) {
        document.getElementById('evalTimeOnTask').textContent = _fmtTime(_totalElapsed);
        const timePct = Math.min(Math.round(_totalElapsed / 18), 100); // 30 min = 100%
        document.getElementById('evalTimeBar').style.width = timePct + '%';
      }

      // Flip status badge to complete
      const statusBadge = document.getElementById('perfStatusBadge');
      if (statusBadge) {
        statusBadge.className = 'cb-perf-status complete';
        statusBadge.innerHTML = '<svg width="7" height="7" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg> Final · complete report';
      }

      const res=document.getElementById('evalResults'); res.hidden=false;
      res.scrollIntoView({behavior:'smooth',block:'start'});
    } catch(e){ toast(e.message); } finally { spin('evaluateBtn',false); }
  }

  document.addEventListener('keydown', e => { if (e.key === 'Escape' && _kgExpanded) collapseKg(); });

  setupSpeechRecognition();

  /* ── CXR viewer controls ── */
  let _cxrMode = 'wl';
  let _cxrInverted = false;
  let _cxrScale = 1, _cxrTx = 0, _cxrTy = 0;
  let _cxrW = 1500, _cxrL = -600;
  let _cxrDrag = null;

  function _cxrApply() {
    const img = document.getElementById('cxrImage');
    if (!img) return;
    const contrast   = Math.max(0.3, Math.min(4, 1500 / Math.max(100, _cxrW)));
    const brightness = Math.max(0.1, Math.min(4, 1 + (_cxrL + 600) / 1200));
    const inv = _cxrInverted ? 'invert(1) ' : '';
    img.style.filter = `${inv}brightness(${brightness.toFixed(2)}) contrast(${contrast.toFixed(2)})`;
    img.style.transform = `scale(${_cxrScale}) translate(${(_cxrTx/_cxrScale).toFixed(1)}px,${(_cxrTy/_cxrScale).toFixed(1)}px)`;
    img.style.transformOrigin = 'center center';
    const spans = document.querySelectorAll('.cb-cxr-window-info span');
    if (spans[0]) spans[0].textContent = `W: ${Math.round(_cxrW)}   L: ${Math.round(_cxrL)}`;
    if (spans[1]) spans[1].textContent = `ZOOM ${Math.round(_cxrScale * 100)}% · PA`;
  }

  function setCxrMode(mode) {
    _cxrMode = mode;
    document.querySelectorAll('.cb-cxr-tbtn[data-mode]').forEach(b => {
      if (b.dataset.mode !== 'invert') b.classList.toggle('on', b.dataset.mode === mode);
    });
    const stage = document.querySelector('.cb-cxr-stage');
    if (stage) stage.style.cursor = mode === 'pan' ? 'grab' : mode === 'zoom' ? 'ns-resize' : 'crosshair';
  }

  function toggleCxrInvert() {
    _cxrInverted = !_cxrInverted;
    const btn = document.querySelector('.cb-cxr-tbtn[data-mode="invert"]');
    if (btn) btn.classList.toggle('on', _cxrInverted);
    _cxrApply();
  }

  function _cxrReset() {
    _cxrScale = 1; _cxrTx = 0; _cxrTy = 0; _cxrW = 1500; _cxrL = -600; _cxrInverted = false;
    const btn = document.querySelector('.cb-cxr-tbtn[data-mode="invert"]');
    if (btn) btn.classList.remove('on');
    _cxrApply();
  }

  (function initCxrControls() {
    const stage = document.querySelector('.cb-cxr-stage');
    if (!stage) return;
    stage.style.cursor = 'crosshair';

    stage.addEventListener('mousedown', e => {
      if (!document.getElementById('cxrImage').src) return;
      e.preventDefault();
      _cxrDrag = {x: e.clientX, y: e.clientY, scale: _cxrScale, tx: _cxrTx, ty: _cxrTy, w: _cxrW, l: _cxrL};
      if (_cxrMode === 'pan') stage.style.cursor = 'grabbing';
    });

    stage.addEventListener('mousemove', e => {
      if (!_cxrDrag) return;
      const dx = e.clientX - _cxrDrag.x;
      const dy = e.clientY - _cxrDrag.y;
      if (_cxrMode === 'pan') {
        _cxrTx = _cxrDrag.tx + dx;
        _cxrTy = _cxrDrag.ty + dy;
      } else if (_cxrMode === 'zoom') {
        _cxrScale = Math.max(0.25, Math.min(8, _cxrDrag.scale * Math.pow(1.012, -dy)));
      } else {
        _cxrW = Math.max(100, Math.min(4000, _cxrDrag.w + dx * 6));
        _cxrL = Math.max(-2000, Math.min(2000, _cxrDrag.l - dy * 3));
      }
      _cxrApply();
    });

    const endDrag = () => { _cxrDrag = null; if (_cxrMode === 'pan') stage.style.cursor = 'grab'; };
    stage.addEventListener('mouseup', endDrag);
    stage.addEventListener('mouseleave', endDrag);

    stage.addEventListener('wheel', e => {
      if (!document.getElementById('cxrImage').src) return;
      e.preventDefault();
      _cxrScale = Math.max(0.25, Math.min(8, _cxrScale * (e.deltaY < 0 ? 1.12 : 0.9)));
      _cxrApply();
    }, {passive: false});
  })();

  /* ── Pause modal keyboard shortcuts ── */
  document.addEventListener('keydown', (e) => {
    if (!document.getElementById('pauseBackdrop').classList.contains('on')) return;
    if (e.key === 'Enter')  { e.preventDefault(); resumeFromPause(); }
    if (e.key === 'Escape') { e.preventDefault(); location.href = '/dashboard'; }
  });

  /* ── Startup ── */
  (async function() {
    await checkAuth();
    const _resumeId = new URLSearchParams(window.location.search).get('resume');
    if (_resumeId) resumeCase(_resumeId);
  })();
</script>
</body>
</html>
"""
