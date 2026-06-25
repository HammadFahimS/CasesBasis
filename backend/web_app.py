from __future__ import annotations

import logging
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pathlib import Path

from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

    # These must be set before torch is imported for the first time.
    # PyTorch's OMP/MKL thread pool creates one POSIX semaphore per worker
    # thread. On a multi-core Mac that's ~35 semaphores. Python's
    # resource_tracker atexit handler runs before PyTorch's C++ static
    # destructors, so it incorrectly warns about "leaked" semaphores on every
    # clean shutdown. Limiting the pool to 1 thread prevents the large
    # allocation entirely — inference on short text strings doesn't benefit
    # from parallelism anyway.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # ── Startup environment validation ─────────────────────────────────────────
    env = os.getenv("ENV", "development").lower()
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    _required: dict[str, str] = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "together": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }
    if provider in _required and not os.getenv(_required[provider]):
        raise RuntimeError(
            f"LLM_PROVIDER is '{provider}' but {_required[provider]} is not set. "
            "Add it to your .env file."
        )
    if env == "production" and not os.getenv("APP_SECRET"):
        raise RuntimeError(
            "APP_SECRET must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    run_migrations()

    from backend.simulated_case_repo import seed_gold_cases
    try:
        seed_gold_cases()
    except Exception as exc:  # never block startup on seeding
        logging.getLogger(__name__).warning("gold case seeding failed: %s", exc)

    from backend.cxr_service import _faiss_index
    logger = logging.getLogger(__name__)
    logger.info("Warming up CXR FAISS index ...")
    await asyncio.get_event_loop().run_in_executor(None, _faiss_index._load)
    logger.info("CXR FAISS index ready")

    yield

    _faiss_index._model = None
    gc.collect()


_limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(title="Clinical Case Simulator", lifespan=lifespan)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],  # "*" only until ALLOWED_ORIGINS is set in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["content-type"],
)


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


def _strip(v: str) -> str:
    return v.strip()


class RequestLinkRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) > 254:
            raise ValueError("email too long")
        return v


class ResidentLoginRequest(BaseModel):
    resident_code: str
    last_name: str

    @field_validator("resident_code", "last_name")
    @classmethod
    def validate_fields(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 100:
            raise ValueError("value too long")
        return v


class GenerateCaseRequest(BaseModel):
    diagnosis: str
    difficulty: str = "medium"

    @field_validator("diagnosis")
    @classmethod
    def validate_diagnosis(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("diagnosis is required")
        if len(v) > 200:
            raise ValueError("diagnosis must be under 200 characters")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")
        return v


class QuestionRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is required")
        if len(v) > 500:
            raise ValueError("question must be under 500 characters")
        return v


class InvestigationRequest(BaseModel):
    session_id: str
    investigation: str

    @field_validator("investigation")
    @classmethod
    def validate_investigation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("investigation is required")
        if len(v) > 200:
            raise ValueError("investigation must be under 200 characters")
        return v


class InterpretationRequest(BaseModel):
    session_id: str
    interpretation: str

    @field_validator("interpretation")
    @classmethod
    def validate_interpretation(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 2000:
            raise ValueError("interpretation must be under 2000 characters")
        return v


class DiagnosisRequest(BaseModel):
    session_id: str
    diagnosis: str

    @field_validator("diagnosis")
    @classmethod
    def validate_diagnosis(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("diagnosis is required")
        if len(v) > 200:
            raise ValueError("diagnosis must be under 200 characters")
        return v


class DoctorConsultRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is required")
        if len(v) > 500:
            raise ValueError("question must be under 500 characters")
        return v


class EvaluateRequest(BaseModel):
    session_id: str
    elapsed_seconds: int = 0


class PauseRequest(BaseModel):
    elapsed_seconds: int = 0


class ECGRerenderRequest(BaseModel):
    """Re-render the most recent ECG investigation at a different sample rate."""
    session_id: str
    sample_rate_hz: int = 500  # 100 or 500


class ECGAlgorithmRequest(BaseModel):
    """Inputs for the doctor-facing synthetic-ECG generator."""
    diagnosis: str
    heart_rate: int | None = None
    age: int | None = None
    gender: str | None = None
    blood_pressure: str | None = None


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


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/graph/{session_id}")
def get_graph_data(session_id: str, http_request: Request) -> dict[str, Any]:
    _get_owned_session(session_id, http_request)  # 404/403 if not the caller's
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
    return _inject_drawer(HTML_PAGE)


@app.get("/simulator/simulated", response_class=HTMLResponse)
def simulated_intro_page() -> str:
    """Intro / explainer page for the new simulated-dynamic-case mode.

    Reached from the 'New case' chooser modal when the resident picks the
    simulated option; the 'Start' button hands off to the simulator with
    `?mode=simulated`. Standalone interstitial — it carries its own slim top
    bar rather than the full home/analytics chrome."""
    return (Path(__file__).parent / "static" / "simulated_intro.html").read_text()


@app.get("/simulator/voice-room", response_class=HTMLResponse)
def voice_room_page() -> str:
    """Full-screen voice-driven simulation room (the simulated-case runtime UI).

    Standalone immersive page; the case id is read from ?case= in JS and the
    patient HUD + vitals are seeded from the loaded simulated case."""
    return (Path(__file__).parent / "static" / "voice_room.html").read_text()


# ── Voice model I/O (minimal request/response plumbing) ─────────────────────
def _voice_session_payload(session: Any) -> dict[str, Any]:
    """Seed data the Voice Room needs to render a freshly started session."""
    case = session.case
    engine = session.engine
    return {
        "session_id": session.id,
        "case_id": case.id,
        "meta": case.meta.model_dump(),
        "patient": case.patient.model_dump(exclude_none=True),
        "presentation": {
            "chief_complaint": case.presentation.chief_complaint,
            "opening_vignette": case.presentation.opening_vignette,
        },
        "vitals": engine.current_vitals().model_dump(exclude_none=True),
        "state": {"id": engine.run.current_state_id, "name": engine.state.name},
        # The case's own menu, so the panels offer only orderable tests / examinable
        # systems / pageable consultants (everything here returns a real result).
        "menu": {
            "exam_systems": [e.system for e in case.exam],
            "investigations": [
                {"id": i.id, "name": i.name, "category": i.category}
                for i in case.investigations
            ],
            "consultants": [
                {"id": a.id, "role": a.role} for a in _case_consultants(case)
            ],
        },
        # Everyone the resident can hear from, with the speaker class that styles
        # and voices them — drives the unified Room log and addressee routing.
        "roster": _case_roster(case),
    }


def _case_consultants(case: Any) -> list[Any]:
    from backend.consult_voice import consultants
    return consultants(case)


def _case_roster(case: Any) -> list[dict[str, str]]:
    """The room's voices, each tagged with its speaker class (patient/nurse/ems/
    family/rt/consult). Used by the front-end for avatars, colours and addressing."""
    from backend.actor_voice import actor_speaker
    return [
        {
            "id": a.id,
            "role": a.role,
            "speaker": actor_speaker(a),
            # None → present from the start; a number → en route until the case
            # clock reaches it. The front-end compares this to the live clock.
            "available_after_seconds": a.available_after_seconds,
        }
        for a in case.actors
    ]


def _persist_turn(session: Any, turn: Any) -> None:
    """Write a session through after a turn (ended turns are parked as concluded)."""
    from backend.voice_session import persist_session
    persist_session(session, "ended" if getattr(turn, "ended", False) else "active")


def _turn_payload(turn: Any) -> dict[str, Any]:
    """The wire shape of one grounded turn — shared by the voice and action routes."""
    return {
        "reply": turn.reply,
        "speaker": turn.speaker,
        "vitals": turn.vitals,
        "state": {"id": turn.state_id, "name": turn.state_name},
        "revealed": turn.revealed,
        "transitioned": turn.transitioned,
        "critical": turn.critical_error,
        "ended": turn.ended,
        "clock_events": turn.clock_events,
        "diagnostic": turn.diagnostic,
        "working_dx": getattr(turn, "working_dx", None),
        "cues": getattr(turn, "cues", []),
    }


@app.post("/api/voice/session")
@_limiter.limit("30/minute")
def api_voice_session(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Start a live engine-backed voice session for a case. Returns seed data."""
    resident_id = require_resident(request)
    case_id = (body.get("case_id") or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id required")
    from backend.nurse_brain import make_intent_matcher, opening_narration
    from backend.voice_session import CaseNotFound, start_session
    try:
        session = start_session(case_id, resident_id, intent_matcher=make_intent_matcher())
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="case not found")
    payload = _voice_session_payload(session)
    payload["intro"] = opening_narration(session)  # nurse's spoken bedside handoff
    return payload


@app.post("/api/voice/resume")
@_limiter.limit("30/minute")
def api_voice_resume(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Rehydrate a parked voice run for full visual resume (transcript + cards + state).

    Returns the same seed shape as ``/api/voice/session`` plus the saved transcript
    channels and diagnostics cards, and no ``intro`` — the handoff is not re-narrated."""
    resident_id = require_resident(request)
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    from backend.voice_session import AccessDenied, CaseNotFound, get_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    payload = _voice_session_payload(session)
    payload["resumed"] = True
    payload["history"] = session.history
    payload["patient_history"] = session.patient_history
    payload["cards"] = session.cards
    return payload


@app.post("/api/voice/stt")
@_limiter.limit("60/minute")
async def api_voice_stt(request: Request) -> dict[str, Any]:
    """Speech-to-text. The browser POSTs the raw recorded audio as the body."""
    require_resident(request)
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio")
    content_type = request.headers.get("content-type", "audio/webm")
    from backend.voice_service import transcribe
    try:
        text = transcribe(audio, content_type)
    except Exception as exc:
        logging.getLogger(__name__).warning("STT failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"transcription failed: {exc}")
    return {"text": text}


@app.post("/api/voice/llm")
@_limiter.limit("60/minute")
def api_voice_llm(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """The nurse's reply to one resident utterance.

    With a ``session_id`` the reply is grounded in the live SimulationEngine
    (real vitals/results, state transitions). Without one it falls back to the
    stateless generic nurse — both via the app's Cerebras provider."""
    resident_id = require_resident(request)
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        from backend.voice_service import nurse_reply
        try:
            return {"reply": nurse_reply(text, body.get("history"))}
        except Exception as exc:
            logging.getLogger(__name__).warning("LLM reply failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"nurse reply failed: {exc}")

    from backend.nurse_brain import run_turn
    from backend.voice_session import AccessDenied, CaseNotFound, get_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    try:
        turn = run_turn(session, text)
    except Exception as exc:
        logging.getLogger(__name__).warning("nurse turn failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"nurse reply failed: {exc}")
    _persist_turn(session, turn)
    return _turn_payload(turn)


@app.post("/api/voice/action")
@_limiter.limit("60/minute")
def api_voice_action(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """A grounded turn from an explicit panel click (examine a system, order a test,
    call a consultant). Deterministic — skips the interpret LLM call — and returns
    the same shape as ``/api/voice/llm`` so the front-end applies it identically."""
    resident_id = require_resident(request)
    session_id = (body.get("session_id") or "").strip()
    op = (body.get("op") or "").strip()
    target = (body.get("target") or "").strip() or None
    if not session_id or not op:
        raise HTTPException(status_code=400, detail="session_id and op required")

    from backend.nurse_brain import OPS, run_action
    if op not in OPS:
        raise HTTPException(status_code=400, detail=f"unknown op: {op}")
    from backend.voice_session import AccessDenied, CaseNotFound, get_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    try:
        turn = run_action(session, op, target)
    except Exception as exc:
        logging.getLogger(__name__).warning("voice action failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"action failed: {exc}")
    _persist_turn(session, turn)
    return _turn_payload(turn)


@app.post("/api/voice/llm-patient")
@_limiter.limit("60/minute")
def api_voice_patient(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """The patient's in-character spoken reply (role-play, not engine-grounded)."""
    resident_id = require_resident(request)
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    from backend.patient_voice import patient_reply
    from backend.voice_session import AccessDenied, CaseNotFound, get_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    try:
        reply = patient_reply(session, text)
    except Exception as exc:
        logging.getLogger(__name__).warning("patient reply failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"patient reply failed: {exc}")
    from backend.voice_session import persist_session
    persist_session(session)  # the patient exchange is part of the resumable transcript
    return {"reply": reply, "speaker": "patient"}


@app.post("/api/voice/debrief")
@_limiter.limit("30/minute")
def api_voice_debrief(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Conclude a run and return its end-of-case debrief (auto-end or manual 'End case')."""
    resident_id = require_resident(request)
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    from backend.voice_debrief import build_debrief
    from backend.voice_session import AccessDenied, CaseNotFound, end_session, get_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    try:
        debrief = build_debrief(session)
    except Exception as exc:
        logging.getLogger(__name__).warning("debrief build failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"debrief failed: {exc}")
    # Score the run into the resident's analytics (best-effort — never block the
    # debrief). The criteria are already in the payload; reduce to a 0-100 figure.
    try:
        from backend import voice_persistence
        from backend.sim_scoring import Criterion, overall_score
        assessment = debrief.get("assessment") or []
        score = overall_score([Criterion(**c) for c in assessment])
        duration = (debrief.get("summary") or {}).get("elapsed_seconds") or 0
        voice_persistence.record_evaluation(session_id, score, assessment, duration)
        debrief["overall_score"] = score
    except Exception as exc:
        logging.getLogger(__name__).warning("voice scoring failed for %s: %s", session_id, exc)
    end_session(session_id)  # viewing the debrief concludes the case
    return debrief


@app.post("/api/voice/pause")
@_limiter.limit("60/minute")
def api_voice_pause(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Park a run (status=paused) so the resident can leave and resume it later."""
    resident_id = require_resident(request)
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    from backend.voice_session import AccessDenied, CaseNotFound, get_session, persist_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    persist_session(session, "paused")
    return {"ok": True}


@app.post("/api/voice/unpause")
@_limiter.limit("60/minute")
def api_voice_unpause(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Resume a paused run in place: reset the clock so parked wall-time does not
    advance the engine, and flip the status back to active."""
    resident_id = require_resident(request)
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    import time

    from backend.voice_session import AccessDenied, CaseNotFound, get_session, persist_session
    try:
        session = get_session(session_id, resident_id)
    except CaseNotFound:
        raise HTTPException(status_code=404, detail="session not found")
    except AccessDenied:
        raise HTTPException(status_code=403, detail="access denied")
    session.last_tick = time.monotonic()   # the time parked must not tick the engine
    persist_session(session, "active")
    return {"ok": True}


@app.get("/api/voice/sessions")
@_limiter.limit("60/minute")
def api_voice_sessions(request: Request) -> dict[str, Any]:
    """The resident's recent/resumable voice runs (for the home 'resume' surface)."""
    resident_id = require_resident(request)
    from backend import voice_persistence
    from backend.simulated_case_repo import get_case
    titles: dict[str, str] = {}
    out = []
    for s in voice_persistence.list_for_resident(resident_id):
        cid = s["case_id"]
        if cid not in titles:
            row = get_case(cid)
            titles[cid] = (row["case"].get("meta", {}).get("title") if row else None) or cid
        out.append({**s, "title": titles[cid], "resumable": s["status"] != "ended"})
    return {"sessions": out}


@app.delete("/api/voice/sessions/{session_id}")
@_limiter.limit("60/minute")
def api_voice_session_delete(session_id: str, request: Request) -> dict[str, Any]:
    """Delete one of the resident's voice runs (owner-scoped)."""
    resident_id = require_resident(request)
    from backend import voice_persistence, voice_session
    if not voice_persistence.delete(session_id, resident_id):
        raise HTTPException(status_code=404, detail="session not found")
    voice_session._SESSIONS.pop(session_id, None)   # drop any live copy too
    return {"ok": True}


# Per-speaker TTS voices (each distinct so the room is tellable apart by ear) live
# in the cast registry, which resolves them from env at call time. The nurse
# default lives in voice_service; a rejected voice falls back there automatically.
from backend.actors import speaker_voice as _speaker_voice


@app.post("/api/voice/tts")
@_limiter.limit("60/minute")
def api_voice_tts(body: dict[str, Any], request: Request) -> Response:
    """Text-to-speech. Returns audio bytes (mp3) for the browser to play.

    ``speaker`` selects the voice (nurse default vs patient); an explicit
    ``voice`` still wins for auditioning."""
    require_resident(request)
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    voice = body.get("voice")
    if not voice:
        voice = _speaker_voice(body.get("speaker"))  # None → nurse default in voice_service
    from backend.voice_service import synthesize
    try:
        audio = synthesize(text, voice)
    except Exception as exc:
        logging.getLogger(__name__).warning("TTS failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}")
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/api/voice/sample")
def api_voice_sample(request: Request, voice: str | None = None, text: str | None = None) -> Response:
    """Audition a TTS voice in the browser. GET so it's clickable.

    e.g. /api/voice/sample?voice=professional%20woman"""
    require_resident(request)
    line = text or (
        "Heart rate is one hundred ten, sinus tachycardia. Fifteen litres "
        "non-rebreather is on — I'll read the gas back to you when it's up."
    )
    from backend.voice_service import synthesize
    try:
        audio = synthesize(line, voice)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}")
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/crud-simulations", response_class=HTMLResponse)
def crud_simulations_page() -> str:
    """CRUD catalogue for authored simulated dynamic cases."""
    return _serve_with_chrome("crud_simulations.html")


@app.get("/crud-simulations/edit", response_class=HTMLResponse)
def crud_simulations_editor_page() -> str:
    """Graph editor for one simulated case (case id read from ?id= in JS).

    Standalone full-screen tool — carries its own slim top bar instead of the
    shared chrome so the graph canvas gets the full viewport."""
    return (Path(__file__).parent / "static" / "sim_case_editor.html").read_text()


# ── Simulated-case CRUD API ─────────────────────────────────────────────────
@app.get("/api/simulated-cases")
def api_list_simulated_cases(http_request: Request) -> dict[str, Any]:
    require_resident(http_request)
    from backend.simulated_case_repo import list_cases as _list
    return {"cases": _list()}


@app.get("/api/simulated-cases/{case_id}")
def api_get_simulated_case(case_id: str, http_request: Request) -> dict[str, Any]:
    require_resident(http_request)
    from backend.simulated_case_repo import get_case
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@app.post("/api/simulated-cases")
@_limiter.limit("30/minute")
def api_create_simulated_case(body: dict[str, Any], request: Request) -> dict[str, Any]:
    resident_id = require_resident(request)
    from backend.simulated_case_repo import CaseValidationError, create_case
    try:
        return create_case(body.get("case", {}), body.get("layout", {}), created_by=resident_id)
    except CaseValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.put("/api/simulated-cases/{case_id}")
@_limiter.limit("60/minute")
def api_update_simulated_case(case_id: str, body: dict[str, Any], request: Request) -> dict[str, Any]:
    require_resident(request)
    from backend.simulated_case_repo import (
        CaseProtectedError,
        CaseValidationError,
        update_case,
    )
    try:
        return update_case(case_id, body.get("case", {}), body.get("layout"))
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")
    except CaseProtectedError:
        raise HTTPException(status_code=403, detail="gold cases cannot be edited — duplicate it first")
    except CaseValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/api/simulated-cases/{case_id}")
def api_delete_simulated_case(case_id: str, http_request: Request) -> dict[str, Any]:
    require_resident(http_request)
    from backend.simulated_case_repo import CaseProtectedError, delete_case
    try:
        delete_case(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")
    except CaseProtectedError:
        raise HTTPException(status_code=403, detail="gold cases cannot be deleted")
    return {"ok": True}


@app.post("/api/simulated-cases/{case_id}/duplicate")
@_limiter.limit("30/minute")
def api_duplicate_simulated_case(case_id: str, request: Request) -> dict[str, Any]:
    resident_id = require_resident(request)
    from backend.simulated_case_repo import duplicate_case
    try:
        return duplicate_case(case_id, created_by=resident_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


def _serve_with_chrome(filename: str) -> str:
    """Inject the shared top-bar + drawer markup into a page.

    Pages that opt into the new chrome contain a literal `<!--CHROME-->`
    placeholder right after `<body>`; this helper swaps it for the contents
    of `_chrome.html` at serve time so we don't duplicate ~150 lines of
    markup across pages. The `<!--DRAWER-->` placeholder is also supported
    for pages that have their own top bar but still need the drawer + burger
    behaviour (simulator, ECG-algorithm).
    """
    static_dir = Path(__file__).parent / "static"
    body = (static_dir / filename).read_text()
    body = body.replace("<!--CHROME-->", (static_dir / "_chrome.html").read_text())
    body = body.replace("<!--DRAWER-->", (static_dir / "_drawer.html").read_text())
    return body


def _inject_drawer(html: str) -> str:
    """Swap a `<!--DRAWER-->` placeholder for the drawer-only partial."""
    drawer = (Path(__file__).parent / "static" / "_drawer.html").read_text()
    return html.replace("<!--DRAWER-->", drawer)


@app.get("/home", response_class=HTMLResponse)
def home_page() -> str:
    """The new calm landing page after login — greeting, focus card, glance, recs."""
    return _serve_with_chrome("home.html")


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page() -> str:
    """The dense analytics & performance view (was /dashboard)."""
    return _serve_with_chrome("analytics.html")


@app.get("/dashboard")
def dashboard_redirect() -> RedirectResponse:
    """Legacy `/dashboard` URL — kept as a redirect so old bookmarks still land
    somewhere useful. The post-login default is now `/home`."""
    return RedirectResponse(url="/home", status_code=302)


@app.get("/ecg-algorithm", response_class=HTMLResponse)
def ecg_algorithm_page() -> str:
    """Doctor-facing review page for the synthetic ECG generation algorithm."""
    return _serve_with_chrome("ecg_algorithm.html")


@app.post("/api/ecg-algorithm/render")
@_limiter.limit("20/minute")
def render_synthetic_ecg(body: ECGAlgorithmRequest, request: Request) -> dict[str, Any]:
    """Force the legacy LLM+numpy synthetic ECG path, regardless of PTB-XL.

    Returns the SVG, the LLM-produced ECGParameters, and a markdown trace that
    explains how the SVG was constructed. Used by the algorithm-review page so
    clinicians can audit each step.
    """
    from agents.ecg import ECGParameterAgent
    from agents.prompt_loader import load
    from backend.ecg_algorithm import build_algorithm_markdown
    from backend.ecg_synthesizer import synthesize_ecg_svg

    diagnosis = (body.diagnosis or "").strip()
    if not diagnosis:
        raise HTTPException(status_code=400, detail="diagnosis is required")

    agent = ECGParameterAgent()
    params = agent.generate(diagnosis, heart_rate=body.heart_rate)
    svg = synthesize_ecg_svg(params)
    system_prompt = load("ecg", "system")
    markdown = build_algorithm_markdown(
        diagnosis=diagnosis,
        heart_rate=body.heart_rate,
        params=params,
        system_prompt=system_prompt,
        provider=agent.provider,
        model=agent.model,
        case_summary={
            "age": body.age,
            "gender": body.gender,
            "blood_pressure": body.blood_pressure,
        },
    )
    return {
        "svg": svg,
        "parameters": params.model_dump(),
        "markdown": markdown,
        "model": agent.model,
        "provider": agent.provider,
    }


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/auth/request-link")
@_limiter.limit("5/minute")
def request_magic_link(request: Request, body: RequestLinkRequest) -> dict[str, str]:
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
    redirect = RedirectResponse(url="/home", status_code=302)
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
@_limiter.limit("10/minute")
def login_resident(request: Request, body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
    resident_id = find_resident_by_code(body.resident_code, body.last_name)
    if resident_id is None:
        raise HTTPException(status_code=404, detail="not_found")
    _set_auth_cookie(response, resident_id)
    return {"ok": True}


@app.post("/api/auth/signup")
@_limiter.limit("5/minute")
def signup_resident(request: Request, body: ResidentLoginRequest, response: Response) -> dict[str, Any]:
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
    from backend import voice_persistence
    sessions = get_resident_sessions(resident_id)
    sessions += voice_persistence.get_resident_evaluations(resident_id)
    # newest first across both sources (voice runs carry source="voice")
    sessions.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return {"sessions": sessions}


@app.get("/api/dashboard/stats")
def dashboard_stats(http_request: Request) -> dict[str, Any]:
    resident_id = require_resident(http_request)
    from backend import voice_persistence
    stats = get_resident_stats(resident_id)
    # Fold concluded voice runs into the headline figures so the dashboard reflects
    # both training modes. (Detailed per-criterion review lives on each run.)
    voice = voice_persistence.get_resident_evaluations(resident_id)
    if voice:
        v_scores = [v["overall_score"] for v in voice if v.get("overall_score") is not None]
        prior_n = stats.get("cases_completed") or 0
        prior_avg = stats.get("avg_score")
        total_n = prior_n + len(v_scores)
        if v_scores and total_n:
            blended = ((prior_avg or 0) * prior_n + sum(v_scores)) / total_n
            stats["avg_score"] = round(blended, 2)
            stats["cases_completed"] = total_n
    return stats


# Per-resident coaching-insight cache: resident_id -> (evaluation signature, insight).
# The insight is a pure function of the recent evaluation set, so we only pay the
# LLM call when that set actually changes (a new case completes), not on every
# analytics page view.
_COACHING_CACHE: dict[str, tuple[str, str]] = {}


@app.get("/api/dashboard/coaching-insight")
async def dashboard_coaching_insight(http_request: Request) -> dict[str, Any]:
    import asyncio
    import hashlib

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

    # The insight depends only on this evaluation set — serve the cached one
    # unless the set changed since we last generated it.
    signature = hashlib.sha256("\n\n".join(summaries).encode()).hexdigest()
    cached = _COACHING_CACHE.get(resident_id)
    if cached and cached[0] == signature:
        return {"insight": cached[1], "has_data": True}

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
        insight = insight.strip()
        _COACHING_CACHE[resident_id] = (signature, insight)
        return {"insight": insight, "has_data": True}
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


@app.get("/api/dashboard/voice-review/{session_id}")
def dashboard_voice_review(session_id: str, http_request: Request) -> dict[str, Any]:
    """The stored assessment for one concluded voice run (analytics review drawer)."""
    resident_id = require_resident(http_request)
    from backend import voice_persistence
    review = voice_persistence.get_evaluation(session_id, resident_id)
    if review is None:
        raise HTTPException(status_code=404, detail="voice run not found")
    return review


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
def pause(session_id: str, body: PauseRequest, http_request: Request) -> dict[str, bool]:
    _get_owned_session(session_id, http_request)  # 404/403 if not the caller's
    graph = get_graph()
    snapshot = graph.get_context(session_id) if graph and graph.enabled else None
    pause_session(session_id, snapshot, body.elapsed_seconds)
    log_event(session_id, "session.paused", {"elapsed_seconds": body.elapsed_seconds})
    return {"ok": True}


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, http_request: Request) -> dict[str, Any]:
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    from backend.db import get_db as _get_db
    conn = _get_db()
    row = conn.execute(
        "SELECT status, duration_seconds, resident_id FROM web_sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    # Only the owning resident may rehydrate a run (anonymous runs stay open).
    owner = row["resident_id"] if row else None
    caller = get_current_resident_id(http_request)
    if owner is not None and caller is not None and owner != caller:
        raise HTTPException(status_code=403, detail="access denied")

    SESSIONS[session_id] = session
    log_event(session_id, "session.resumed")
    status = row["status"] if row else "active"
    duration_seconds = row["duration_seconds"] if row else 0
    return _full_state_response(session, session_id, status, duration_seconds=duration_seconds or 0)


@app.post("/api/cases")
@_limiter.limit("10/minute")
def generate_case(body: GenerateCaseRequest, request: Request) -> dict[str, Any]:
    resident_id = get_current_resident_id(request)
    clinical_case = get_or_generate_cached_case(body.diagnosis, body.difficulty)
    session_id = str(uuid4())
    SESSIONS[session_id] = WebSession(case=clinical_case, resident_id=resident_id)
    create_session(session_id, body.diagnosis, body.difficulty, clinical_case, resident_id=resident_id)
    log_event(session_id, "session.created", {"diagnosis": body.diagnosis, "difficulty": body.difficulty}, resident_id=resident_id)
    graph = get_graph()
    if graph and graph.enabled:
        graph.init_session(session_id, clinical_case)
    return {
        "session_id": session_id,
        "case_preview": case_preview(clinical_case),
    }


@app.post("/api/questions")
@_limiter.limit("60/minute")
def ask_question(body: QuestionRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    history = list(zip(session.student_questions, session.patient_responses))
    answer = get_patient_service().answer_question(
        session.case, body.question,
        conversation_history=history,
        revealed_symptoms=session.revealed_symptoms,
    )
    session.student_questions.append(body.question)
    session.patient_responses.append(answer)
    for symptom in session.case.symptoms:
        if symptom not in session.revealed_symptoms and symptom.lower() in answer.lower():
            session.revealed_symptoms.append(symptom)
    graph = get_graph()
    if graph and graph.enabled:
        graph.add_turn(body.session_id, body.question, answer)
    save_session(body.session_id, session)
    log_event(body.session_id, "question.asked", {
        "question": body.question,
        "turn_index": len(session.student_questions) - 1,
    })
    return {
        "question": body.question,
        "answer": answer,
    }


@app.post("/api/investigations")
@_limiter.limit("30/minute")
def order_investigation(body: InvestigationRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    results = get_investigation_service().get_results(session.case, body.investigation)
    session.investigations.extend(results)
    graph = get_graph()
    if graph and graph.enabled:
        for result in results:
            graph.add_investigation(body.session_id, result)
    save_session(body.session_id, session)
    for result in results:
        log_event(body.session_id, "investigation.ordered", {
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


@app.post("/api/investigations/ecg-rerender")
@_limiter.limit("60/minute")
def rerender_ecg(body: ECGRerenderRequest, request: Request) -> dict[str, Any]:
    """Re-render the most recent ECG investigation at a chosen sample rate.

    Powers the 100 Hz / 500 Hz toggle in the ECG panel. The picked PTB-XL
    record is unchanged — only the source signal's sampling rate. The session's
    stored investigation result is updated so the new SVG persists across
    reloads.
    """
    session = _get_owned_session(body.session_id, request)
    if body.sample_rate_hz not in (100, 500):
        raise HTTPException(status_code=400, detail="sample_rate_hz must be 100 or 500")

    # Find the most recent ECG investigation with a PTB-XL source_record.
    ecg_idx = next(
        (i for i in range(len(session.investigations) - 1, -1, -1)
         if "ecg" in (session.investigations[i].investigation or "").lower()
         and session.investigations[i].visual_payload is not None
         and getattr(session.investigations[i].visual_payload, "source_record", "")),
        None,
    )
    if ecg_idx is None:
        raise HTTPException(status_code=404, detail="no ECG investigation to re-render")

    inv = session.investigations[ecg_idx]
    source_record = inv.visual_payload.source_record  # type: ignore[union-attr]

    from backend.ptbxl_service import get_ptbxl_service
    svg = get_ptbxl_service().rerender_record(source_record, sample_rate_hz=body.sample_rate_hz)
    if svg is None:
        raise HTTPException(status_code=500, detail="re-render failed")

    # Mutate the visual_payload in place and persist.
    inv.visual_payload.svg = svg  # type: ignore[union-attr]
    save_session(body.session_id, session)
    log_event(body.session_id, "ecg.rerendered", {
        "sample_rate_hz": body.sample_rate_hz,
        "source_record": source_record,
    })
    return {"svg": svg, "sample_rate_hz": body.sample_rate_hz, "source_record": source_record}


@app.post("/api/interpretations")
@_limiter.limit("30/minute")
def submit_interpretation(body: InterpretationRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    session.interpretations.append(body.interpretation)
    save_session(body.session_id, session)
    log_event(body.session_id, "interpretation.submitted", {"interpretation": body.interpretation})
    return {
        "interpretations": session.interpretations,
    }


@app.post("/api/diagnosis")
@_limiter.limit("10/minute")
def submit_diagnosis(body: DiagnosisRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    session.diagnosis_guess = body.diagnosis
    save_session(body.session_id, session)
    log_event(body.session_id, "diagnosis.submitted", {"guess": body.diagnosis})
    return {
        "diagnosis_guess": body.diagnosis,
    }


@app.post("/api/doctor-consult")
@_limiter.limit("10/minute")
def call_doctor(body: DoctorConsultRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    graph = get_graph()
    graph_context = graph.get_context(body.session_id) if graph and graph.enabled else ""
    response = get_doctor_consult_service().consult(
        session.case,
        session.student_questions,
        session.patient_responses,
        session.investigations,
        body.question,
        session_context={"clinical_picture": graph_context} if graph_context else None,
    )
    session.doctor_consults.append(response)
    save_session(body.session_id, session)
    log_event(body.session_id, "consult.called", {"question": body.question})
    return response.model_dump()


@app.post("/api/evaluate")
@_limiter.limit("5/minute")
def evaluate(body: EvaluateRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    questions = list(session.student_questions)
    responses = list(session.patient_responses)
    if session.diagnosis_guess:
        questions.append(f"Diagnosis guess: {session.diagnosis_guess}")
        responses.append("I'm not sure.")

    graph = get_graph()
    graph_entity_names = graph.get_entity_names(body.session_id) if graph and graph.enabled else []

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
    save_evaluation(body.session_id, evaluation, elapsed_seconds=body.elapsed_seconds)
    log_event(body.session_id, "session.evaluated", {"overall_score": getattr(evaluation, "overall_score", None), "elapsed_seconds": body.elapsed_seconds})
    return evaluation.model_dump()


@app.post("/api/physical-exam")
@_limiter.limit("20/minute")
def physical_exam(body: PhysicalExamRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    already_examined = [e.system for e in session.physical_exams]
    new_systems = [s for s in body.systems if s not in already_examined]
    if not new_systems:
        return {"results": [e.model_dump() for e in session.physical_exams]}

    results = get_physical_exam_agent().examine(session.case, new_systems, already_examined)
    session.physical_exams.extend(results)

    graph = get_graph()
    if graph and graph.enabled:
        for finding in results:
            try:
                graph.add_examination(body.session_id, finding)
            except Exception:
                pass

    save_session(body.session_id, session)
    log_event(body.session_id, "exam.performed", {"systems": new_systems})
    return {"results": [e.model_dump() for e in results]}


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
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack.min.css"/>
  <script src="https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack-all.js"></script>
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
    .cb-topbar-btn-submit{background:var(--teal-600);color:#fff;border:0;font:inherit;font-size:11px;font-weight:600;padding:5px 10px;border-radius:5px;cursor:pointer;transition:background .15s;white-space:nowrap;display:inline-flex;align-items:center;gap:5px;line-height:1;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.12)}
    .cb-topbar-btn-submit:hover{background:var(--teal-500)}
    .cb-topbar-btn-save-exit{background:linear-gradient(135deg,rgba(46,194,126,0.2),rgba(70,166,167,0.2));color:#3dd68c;border:1px solid rgba(46,194,126,0.4);font:inherit;font-size:11px;font-weight:600;padding:5px 10px;border-radius:5px;cursor:pointer;transition:background .15s,color .15s,border-color .15s;white-space:nowrap;display:inline-flex;align-items:center;gap:5px;line-height:1}
    .cb-topbar-btn-save-exit:hover{background:linear-gradient(135deg,rgba(46,194,126,0.32),rgba(70,166,167,0.32));color:#6aeaa6;border-color:rgba(46,194,126,0.6)}
    /* Submit confirmation modal */
    .cb-submit-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:500;backdrop-filter:blur(6px);align-items:center;justify-content:center;animation:kgFadeIn .18s ease}
    .cb-submit-backdrop.on{display:flex}
    .cb-submit-card{width:420px;max-width:calc(100vw - 32px);background:linear-gradient(180deg,#121a28 0%,#0d1420 100%);color:#e9edf2;border:1px solid rgba(255,255,255,0.08);border-radius:16px;box-shadow:0 30px 80px rgba(8,12,20,0.6);padding:32px 28px;animation:cbPauseRise .3s cubic-bezier(.2,.7,.1,1) both}
    .cb-submit-icon{width:52px;height:52px;border-radius:14px;background:rgba(70,166,167,0.12);border:1px solid rgba(70,166,167,0.25);display:grid;place-items:center;margin:0 auto 18px;color:var(--teal-400)}
    .cb-submit-title{font-size:20px;font-weight:700;text-align:center;margin-bottom:8px;letter-spacing:-.02em}
    .cb-submit-body{font-size:13px;color:var(--ink-300);text-align:center;line-height:1.55;margin-bottom:24px}
    .cb-submit-actions{display:flex;gap:10px}
    .cb-submit-cancel{flex:1;background:var(--ink-700);color:var(--ink-200);border:1px solid var(--ink-600);font:inherit;font-size:13px;font-weight:500;padding:11px;border-radius:9px;cursor:pointer;transition:background .15s}
    .cb-submit-cancel:hover{background:var(--ink-600);color:white}
    .cb-submit-confirm{flex:1;background:linear-gradient(135deg,var(--teal-600),var(--teal-500));color:white;border:0;font:inherit;font-size:13px;font-weight:700;padding:11px;border-radius:9px;cursor:pointer;transition:opacity .15s;display:flex;align-items:center;justify-content:center;gap:7px}
    .cb-submit-confirm:hover{opacity:0.9}
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
    /* ── Section label (still used in performance panel summary) ── */
    .cb-ps-label{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-400);font-weight:600;margin-bottom:8px}
    /* ── Imaging panel — modality launch cards ── */
    #panel-imaging .wb-fp-body{padding:14px 14px 16px;display:flex;flex-direction:column;gap:9px;background:var(--paper,#fbfcfd)}
    .img-card{display:grid;grid-template-columns:42px 1fr auto;align-items:center;gap:12px;padding:11px 13px;border:1.5px solid var(--ink-200);border-radius:10px;background:white;cursor:pointer;transition:border-color .15s ease,background .15s ease,transform .12s ease;text-align:left;font:inherit;color:inherit}
    .img-card:hover{background:var(--ink-50)}
    .img-card[data-state="idle"]:hover{border-color:var(--teal-500);transform:translateY(-1px)}
    .img-card[data-state="pending"]{border-color:var(--teal-500);background:var(--teal-50);cursor:wait;animation:imgPulse 1.4s ease-in-out infinite}
    .img-card[data-state="available"]{border-color:#2ec27e;background:#f0fbf4}
    .img-card[data-state="available"]:hover{border-color:#1f9a60;transform:translateY(-1px)}
    .img-card[data-state="disabled"]{border-style:dashed;border-color:var(--ink-100);background:var(--ink-50);cursor:not-allowed;opacity:0.65}
    .img-card[data-state="disabled"]:hover{background:var(--ink-50);transform:none}
    @keyframes imgPulse{0%,100%{box-shadow:0 0 0 0 rgba(70,166,167,0.18)}50%{box-shadow:0 0 0 4px rgba(70,166,167,0.10)}}
    .img-card-icon{width:42px;height:42px;border-radius:9px;background:var(--ink-50);display:grid;place-items:center;color:var(--ink-500);flex-shrink:0;border:1px solid var(--ink-100)}
    .img-card[data-state="pending"] .img-card-icon{background:var(--teal-50);color:var(--teal-600);border-color:color-mix(in oklch,var(--teal-600) 22%,white)}
    .img-card[data-state="available"] .img-card-icon{background:#e6f8ee;color:#1f7a55;border-color:#bce7d0}
    .img-card[data-state="disabled"] .img-card-icon{color:var(--ink-400)}
    .img-card-text{display:flex;flex-direction:column;gap:2px;min-width:0}
    .img-card-name{font-size:13.5px;font-weight:600;color:var(--ink-900);letter-spacing:-0.01em}
    .img-card[data-state="disabled"] .img-card-name{color:var(--ink-500)}
    .img-card-sub{font-size:11px;color:var(--ink-500);font-family:'JetBrains Mono',monospace;letter-spacing:0.02em}
    .img-card[data-state="pending"] .img-card-sub{color:var(--teal-600)}
    .img-card[data-state="available"] .img-card-sub{color:#1f7a55}
    .img-card-cta{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--ink-400);font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;text-transform:uppercase}
    .img-card[data-state="idle"] .img-card-cta{color:var(--ink-500)}
    .img-card[data-state="idle"]:hover .img-card-cta{color:var(--teal-600)}
    .img-card[data-state="pending"] .img-card-cta{color:var(--teal-600)}
    .img-card[data-state="available"] .img-card-cta{color:#1f7a55}
    .img-card[data-state="disabled"] .img-card-cta{color:var(--ink-300)}
    .img-card-spin{display:none;width:14px;height:14px;border:2px solid color-mix(in oklch,var(--teal-600) 30%,white);border-top-color:var(--teal-600);border-radius:50%;animation:imgSpin .8s linear infinite}
    .img-card[data-state="pending"] .img-card-spin{display:inline-block}
    @keyframes imgSpin{to{transform:rotate(360deg)}}
    .img-card-check{display:none;width:16px;height:16px;color:#1f7a55}
    .img-card[data-state="available"] .img-card-check{display:inline-block}
    /* ── Consults panel — department call cards ── */
    #panel-consults .wb-fp-body{padding:14px 14px 16px;display:flex;flex-direction:column;gap:9px;background:var(--paper,#fbfcfd)}
    .cons-card{display:grid;grid-template-columns:42px 1fr auto;align-items:center;gap:12px;padding:11px 13px;border:1.5px solid var(--ink-200);border-radius:10px;background:white;cursor:not-allowed;transition:border-color .15s ease,background .15s ease,transform .12s ease;text-align:left;font:inherit;color:inherit;border-style:dashed;opacity:0.85}
    .cons-card[data-state="idle"]{cursor:pointer;border-style:solid;opacity:1}
    .cons-card[data-state="idle"]:hover{border-color:var(--teal-500);background:var(--ink-50);transform:translateY(-1px)}
    .cons-card-icon{width:42px;height:42px;border-radius:9px;display:grid;place-items:center;color:white;flex-shrink:0;font-weight:600;font-size:13px;letter-spacing:0.04em;font-family:'JetBrains Mono',monospace}
    .cons-card[data-dept="neurology"] .cons-card-icon{background:linear-gradient(140deg,#6366f1,#4338ca)}
    .cons-card[data-dept="nephrology"] .cons-card-icon{background:linear-gradient(140deg,#10b981,#047857)}
    .cons-card[data-dept="icu"]        .cons-card-icon{background:linear-gradient(140deg,#ef4444,#b91c1c)}
    .cons-card[data-state="disabled"] .cons-card-icon{filter:grayscale(0.6);opacity:0.75}
    .cons-card-text{display:flex;flex-direction:column;gap:2px;min-width:0}
    .cons-card-name{font-size:13.5px;font-weight:600;color:var(--ink-900);letter-spacing:-0.01em}
    .cons-card[data-state="disabled"] .cons-card-name{color:var(--ink-600)}
    .cons-card-sub{font-size:11px;color:var(--ink-500);font-family:'JetBrains Mono',monospace;letter-spacing:0.02em}
    .cons-card-cta{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--ink-400);font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;text-transform:uppercase}
    .cons-card[data-state="idle"]:hover .cons-card-cta{color:var(--teal-600)}
    .cons-card-cta svg{width:13px;height:13px}
    /* ── Patient panel (vertical, lives in the GridStack workbench) ── */
    #panel-patient .wb-fp-body{padding:0;display:flex;flex-direction:column;overflow:hidden;background:var(--paper,#fbfcfd)}
    .pp-scroll{flex:1;min-height:0;overflow-y:auto;display:flex;flex-direction:column}
    .pp-head{padding:16px 16px 14px;background:linear-gradient(180deg,#ffffff,#fbfcfd);border-bottom:1px solid var(--ink-100);display:flex;flex-direction:column;gap:12px}
    .pp-id{display:flex;align-items:center;gap:12px}
    .pp-avatar{width:46px;height:46px;border-radius:50%;flex-shrink:0;background:linear-gradient(140deg,#9a8468,#6f5a40);color:#fff;font-weight:600;font-size:16px;display:grid;place-items:center;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.18),0 2px 6px rgba(111,90,64,0.3);letter-spacing:0.02em}
    .pp-id-text{min-width:0;flex:1}
    .pp-name{font-size:16.5px;font-weight:600;letter-spacing:-0.01em;line-height:1.2;color:var(--ink-900)}
    .pp-name .pp-desc{color:var(--ink-500);font-weight:400;font-size:12.5px}
    .pp-live{display:inline-flex;align-items:center;gap:6px;margin-top:3px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#1f7a55}
    .pp-live .pp-live-dot{width:7px;height:7px;border-radius:50%;background:#1f7a55;animation:ppBlink 1.6s infinite}
    @keyframes ppBlink{0%,100%{opacity:1}50%{opacity:0.25}}
    .pp-persona{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-size:14.5px;line-height:1.4;color:var(--ink-600);padding:10px 12px;border-radius:9px;background:var(--ink-50);border:1px solid var(--ink-100)}
    .pp-persona:empty{display:none}
    .pp-chips{display:flex;flex-wrap:wrap;gap:6px}
    .pp-chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--ink-600);background:white;border:1px solid var(--ink-100);border-radius:7px;padding:4px 9px}
    .pp-chip b{color:var(--ink-900);font-weight:600}
    .pp-chip.role{background:var(--teal-50);border-color:color-mix(in oklch,var(--teal-600) 22%,white);color:var(--teal-700);font-weight:600}
    .pp-cc{padding:14px 16px;border-bottom:1px solid var(--ink-100)}
    .pp-sec-label{font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:var(--ink-400);font-weight:700;margin-bottom:8px;display:flex;align-items:center;gap:8px}
    .pp-sec-label::after{content:'';flex:1;height:1px;background:var(--ink-100)}
    .pp-cc-quote{font-family:'Instrument Serif',Georgia,serif;font-size:24px;line-height:1.2;color:var(--ink-900)}
    .pp-cc-quote em{font-style:italic}
    .pp-vitals{padding:14px 16px}
    .pp-vitals-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .pp-vital{position:relative;overflow:hidden;border:1px solid var(--ink-100);border-radius:10px;background:white;padding:10px 11px 9px 13px}
    .pp-vital::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:#1f7a55}
    .pp-vital.warn{border-color:#f0d49a;background:#fdf3e2}
    .pp-vital.warn::before{background:#b76e00}
    .pp-v-top{display:flex;align-items:center;gap:6px;margin-bottom:4px}
    .pp-v-label{font-size:10.5px;font-weight:700;letter-spacing:0.06em;color:var(--ink-500);text-transform:uppercase}
    .pp-vital.warn .pp-v-label{color:#b76e00}
    .pp-v-label sub{font-size:8px}
    .pp-v-arrow{margin-left:auto;width:17px;height:17px;border-radius:5px;background:#b76e00;color:#fff;display:grid;place-items:center}
    .pp-v-val{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:22px;line-height:1.05;color:var(--ink-900);display:flex;align-items:baseline;gap:4px}
    .pp-v-val .pp-v-unit{font-size:10.5px;color:var(--ink-400);font-weight:500}
    .pp-v-flag{font-size:10.5px;font-style:italic;color:#b76e00;margin-top:4px;font-weight:500}
    .pp-v-ok{font-size:10.5px;color:var(--ink-400);margin-top:4px}
    /* Lead II ECG monitor — pinned at the bottom of the panel */
    .pp-ecg{background:#07090f;position:relative;border-top:1px solid #000;flex-shrink:0}
    .pp-ecg-bar{display:flex;align-items:center;gap:10px;padding:9px 14px 5px}
    .pp-ecg-lead{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:12.5px;color:#34e7b6;letter-spacing:0.04em;border:1px solid color-mix(in oklch,#34e7b6 40%,transparent);border-radius:6px;padding:2px 7px;background:color-mix(in oklch,#34e7b6 10%,transparent)}
    .pp-ecg-cap{font-size:10.5px;color:#56627a;font-family:'JetBrains Mono',monospace;letter-spacing:0.04em}
    .pp-ecg-hr{margin-left:auto;display:flex;align-items:center;gap:7px}
    .pp-ecg-hr .pp-heart{color:#34e7b6;display:grid;place-items:center;transition:transform .08s ease}
    .pp-ecg-hr .pp-heart.beat{transform:scale(1.35)}
    .pp-ecg-hr .pp-hr-num{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:20px;color:#34e7b6;line-height:1}
    .pp-ecg-hr .pp-hr-unit{font-family:'JetBrains Mono',monospace;font-size:10px;color:#56627a;align-self:flex-end;margin-bottom:2px}
    .pp-ecg-canvas-wrap{position:relative;height:108px}
    #ppEcg{display:block;width:100%;height:100%}
    .pp-ecg-foot{display:flex;align-items:center;justify-content:space-between;padding:5px 14px 10px;font-family:'JetBrains Mono',monospace;font-size:9.5px;color:#3f495e;letter-spacing:0.05em}
    .pp-ecg-foot .pp-gain{color:#56627a}
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
    /* Sample-rate toggle in the ECG panel header */
    .cb-rate-toggle{display:inline-flex;border:1px solid var(--ink-200);border-radius:6px;overflow:hidden;background:var(--ink-50);margin-right:4px}
    .cb-rate-btn{font:inherit;font-size:10.5px;font-weight:600;padding:4px 8px;border:0;background:transparent;color:var(--ink-500);cursor:pointer;letter-spacing:0.02em;transition:background .15s,color .15s}
    .cb-rate-btn:hover{color:var(--ink-700)}
    .cb-rate-btn.active{background:white;color:var(--teal-700);box-shadow:inset 0 0 0 1px var(--teal-500)}
    .cb-rate-btn:disabled{opacity:0.55;cursor:wait}
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
    #panel-kg.kg-expanded{position:fixed;inset:40px;z-index:200;border-radius:12px;box-shadow:0 24px 80px rgba(0,0,0,0.45);display:flex;flex-direction:column;transition:all .28s cubic-bezier(.4,0,.2,1)}
    #panel-kg.kg-expanded .cb-panel-body-wrap{flex:1;min-height:0;display:flex;flex-direction:column}
    #panel-kg.kg-expanded .cb-kg-stage{flex:1;min-height:0;height:auto}
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
    /* ── New Physical Examination panel (design refresh, 2026-06-09) ── */
    .pe-body{padding:14px 16px 4px}
    .pe-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:11px;gap:12px}
    .pe-label{font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-500);font-weight:700}
    .pe-selectbar{display:flex;align-items:center;gap:10px}
    .pe-count-pill{font-family:'JetBrains Mono',monospace;font-size:11.5px;font-weight:500;color:var(--ink-600);background:var(--ink-50);border:1px solid var(--ink-100);border-radius:99px;padding:3px 10px}
    .pe-count-pill b{color:var(--ink-900);font-weight:600}
    .pe-link{background:none;border:0;cursor:pointer;font:inherit;font-size:12px;font-weight:600;color:var(--teal-700);padding:3px 4px;border-radius:6px}
    .pe-link:hover{color:var(--teal-600);text-decoration:underline}
    .pe-link:disabled{color:var(--ink-300);cursor:default;text-decoration:none}
    .pe-systems{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .pe-sys{--c:var(--ink-500);display:flex;align-items:center;gap:11px;padding:11px 13px;border:1px solid var(--ink-100);border-radius:11px;background:white;cursor:pointer;user-select:none;position:relative;transition:border-color .14s,background .14s,box-shadow .14s,transform .06s}
    .pe-sys::before{content:'';position:absolute;left:0;top:9px;bottom:9px;width:3px;border-radius:0 3px 3px 0;background:var(--c);opacity:0;transition:opacity .14s}
    .pe-sys:hover{border-color:var(--ink-200);background:var(--paper)}
    .pe-sys:active{transform:translateY(0.5px)}
    .pe-sys.on{border-color:color-mix(in oklch, var(--c) 55%, var(--ink-200));background:color-mix(in oklch, var(--c) 7%, white);box-shadow:0 1px 2px rgba(15,23,42,0.04)}
    .pe-sys.on::before{opacity:1}
    .pe-sys.done{cursor:default;opacity:0.75}
    .pe-sys.done:hover{background:color-mix(in oklch, var(--c) 5%, white)}
    .pe-sys .dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;background:var(--c);box-shadow:0 0 0 3px color-mix(in oklch, var(--c) 16%, transparent)}
    .pe-sys .name{font-size:14px;font-weight:500;color:var(--ink-800);flex:1;line-height:1.2;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .pe-sys.on .name{color:var(--ink-900);font-weight:600}
    .pe-sys .check{width:20px;height:20px;border-radius:6px;flex-shrink:0;border:1.5px solid var(--ink-200);background:white;display:grid;place-items:center;color:white;transition:background .14s,border-color .14s}
    .pe-sys .check svg{opacity:0;transform:scale(0.6);transition:opacity .14s,transform .14s}
    .pe-sys.on .check{background:var(--c);border-color:var(--c)}
    .pe-sys.on .check svg,.pe-sys.done .check svg{opacity:1;transform:scale(1)}
    .pe-sys.done .check{background:var(--c);border-color:var(--c)}
    .pe-foot{padding:12px 16px 14px;display:flex;align-items:center;gap:14px;border-top:1px solid var(--ink-50)}
    .pe-meta{font-size:12.5px;color:var(--ink-500);line-height:1.3;flex:1;min-width:0}
    .pe-meta b{color:var(--ink-900);font-weight:600}
    .pe-examine{margin-left:auto;display:inline-flex;align-items:center;gap:9px;background:linear-gradient(180deg,var(--ink-800),var(--ink-950));color:white;border:0;border-radius:11px;padding:12px 20px;font:inherit;font-size:13.5px;font-weight:600;cursor:pointer;box-shadow:0 6px 16px rgba(10,15,23,0.22);transition:filter .14s,transform .06s,opacity .14s;flex-shrink:0}
    .pe-examine:hover{filter:brightness(1.12)}
    .pe-examine:active{transform:translateY(1px)}
    .pe-examine .badge{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;background:rgba(255,255,255,0.16);color:white;border-radius:99px;padding:1px 8px;min-width:22px;text-align:center}
    .pe-examine:disabled{background:var(--ink-100);color:var(--ink-400);box-shadow:none;cursor:default;filter:none}
    .pe-examine:disabled .badge{background:rgba(0,0,0,0.06);color:var(--ink-400)}
    @media (max-width:560px){.pe-systems{grid-template-columns:1fr}.pe-foot{flex-wrap:wrap}.pe-examine{margin-left:0;width:100%;justify-content:center}}
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
    /* ── Inline rail tabs that live in the status bar (Performance + KG) ── */
    .cb-sb-tabs{display:flex;align-items:center;gap:6px;margin-left:auto}
    .cb-sb-tab{display:inline-flex;align-items:center;gap:6px;background:transparent;border:1px solid var(--ink-700);border-radius:6px;padding:3px 9px;color:var(--ink-300);cursor:pointer;font:inherit;font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:0.04em;line-height:1;text-transform:uppercase;transition:background .15s ease,color .15s ease,border-color .15s ease;position:relative}
    .cb-sb-tab:hover{background:rgba(255,255,255,0.04);color:var(--ink-100);border-color:var(--ink-600)}
    .cb-sb-tab.active{background:rgba(70,166,167,0.16);border-color:var(--teal-500);color:var(--teal-500)}
    .cb-sb-tab svg{display:block;flex-shrink:0}
    .cb-sb-tab .cb-sb-tab-meta{font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--ink-500);font-weight:500;letter-spacing:0.02em}
    .cb-sb-tab.active .cb-sb-tab-meta{color:var(--teal-500)}
    .cb-sb-tab.has-update::after{content:'';position:absolute;top:-2px;right:-2px;width:6px;height:6px;background:#2ec27e;border-radius:50%;box-shadow:0 0 0 2px var(--ink-900)}
    .cb-sb-tab.has-alert::after{content:'';position:absolute;top:-2px;right:-2px;width:6px;height:6px;background:var(--critical);border-radius:50%;box-shadow:0 0 0 2px var(--ink-900);animation:wbBlinkAlert 1.6s ease-in-out infinite}
    .cb-sb-brand{color:var(--ink-500)}
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

    /* ================================================================
       WORKBENCH v3 — rails + docks + center stage
       ================================================================ */
    .wb-workbench{display:flex;align-items:stretch;background:radial-gradient(ellipse at 50% 0%,rgba(70,166,167,0.05),transparent 50%),linear-gradient(180deg,var(--ink-50) 0%,#eef1f6 100%);flex:1;min-height:520px;position:relative;overflow:hidden}
    /* Rails */
    .wb-rail{width:84px;flex-shrink:0;background:linear-gradient(180deg,var(--ink-950),var(--ink-900));border-right:1px solid #000;display:flex;flex-direction:column;padding:12px 8px;gap:4px;position:relative;z-index:2}
    .wb-rail.wb-rail-right{border-right:0;border-left:1px solid #000}
    .wb-rail-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--ink-500);font-weight:600;text-align:center;padding:6px 0 4px;border-bottom:1px dashed var(--ink-700);margin-bottom:6px}
    .wb-rail-tab{background:transparent;border:1px solid transparent;border-radius:8px;color:var(--ink-400);cursor:pointer;padding:10px 6px;display:flex;flex-direction:column;align-items:center;gap:5px;font:inherit;font-size:10px;line-height:1.2;text-align:center;position:relative;transition:background .15s ease,color .15s ease}
    .wb-rail-tab:hover{background:rgba(255,255,255,0.04);color:var(--ink-200)}
    .wb-rail-tab.active{background:rgba(70,166,167,0.10);border-color:rgba(70,166,167,0.35);color:var(--teal-500)}
    .wb-rail-tab svg{display:block}
    .wb-rt-label{font-weight:600;font-size:9px;letter-spacing:0em}
    .wb-rt-meta{font-family:'JetBrains Mono',monospace;font-size:8.5px;color:var(--ink-500);font-weight:500;margin-top:1px;white-space:nowrap}
    .wb-rail-tab.active .wb-rt-meta{color:var(--teal-500)}
    .wb-rail-tab.has-update::after{content:'';position:absolute;top:6px;right:8px;width:6px;height:6px;background:#2ec27e;border-radius:50%;box-shadow:0 0 0 2px rgba(46,194,126,0.25)}
    .wb-rail-tab.has-alert::after{content:'';position:absolute;top:6px;right:8px;width:6px;height:6px;background:var(--critical);border-radius:50%;box-shadow:0 0 0 2px rgba(196,58,44,0.25);animation:wbBlinkAlert 1.6s ease-in-out infinite}
    @keyframes wbBlinkAlert{0%,100%{opacity:1}50%{opacity:0.3}}
    .wb-rail-tab.active::before{content:'';position:absolute;top:50%;transform:translateY(-50%);width:3px;height:24px;background:var(--teal-500);border-radius:0 3px 3px 0}
    .wb-rail.wb-rail-left .wb-rail-tab.active::before{left:-8px}
    .wb-rail.wb-rail-right .wb-rail-tab.active::before{right:-8px}
    .wb-rail-spacer{flex:1}
    .wb-rail-foot{padding:6px 0 0;border-top:1px dashed var(--ink-700);font-family:'JetBrains Mono',monospace;font-size:8.5px;color:var(--ink-500);text-align:center;line-height:1.4}
    /* ── Grid area (fills space between rails, scrollable) ── */
    #gs-area{flex:1;min-width:0;overflow:auto;position:relative;background:radial-gradient(ellipse at 50% 0%,rgba(70,166,167,0.07),transparent 60%),linear-gradient(180deg,var(--ink-50) 0%,#eef1f6 100%)}
    #gs-area .grid-stack{min-height:100%;padding:12px;box-sizing:border-box;background:transparent}
    /* Override GridStack default colors so it matches our theme */
    .grid-stack>.grid-stack-item>.grid-stack-item-content{border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(15,23,42,0.06),0 8px 24px rgba(15,23,42,0.10);inset:4px}
    .grid-stack-placeholder>.placeholder-content{border:2px dashed var(--teal-500);border-radius:10px;background:rgba(70,166,167,0.06)}
    /* Float panels — fill their grid cell */
    .wb-float-panel{display:flex;flex-direction:column;width:100%;height:100%;background:white;border:1px solid var(--ink-100);border-radius:10px;overflow:hidden;position:relative}
    @keyframes wbPanelIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
    .grid-stack-item-content .wb-float-panel{animation:wbPanelIn .2s cubic-bezier(.2,.7,.1,1) both}
    /* Drag handle = panel header */
    .wb-fp-head{cursor:move;user-select:none}
    .wb-fp-head .wb-fp-close{cursor:pointer}
    .wb-fp-close,.wb-fp-actions button{cursor:pointer!important}
    .wb-fp-head{display:flex;align-items:center;gap:8px;padding:9px 10px 9px 12px;background:linear-gradient(180deg,white,#fafbfc);border-bottom:1px solid var(--ink-100);flex-shrink:0}
    .wb-fp-title-group{display:flex;align-items:baseline;gap:8px;min-width:0}
    .wb-fp-title{font-size:12px;font-weight:600;color:var(--ink-900);letter-spacing:-0.005em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .wb-fp-sub{font-size:10px;color:var(--ink-400);font-family:'JetBrains Mono',monospace;white-space:nowrap}
    .wb-fp-actions{margin-left:auto;display:flex;gap:2px;align-items:center}
    .wb-fp-close{width:22px;height:22px;border-radius:5px;background:transparent;border:0;color:var(--ink-500);cursor:pointer;display:grid;place-items:center}
    .wb-fp-close:hover{background:var(--critical-bg);color:var(--critical)}
    /* Pin button */
    .wb-pin-btn{width:22px;height:22px;border-radius:5px;background:transparent;border:0;color:var(--ink-400);cursor:pointer;display:grid;place-items:center;transition:color .15s,background .15s;flex-shrink:0}
    .wb-pin-btn:hover{background:rgba(70,166,167,0.12);color:var(--teal-600)}
    .wb-pin-btn.pinned{color:var(--teal-500)}
    .wb-pin-btn.pinned svg{fill:var(--teal-500);stroke:var(--teal-600)}
    /* Pinned panel: teal top accent bar */
    .wb-float-panel.pinned{border-top:2.5px solid var(--teal-500)}
    .wb-float-panel.pinned .wb-fp-head{background:linear-gradient(180deg,rgba(70,166,167,0.06),rgba(70,166,167,0.02))}
    /* Rail tab pinned indicator: small teal dot bottom-right */
    .wb-tab-pin-dot{display:none;position:absolute;top:4px;left:4px;width:16px;height:16px;border-radius:4px;background:rgba(0,0,0,0.28);border:1px solid rgba(255,255,255,0.06);cursor:pointer;place-items:center;color:var(--ink-400);transition:background .15s,color .15s}
    .wb-rail-tab:hover .wb-tab-pin-dot{display:grid}
    .wb-rail-tab.active .wb-tab-pin-dot{display:grid}
    .wb-tab-pin-dot:hover{background:rgba(59,130,246,0.18);color:#60a5fa;border-color:rgba(59,130,246,0.35)}
    .wb-tab-pin-dot.pinned{display:grid;background:rgba(59,130,246,0.22);color:#3b82f6;border-color:rgba(59,130,246,0.5)}
    .wb-tab-pin-dot.pinned svg{fill:#3b82f6;stroke:#2563eb}
    .wb-fp-body{flex:1;overflow:auto;background:white}
    .wb-fp-body::-webkit-scrollbar{width:8px}
    .wb-fp-body::-webkit-scrollbar-thumb{background:var(--ink-200);border-radius:8px}
    /* KG fullscreen expand — handled by the original rule above */
    /* Center stage CSS removed — GridStack handles layout */
    .wb-stage-hero{background:linear-gradient(180deg,white 0%,#fafcfd 100%);border:1px solid var(--ink-100);border-radius:14px;padding:18px 20px 16px;box-shadow:var(--shadow-sm);position:relative;overflow:hidden}
    .wb-stage-hero::before{content:'';position:absolute;right:-40px;top:-40px;width:180px;height:180px;background:radial-gradient(circle,rgba(70,166,167,0.10),transparent 70%);pointer-events:none}
    .wb-hero-row{display:flex;align-items:center;gap:14px;margin-bottom:14px}
    .wb-hero-avatar{width:56px;height:56px;border-radius:50%;background:linear-gradient(140deg,#c8b8a0,#8a6e52);box-shadow:inset 0 -12px 18px rgba(0,0,0,0.18),inset 0 0 0 1.5px rgba(255,255,255,0.25);display:grid;place-items:center;color:white;font-weight:600;font-size:16px;flex-shrink:0}
    .wb-hero-name{font-size:18px;font-weight:600;color:var(--ink-900);letter-spacing:-0.01em}
    .wb-hero-meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-500);margin-top:2px}
    .wb-hero-status-pill{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;font-weight:600;padding:4px 10px;border-radius:99px;background:var(--critical-bg);color:var(--critical);text-transform:uppercase;letter-spacing:0.08em}
    .wb-hero-status-pill::before{content:'';width:6px;height:6px;background:var(--critical);border-radius:50%;animation:wbBlinkAlert 1.4s infinite}
    .wb-hero-status-since{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--ink-500)}
    /* Stage ECG strip */
    .wb-stage-ecg{background:radial-gradient(ellipse at 50% 100%,rgba(40,180,120,0.08),transparent 60%),linear-gradient(180deg,#0a1119,#0d1421);border-radius:12px;border:1px solid rgba(70,166,167,0.18);overflow:hidden;position:relative;height:110px;margin-top:12px}
    .wb-stage-ecg-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(50,180,120,0.05) 1px,transparent 1px),linear-gradient(90deg,rgba(50,180,120,0.05) 1px,transparent 1px),linear-gradient(rgba(50,180,120,0.10) 1px,transparent 1px),linear-gradient(90deg,rgba(50,180,120,0.10) 1px,transparent 1px);background-size:10px 10px,10px 10px,50px 50px,50px 50px;-webkit-mask-image:linear-gradient(90deg,transparent,black 6%,black 94%,transparent);mask-image:linear-gradient(90deg,transparent,black 6%,black 94%,transparent)}
    .wb-stage-ecg-line{position:absolute;inset:0;overflow:hidden;-webkit-mask-image:linear-gradient(90deg,transparent 0%,black 7%,black 93%,transparent 100%);mask-image:linear-gradient(90deg,transparent 0%,black 7%,black 93%,transparent 100%)}
    .wb-stage-ecg-track{position:absolute;left:0;top:0;height:100%;width:200%;animation:wbEcgScroll 2.4s linear infinite}
    @keyframes wbEcgScroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
    .wb-stage-ecg-track svg{width:100%;height:100%;display:block}
    .wb-stage-ecg-track svg .trace{fill:none;stroke:oklch(0.78 0.22 145);stroke-width:2;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 3px oklch(0.88 0.22 145)) drop-shadow(0 0 8px rgba(80,220,140,0.4))}
    .wb-ecg-meta-bar{position:absolute;left:14px;top:10px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:rgba(120,255,180,0.7);font-weight:600;display:flex;gap:8px;align-items:center;z-index:1}
    .wb-ecg-lead-tag{border:1px solid rgba(120,255,180,0.22);padding:1px 5px;border-radius:3px;font-size:9px}
    .wb-ecg-bpm{position:absolute;right:14px;top:10px;text-align:right;font-family:'JetBrains Mono',monospace;color:rgba(120,255,180,0.85);z-index:1}
    .wb-ecg-bpm .v{font-size:20px;font-weight:700;line-height:1;text-shadow:0 0 6px rgba(120,255,180,0.45)}
    .wb-ecg-bpm .k{font-size:8.5px;color:rgba(255,255,255,0.4);letter-spacing:0.12em;text-transform:uppercase;margin-top:2px}
    /* Stage vitals */
    .wb-stage-vitals{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    .wb-stage-vital{background:white;border:1px solid var(--ink-100);border-radius:10px;padding:14px 14px 12px;position:relative;overflow:hidden}
    .wb-stage-vital::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--ink-200)}
    .wb-stage-vital.critical{background:linear-gradient(180deg,var(--critical-bg),white);border-color:#f4c8c2}
    .wb-stage-vital.critical::before{background:var(--critical)}
    .wb-stage-vital.caution{background:linear-gradient(180deg,var(--caution-bg),white);border-color:#f1d9a8}
    .wb-stage-vital.caution::before{background:var(--caution)}
    .wb-stage-vital.normal::before{background:var(--normal)}
    .wb-svital-head{display:flex;align-items:center;gap:6px;margin-bottom:6px}
    .wb-svital-name{font-size:11px;font-weight:600;color:var(--ink-700);letter-spacing:0.02em}
    .wb-svital-flag{margin-left:auto;font-size:9.5px;font-weight:700;padding:1px 6px;border-radius:3px;text-transform:uppercase;letter-spacing:0.04em}
    .wb-svital-flag.low,.wb-svital-flag.high{background:var(--critical);color:white}
    .wb-svital-flag.warn{background:var(--caution);color:white}
    .wb-svital-val{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:30px;color:var(--ink-900);line-height:1;letter-spacing:-0.02em;display:flex;align-items:baseline;gap:4px}
    .wb-svital-val small{font-size:12px;color:var(--ink-500);font-weight:500;letter-spacing:0}
    .wb-svital-trend{font-size:10px;color:var(--ink-500);margin-top:6px;font-family:'JetBrains Mono',monospace}
    /* Stage meta row */
    .wb-stage-meta{display:flex;gap:18px;padding:10px 14px;background:rgba(255,255,255,0.6);border:1px solid var(--ink-100);border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:11px;backdrop-filter:blur(6px);flex-wrap:wrap}
    .wb-stage-meta .mk{color:var(--ink-400);font-size:9.5px;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;display:block}
    .wb-stage-meta .mv{color:var(--ink-900);font-weight:600;font-size:12px}
    .wb-stage-meta .mv.tealv{color:var(--teal-700)}
    /* Grid empty tip */
    .gs-empty-tip{position:absolute;bottom:22px;left:50%;transform:translateX(-50%);font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-400);padding:6px 12px;background:rgba(255,255,255,0.7);backdrop-filter:blur(6px);border:1px dashed var(--ink-200);border-radius:99px;pointer-events:none;white-space:nowrap}
    /* Inner panel bodies — fill their GridStack cell */
    .wb-fp-body .cb-chat-body{max-height:none;flex:1;min-height:200px}
    .wb-fp-body .cb-exam-layout{overflow-y:auto}
    .wb-fp-body .cb-kg-stage{height:100%;min-height:300px}
    .wb-fp-body .cb-cxr-stage{aspect-ratio:1/1}
  </style>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800" style="display:flex;flex-direction:column;min-height:100vh">

<!-- HEADER -->
<header class="cb-topbar sticky top-0 z-50">
  <div class="cb-topbar-row">
    <button class="cb-burger" data-cb-burger type="button" aria-label="Open menu" title="Menu">
      <span class="lines"><span></span><span></span><span></span></span>
    </button>
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
      <button id="submitCaseBtn" hidden onclick="showSubmitConfirm()" class="cb-topbar-btn-submit" title="Submit case for final evaluation">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8l3 3 7-7"/></svg>Submit
      </button>
      <button id="saveExitBtn" hidden onclick="saveAndExit()" class="cb-topbar-btn-save-exit" title="Save evaluated case and return to dashboard">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8l4 4 8-8"/></svg>Save &amp; Exit
      </button>
      <a href="/home" id="dashLink" hidden class="cb-topbar-btn">Home</a>
      <div id="userAvatar" hidden class="cb-avatar" title="" style="cursor:default;background:linear-gradient(135deg,var(--teal-700),var(--teal-500))"></div>
      <button id="logoutBtn" hidden onclick="doLogout()" class="cb-topbar-btn">Log out</button>
      <a href="/" id="loginLink" class="cb-topbar-btn">Log in</a>
    </div>
  </div>
  <!-- Workflow stepper -->
</header>

<main style="width:100%;flex:1;min-height:0;display:flex;flex-direction:column">

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

  <!-- ================================================================
       WORKBENCH v3: left-rail · left-dock · stage · right-dock · right-rail
       ================================================================ -->
  <div class="wb-workbench">

    <!-- ══════════ LEFT RAIL ══════════ -->
    <aside class="wb-rail wb-rail-left" aria-label="Clinician inputs">
      <div class="wb-rail-label">YOU</div>

      <button class="wb-rail-tab active" data-panel="panel-interview" title="Patient Interview">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="M3 5h18v11H8l-4 4z"/><path d="M7 9h10M7 12h7"/></svg>
        <span class="wb-rt-label">Interview</span>
        <span class="wb-rt-meta" id="chatTurnBadge">—</span>
      
        <span class="wb-tab-pin-dot" data-pin-tab="panel-interview" title="Pin panel" onclick="event.stopPropagation();wbTogglePin('panel-interview')"><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></span>
      </button>

      <button class="wb-rail-tab" data-panel="panel-phys-exam" title="Physical Examination">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><circle cx="12" cy="5" r="2.4"/><path d="M12 8v7M8 11l4-2 4 2M9 21l3-6 3 6"/></svg>
        <span class="wb-rt-label">Physical Exam</span>
        <span class="wb-rt-meta" id="examDoneBadge">—</span>
      
        <span class="wb-tab-pin-dot" data-pin-tab="panel-phys-exam" title="Pin panel" onclick="event.stopPropagation();wbTogglePin('panel-phys-exam')"><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></span>
      </button>

      <button class="wb-rail-tab" data-panel="panel-invest" title="Order Investigations">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="M9 3h6v5l4 8a3 3 0 0 1-2.7 4H7.7A3 3 0 0 1 5 16l4-8z"/></svg>
        <span class="wb-rt-label">Investigations</span>
        <span class="wb-rt-meta" id="invCountBadge">—</span>

        <span class="wb-tab-pin-dot" data-pin-tab="panel-invest" title="Pin panel" onclick="event.stopPropagation();wbTogglePin('panel-invest')"><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></span>
      </button>

      <div class="wb-rail-spacer"></div>
      <div class="wb-rail-foot">CLINICIAN<br/>INPUTS</div>
    </aside>

    <!-- ══════════ PANEL STORE (off-grid, panels live here when closed) ══════════ -->
    <div id="gs-store" aria-hidden="true" style="display:none">

      <section class="wb-float-panel pinned" id="panel-patient">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Patient</span>
            <span class="wb-fp-sub" id="ppHeadSub">On presentation</span>
          </div>
        </div>
        <div class="wb-fp-body">
          <div class="pp-scroll">
            <div class="pp-head">
              <div class="pp-id">
                <div class="pp-avatar" id="ppAvatar">?</div>
                <div class="pp-id-text">
                  <div class="pp-name"><span id="ppName">—</span> <span class="pp-desc" id="ppDesc"></span></div>
                  <div class="pp-live"><span class="pp-live-dot"></span>Monitoring · On presentation</div>
                </div>
              </div>
              <div class="pp-persona" id="ppPersona"></div>
              <div class="pp-chips">
                <span class="pp-chip" id="ppAgeChip">Age <b id="ppAge">—</b></span>
                <span class="pp-chip" id="ppSexChip">Sex <b id="ppSex">—</b></span>
                <span class="pp-chip" id="ppRaceChip" hidden>Race <b id="ppRace"></b></span>
                <span class="pp-chip role" id="ppOccChip" hidden><span id="ppOcc"></span></span>
              </div>
            </div>
            <div class="pp-cc">
              <div class="pp-sec-label">Chief complaint</div>
              <div class="pp-cc-quote"><em id="ppCC">—</em></div>
            </div>
            <div class="pp-vitals">
              <div class="pp-sec-label">Vitals on presentation</div>
              <div class="pp-vitals-grid">
                <div class="pp-vital" id="ppVitHR">
                  <div class="pp-v-top">
                    <span class="pp-v-label">HR</span>
                    <span class="pp-v-arrow" id="ppArrowHR" hidden><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 13V3M4 7l4-4 4 4"/></svg></span>
                  </div>
                  <div class="pp-v-val"><span id="ppHR">—</span> <span class="pp-v-unit">bpm</span></div>
                  <div class="pp-v-flag" id="ppFlagHR" hidden></div>
                  <div class="pp-v-ok" id="ppOkHR">within range</div>
                </div>
                <div class="pp-vital" id="ppVitBP">
                  <div class="pp-v-top">
                    <span class="pp-v-label">BP</span>
                    <span class="pp-v-arrow" id="ppArrowBP" hidden><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 13V3M4 7l4-4 4 4"/></svg></span>
                  </div>
                  <div class="pp-v-val"><span id="ppBP">—</span> <span class="pp-v-unit">mmHg</span></div>
                  <div class="pp-v-flag" id="ppFlagBP" hidden></div>
                  <div class="pp-v-ok" id="ppOkBP">within range</div>
                </div>
                <div class="pp-vital" id="ppVitSPO2">
                  <div class="pp-v-top">
                    <span class="pp-v-label">SpO<sub>2</sub></span>
                    <span class="pp-v-arrow" id="ppArrowSPO2" hidden><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v10M4 9l4 4 4-4"/></svg></span>
                  </div>
                  <div class="pp-v-val"><span id="ppSPO2">—</span> <span class="pp-v-unit">%</span></div>
                  <div class="pp-v-flag" id="ppFlagSPO2" hidden></div>
                  <div class="pp-v-ok" id="ppOkSPO2">within range</div>
                </div>
                <div class="pp-vital" id="ppVitRR">
                  <div class="pp-v-top">
                    <span class="pp-v-label">RR</span>
                    <span class="pp-v-arrow" id="ppArrowRR" hidden><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 13V3M4 7l4-4 4 4"/></svg></span>
                  </div>
                  <div class="pp-v-val"><span id="ppRR">—</span> <span class="pp-v-unit">/min</span></div>
                  <div class="pp-v-flag" id="ppFlagRR" hidden></div>
                  <div class="pp-v-ok" id="ppOkRR">within range</div>
                </div>
              </div>
            </div>
          </div>
          <div class="pp-ecg" id="ppEcgWrap" hidden>
            <div class="pp-ecg-bar">
              <span class="pp-ecg-lead">II</span>
              <span class="pp-ecg-cap">Lead II · live</span>
              <div class="pp-ecg-hr">
                <span class="pp-heart" id="ppHeart"><svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 14s-5.5-3.8-5.5-7.3A3 3 0 0 1 8 4.6a3 3 0 0 1 5.5 2.1C13.5 10.2 8 14 8 14Z"/></svg></span>
                <span class="pp-hr-num" id="ppHRMon">—</span>
                <span class="pp-hr-unit">bpm</span>
              </div>
            </div>
            <div class="pp-ecg-canvas-wrap"><canvas id="ppEcg"></canvas></div>
            <div class="pp-ecg-foot">
              <span class="pp-gain">25 mm/s · 10 mm/mV</span>
              <span id="ppRhythm">SINUS RHYTHM</span>
            </div>
          </div>
        </div>
      </section>

      <section class="wb-float-panel" id="panel-interview">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Patient Interview</span>
            <span class="wb-fp-sub" id="chatTurnSub">—</span>
          </div>
          <div class="wb-fp-actions">
            <button class="cb-iconbtn" title="Toggle voice" onclick="toggleTts()" style="width:22px;height:22px">
              <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/></svg>
            </button>
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-interview')" data-pin-id="panel-interview" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-interview')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body" style="display:flex;flex-direction:column">
          <div id="chatArea" class="cb-chat-body" style="flex:1;max-height:none">
            <div class="cb-chat-divider">Generate a case to begin</div>
          </div>
          <div class="cb-chat-foot">
            <div class="cb-chat-ctrl">
              <label class="cb-toggle" onclick="toggleTts()">
                <span id="ttsTrack" class="cb-toggle-track"></span>
                <span>Read responses aloud</span>
              </label>
              <span id="ttsVoiceLabel" style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--ink-400)">EN-US · Auto</span>
            </div>
            <div class="cb-chat-input">
              <input id="question" placeholder="Ask the patient a question…"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQuestion()}"/>
              <button id="micBtn" onclick="toggleMic()" title="Speak your question" class="cb-chat-mic">
                <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
              </button>
              <button id="askBtn" onclick="askQuestion()" class="cb-chat-send" title="Send">
                <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8l12-6-4 14-3-6-5-2z"/></svg>
              </button>
            </div>
          </div>
        </div>
      </section>

      <section class="wb-float-panel" id="panel-phys-exam">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Physical Examination</span>
            <span class="wb-fp-sub">Select systems, then examine</span>
          </div>
          <div class="wb-fp-actions">
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-phys-exam')" data-pin-id="panel-phys-exam" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-phys-exam')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div class="pe-body">
            <div class="pe-top">
              <span class="pe-label">Body systems</span>
              <div class="pe-selectbar">
                <span class="pe-count-pill"><b id="peCount">0</b> of 9 selected</span>
                <button class="pe-link" id="peToggleAll" type="button" onclick="peToggleAll()">Select all</button>
              </div>
            </div>
            <div class="pe-systems" id="examSystems">
              <div class="pe-sys" data-name="General"             style="--c:#5b6b85" onclick="toggleExamSys(this,'General')"><span class="dot"></span><span class="name">General</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Cardiovascular"      style="--c:#ef4444" onclick="toggleExamSys(this,'Cardiovascular')"><span class="dot"></span><span class="name">Cardiovascular</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Respiratory"         style="--c:#06b6d4" onclick="toggleExamSys(this,'Respiratory')"><span class="dot"></span><span class="name">Respiratory</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Abdominal"           style="--c:#f97316" onclick="toggleExamSys(this,'Abdominal')"><span class="dot"></span><span class="name">Abdominal</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Neurological"        style="--c:#8b5cf6" onclick="toggleExamSys(this,'Neurological')"><span class="dot"></span><span class="name">Neurological</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Musculoskeletal"     style="--c:#10b981" onclick="toggleExamSys(this,'Musculoskeletal')"><span class="dot"></span><span class="name">Musculoskeletal</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Dermatological"      style="--c:#ec4899" onclick="toggleExamSys(this,'Dermatological')"><span class="dot"></span><span class="name">Dermatological</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="ENT/Head &amp; Neck"  style="--c:#f59e0b" onclick="toggleExamSys(this,'ENT/Head &amp; Neck')"><span class="dot"></span><span class="name">ENT / Head &amp; Neck</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
              <div class="pe-sys" data-name="Peripheral Vascular" style="--c:#6366f1" onclick="toggleExamSys(this,'Peripheral Vascular')"><span class="dot"></span><span class="name">Peripheral Vascular</span><span class="check"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span></div>
            </div>
          </div>
          <div class="pe-foot">
            <div class="pe-meta" id="peMeta">No systems selected yet.</div>
            <button class="pe-examine" id="examBtn" type="button" onclick="performExam()" disabled>
              Examine selected
              <span class="badge" id="peBadge">0</span>
            </button>
          </div>
          <div id="examResults" class="cb-exam-results"></div>
        </div>
      </section>

      <section class="wb-float-panel wide" id="panel-invest">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Investigations</span>
            <span class="wb-fp-sub" id="invOrderedSub">—</span>
          </div>
          <div class="wb-fp-actions">
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-invest')" data-pin-id="panel-invest" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-invest')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div style="padding:12px 14px;display:flex;flex-direction:column;gap:10px">
            <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
              <span style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-400);font-weight:600;margin-right:2px;flex-shrink:0">Quick order</span>
              <button class="qchip" data-inv="abg"      onclick="quickOrder('ABG')">ABG</button>
              <button class="qchip" data-inv="cbc"      onclick="quickOrder('CBC')">CBC</button>
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
                style="font:inherit;font-size:11px;font-weight:500;padding:6px 10px;border-radius:6px;background:white;border:1px solid var(--ink-200);color:var(--ink-700);cursor:pointer;flex-shrink:0">Order</button>
            </div>
            <div id="invResults" style="display:flex;flex-direction:column;gap:10px"></div>
          </div>
        </div>
      </section>

      <section class="wb-float-panel wide" id="panel-reasoning">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Reasoning &amp; Diagnosis</span>
            <span class="wb-fp-sub" id="interpCharCount">0 / 1000</span>
          </div>
          <div class="wb-fp-actions">
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-reasoning')" data-pin-id="panel-reasoning" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-reasoning')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div class="cb-panel-body" style="gap:14px">
            <div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--ink-500);font-weight:600">Clinical Interpretation</span>
              </div>
              <textarea id="interpretation" rows="4" placeholder="Your interpretation of ordered tests…"
                class="cb-field" oninput="updateInterpCount(this)" style="resize:vertical;width:100%"></textarea>
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-top:6px;gap:8px">
                <ul id="interpList" style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:3px;flex:1;font-size:11.5px;color:var(--ink-700)"></ul>
                <button id="interpBtn" onclick="submitInterpretation()"
                  style="background:var(--ink-900);color:white;border:0;font:inherit;font-size:11.5px;font-weight:600;padding:7px 12px;border-radius:6px;cursor:pointer;flex-shrink:0"
                  onmouseover="this.style.background='var(--ink-800)'" onmouseout="this.style.background='var(--ink-900)'">Save</button>
              </div>
            </div>
            <div class="cb-section-div">Working Diagnosis</div>
            <div>
              <div style="display:flex;gap:6px">
                <input id="diagnosisGuess" placeholder="Your working diagnosis…"
                  class="cb-field" style="flex:1;min-width:0"
                  onkeydown="if(event.key==='Enter')submitDiagnosis()"/>
                <button id="diagnosisBtn" onclick="submitDiagnosis()"
                  style="background:var(--teal-600);color:white;border:0;font:inherit;font-size:11.5px;font-weight:600;padding:8px 12px;border-radius:6px;cursor:pointer;flex-shrink:0;white-space:nowrap;display:inline-flex;align-items:center;gap:5px"
                  onmouseover="this.style.background='var(--teal-500)'" onmouseout="this.style.background='var(--teal-600)'">
                  Submit <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M3 8l3 3 7-7"/></svg>
                </button>
              </div>
              <div id="diagnosisResult" hidden style="margin-top:8px;background:var(--teal-50);border:1px solid oklch(0.85 0.06 195);border-radius:7px;padding:8px 10px;font-size:12px;color:var(--teal-700)" class="fade-in"></div>
            </div>
            <div class="cb-section-div">Senior Doctor</div>
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
                  onmouseover="this.style.background='var(--ink-800)'" onmouseout="this.style.background='var(--ink-900)'">Ask</button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="wb-float-panel" id="panel-imaging">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Imaging</span>
            <span class="wb-fp-sub" id="imagingPanelSub">Order studies · open results</span>
          </div>
          <div class="wb-fp-actions">
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-imaging')" data-pin-id="panel-imaging" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-imaging')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <button type="button" class="img-card" data-modality="cxr" data-state="idle" onclick="onImagingCardClick('cxr')">
            <span class="img-card-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 7c-1 4-1 7 0 11M15 7c1 4 1 7 0 11M12 5v15"/></svg></span>
            <span class="img-card-text">
              <span class="img-card-name">Chest X-ray</span>
              <span class="img-card-sub" data-role="status">Not ordered</span>
            </span>
            <span class="img-card-cta"><span data-role="cta">Order</span>
              <span class="img-card-spin"></span>
              <svg class="img-card-check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8l3 3 7-7"/></svg>
            </span>
          </button>
          <button type="button" class="img-card" data-modality="ecg" data-state="idle" onclick="onImagingCardClick('ecg')">
            <span class="img-card-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h3l2-6 3 12 2-8 2 4h8"/></svg></span>
            <span class="img-card-text">
              <span class="img-card-name">12-Lead ECG</span>
              <span class="img-card-sub" data-role="status">Not ordered</span>
            </span>
            <span class="img-card-cta"><span data-role="cta">Order</span>
              <span class="img-card-spin"></span>
              <svg class="img-card-check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8l3 3 7-7"/></svg>
            </span>
          </button>
          <button type="button" class="img-card" data-modality="us" data-state="disabled" disabled>
            <span class="img-card-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="M5 14a7 7 0 0 1 14 0"/><path d="M5 14v3a3 3 0 0 0 3 3M19 14v3a3 3 0 0 1-3 3"/><circle cx="12" cy="14" r="1.6"/></svg></span>
            <span class="img-card-text">
              <span class="img-card-name">Ultrasound</span>
              <span class="img-card-sub" data-role="status">Coming soon</span>
            </span>
            <span class="img-card-cta"><span data-role="cta">TBA</span></span>
          </button>
          <button type="button" class="img-card" data-modality="ct" data-state="disabled" disabled>
            <span class="img-card-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg></span>
            <span class="img-card-text">
              <span class="img-card-name">CT Scan</span>
              <span class="img-card-sub" data-role="status">Coming soon</span>
            </span>
            <span class="img-card-cta"><span data-role="cta">TBA</span></span>
          </button>
          <button type="button" class="img-card" data-modality="mri" data-state="disabled" disabled>
            <span class="img-card-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 6v12M17 6v12M3 12h18"/></svg></span>
            <span class="img-card-text">
              <span class="img-card-name">MRI</span>
              <span class="img-card-sub" data-role="status">Coming soon</span>
            </span>
            <span class="img-card-cta"><span data-role="cta">TBA</span></span>
          </button>
        </div>
      </section>

      <section class="wb-float-panel" id="panel-consults">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Consults</span>
            <span class="wb-fp-sub" id="consultsPanelSub">Call a specialist team</span>
          </div>
          <div class="wb-fp-actions">
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-consults')" data-pin-id="panel-consults" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-consults')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <button type="button" class="cons-card" data-dept="neurology" data-state="disabled" disabled>
            <span class="cons-card-icon">NEU</span>
            <span class="cons-card-text">
              <span class="cons-card-name">Neurology</span>
              <span class="cons-card-sub">On call · TBA</span>
            </span>
            <span class="cons-card-cta">TBA</span>
          </button>
          <button type="button" class="cons-card" data-dept="nephrology" data-state="disabled" disabled>
            <span class="cons-card-icon">NEP</span>
            <span class="cons-card-text">
              <span class="cons-card-name">Nephrology</span>
              <span class="cons-card-sub">On call · TBA</span>
            </span>
            <span class="cons-card-cta">TBA</span>
          </button>
          <button type="button" class="cons-card" data-dept="icu" data-state="disabled" disabled>
            <span class="cons-card-icon">ICU</span>
            <span class="cons-card-text">
              <span class="cons-card-name">Intensive Care</span>
              <span class="cons-card-sub">On call · TBA</span>
            </span>
            <span class="cons-card-cta">TBA</span>
          </button>
        </div>
      </section>

      <section class="wb-float-panel wide" id="panel-kg">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Knowledge Graph</span>
            <span class="wb-fp-sub" id="graphBadge">—</span>
          </div>
          <div class="wb-fp-actions">
            <div id="graphControls" hidden style="display:flex;align-items:center;gap:2px">
              <button onclick="event.stopPropagation();graphZoomIn()" class="cb-iconbtn" style="width:22px;height:22px;font-size:14px;font-weight:700">+</button>
              <button onclick="event.stopPropagation();graphZoomOut()" class="cb-iconbtn" style="width:22px;height:22px;font-size:14px;font-weight:700">−</button>
              <button onclick="event.stopPropagation();graphFit()" class="cb-iconbtn" style="width:30px;height:22px;font-size:10px">Fit</button>
            </div>
            <button onclick="toggleKgExpand()" id="kgExpandBtn" class="cb-iconbtn" style="width:22px;height:22px" title="Expand">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M1 4.5V1h3.5M7.5 1H11v3.5M11 7.5V11H7.5M4.5 11H1V7.5"/></svg>
            </button>
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-kg')" data-pin-id="panel-kg" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-kg')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div class="cb-kg-stage">
            <div class="cb-kg-grid"></div>
            <div id="kgStatBar" hidden class="cb-kg-stat">
              <span><strong id="kgNodeCount">0</strong> nodes</span>
              <span><strong id="kgEdgeCount">0</strong> edges</span>
              <span id="kgRecentSpan" hidden><strong id="kgRecentCount">+0</strong> last min</span>
            </div>
            <div id="graphEmpty" class="absolute inset-0 flex flex-col items-center justify-center gap-2 pointer-events-none">
              <svg class="w-10 h-10" fill="none" stroke="rgba(255,255,255,0.18)" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"/></svg>
              <p class="text-xs" style="color:rgba(255,255,255,0.3)">Ask questions to build the graph</p>
            </div>
            <div id="graphStatus" hidden class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <p class="text-xs italic" style="color:rgba(255,255,255,0.4)">Waiting…</p>
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
      </section>

      <section class="wb-float-panel viewer" id="panel-cxr">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Chest Radiograph</span>
            <span class="wb-fp-sub" id="cxrFindingLabel">PA · SYNTH</span>
          </div>
          <div class="wb-fp-actions">
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-cxr')" data-pin-id="panel-cxr" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-cxr')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body" style="padding:0">
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
              <span>RoentGen-v2 · synth</span>
            </div>
            <div id="cxrFindingBanner" hidden class="cb-cxr-finding-banner">
              <span class="cb-cxr-finding-tag">AI · Provisional</span>
              <div id="cxrFindingText" class="cb-cxr-finding-text"></div>
            </div>
          </div>
        </div>
      </section>

      <section class="wb-float-panel wide" id="panel-ecg">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Electrocardiogram · 12-lead</span>
            <span class="wb-fp-sub" id="ecgPanelSub">—</span>
          </div>
          <div class="wb-fp-actions">
            <div id="ecgRateToggle" class="cb-rate-toggle" hidden title="Sample rate of the source signal">
              <button type="button" class="cb-rate-btn" data-rate="100" onclick="setEcgRate(100)">100 Hz</button>
              <button type="button" class="cb-rate-btn active" data-rate="500" onclick="setEcgRate(500)">500 Hz</button>
            </div>
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-ecg')" data-pin-id="panel-ecg" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-ecg')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div id="ecgContent" class="ecg-wrap"></div>
          <div id="ecgMeta" hidden class="cb-ecg-meta">
            <span id="ecgMetaLeft">—</span>
            <span id="ecgMetaRight" style="text-align:right;max-width:60%"></span>
          </div>
        </div>
      </section>

      <section class="wb-float-panel wide" id="panel-perf">
        <div class="wb-fp-head">
          <div class="wb-fp-title-group">
            <span class="wb-fp-title">Performance Report</span>
            <span class="wb-fp-sub" id="perfScoreSub">—</span>
          </div>
          <div class="wb-fp-actions">
            
            <button class="wb-pin-btn" onclick="wbTogglePin('panel-perf')" data-pin-id="panel-perf" title="Pin panel"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></button>
            <button class="wb-fp-close" onclick="wbClosePanel('panel-perf')" title="Collapse">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>
            </button>
          </div>
        </div>
        <div class="wb-fp-body">
          <div id="evalResults" class="cb-panel-body">
            <div id="perfStatusBadge" class="cb-perf-status in-progress" hidden>
              <svg width="7" height="7" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="currentColor"/></svg>
              In Progress · partial report
            </div>
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
            <div class="cb-perf-metrics">
              <div class="cb-perf-metric"><div class="cb-perf-label">Diagnosis</div><div id="evalDx" style="font-size:12px;margin-top:3px;color:var(--ink-900)">—</div></div>
              <div class="cb-perf-metric"><div class="cb-perf-label">Symptoms</div><div id="evalSymptoms" style="font-size:12px;margin-top:3px">—</div></div>
              <div class="cb-perf-metric"><div class="cb-perf-label">Investigations</div><div id="evalInvestigations" style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600;color:var(--ink-900);margin-top:2px;display:flex;align-items:baseline;gap:3px">—</div><div class="cb-perf-bar"><div id="evalInvBar" class="cb-perf-bar-fill" style="width:0%;background:var(--teal-500)"></div></div></div>
              <div class="cb-perf-metric"><div class="cb-perf-label">Time on Task</div><div id="evalTimeOnTask" style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600;color:var(--ink-900);margin-top:2px">—</div><div class="cb-perf-bar"><div id="evalTimeBar" class="cb-perf-bar-fill" style="width:0%;background:var(--ink-400)"></div></div></div>
              <div class="cb-perf-metric" style="grid-column:1/-1"><div class="cb-perf-label">Question Quality</div><div id="evalQuality" style="font-size:12px;margin-top:3px;color:var(--ink-700)">—</div><div class="cb-perf-bar"><div id="evalQualityBar" class="cb-perf-bar-fill" style="width:0%;background:var(--teal-500)"></div></div></div>
            </div>
            <div><div class="cb-section-div">Reasoning</div><div id="evalReasoning" style="display:flex;flex-direction:column;gap:6px"></div></div>
            <div><div class="cb-section-div">Learning</div><div id="evalLearning" style="display:flex;flex-direction:column;gap:6px"></div></div>
          </div>
        </div>
      </section>
    </div>

    <!-- ══════════ GRID AREA ══════════ -->
    <div id="gs-area">
      <div class="grid-stack" id="panelGrid"></div>
      <div class="gs-empty-tip" id="wbEmptyTip">Tap a tab to open a panel &nbsp;·&nbsp; Esc closes all &nbsp;·&nbsp; Generate a case to begin</div>
    </div>

    <!-- ══════════ RIGHT RAIL ══════════ -->
    <aside class="wb-rail wb-rail-right" aria-label="Outputs and references">
      <div class="wb-rail-label">DATA</div>

      <button class="wb-rail-tab" data-panel="panel-imaging" title="Imaging">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><rect x="3" y="4" width="18" height="14" rx="2"/><circle cx="12" cy="11" r="3"/><path d="M3 8h3M18 8h3"/></svg>
        <span class="wb-rt-label">Imaging</span>
        <span class="wb-rt-meta" id="imagingRailMeta">0/2</span>

        <span class="wb-tab-pin-dot" data-pin-tab="panel-imaging" title="Pin panel" onclick="event.stopPropagation();wbTogglePin('panel-imaging')"><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></span>
      </button>

      <button class="wb-rail-tab" data-panel="panel-consults" title="Consults">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="M5 4h11l3 3v9a2 2 0 0 1-2 2H8l-3 3z"/><path d="M9 10h6M9 13h4"/></svg>
        <span class="wb-rt-label">Consults</span>
        <span class="wb-rt-meta" id="consultsRailMeta">0/3</span>

        <span class="wb-tab-pin-dot" data-pin-tab="panel-consults" title="Pin panel" onclick="event.stopPropagation();wbTogglePin('panel-consults')"><svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h6l-1 4.5 2 2H4l2-2z"/><line x1="8" y1="14" x2="8" y2="8.5"/></svg></span>
      </button>

      <div class="wb-rail-spacer"></div>
      <div class="wb-rail-foot">RESULTS<br/>&amp; OUTPUTS</div>
    </aside>

  </div>
</main>

<!-- STATUS BAR -->
<footer class="cb-statusbar" aria-label="Status bar">
  <span id="sbCase"><strong>CASE</strong> —</span>
  <span id="sbSession"><strong>SESSION</strong> inactive</span>
  <span id="sbDiag"><strong>Dx</strong> —</span>
  <span id="sbAutosave" hidden><span class="cb-sb-dot"></span><span id="sbSaveTime">—</span></span>
  <span id="sbTimer" hidden><strong>TIME</strong> <span id="sbTimerVal">00:00</span></span>
  <div class="cb-sb-tabs">
    <button class="cb-sb-tab" data-panel="panel-perf" title="Performance Report" onclick="wbTogglePanel('panel-perf')">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 9 9h-9z" fill="currentColor" opacity="0.18" stroke="none"/><path d="M12 12V6M12 12l5 3"/></svg>
      <span>Performance</span>
      <span class="cb-sb-tab-meta" id="perfRailMeta">—</span>
    </button>
    <button class="cb-sb-tab" data-panel="panel-kg" title="Knowledge Graph" onclick="wbTogglePanel('panel-kg')">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"><circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="12" r="2.4"/><circle cx="6" cy="18" r="2.4"/><circle cx="18" cy="18" r="2.4"/><path d="M8 7l3 4M16 7l-3 4M10 13l-3 3M14 13l3 3"/></svg>
      <span>Knowledge Graph</span>
      <span class="cb-sb-tab-meta" id="kgNodeCountBadge">—</span>
    </button>
    <span class="cb-sb-brand">CasesBasis · clinical simulation</span>
  </div>
</footer>

<!-- SUBMIT CONFIRMATION MODAL -->
<div id="submitBackdrop" class="cb-submit-backdrop" role="dialog" aria-modal="true" onclick="if(event.target===this)hideSubmitConfirm()">
  <div class="cb-submit-card">
    <div class="cb-submit-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="10"/>
      </svg>
    </div>
    <div class="cb-submit-title">Submit Case?</div>
    <div class="cb-submit-body">Once submitted, you won't be able to make any further changes.<br>Your performance will be evaluated and saved.</div>
    <div class="cb-submit-actions">
      <button class="cb-submit-cancel" onclick="hideSubmitConfirm()">Cancel</button>
      <button class="cb-submit-confirm" id="submitConfirmBtn" onclick="confirmSubmitCase()">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8l3 3 7-7"/></svg>
        Yes, submit
      </button>
    </div>
  </div>
</div>

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
        <button class="cb-pause-btn cb-pause-ghost" onclick="location.href='/home'">
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

  /* ── Submit confirmation ── */
  function showSubmitConfirm() {
    if (!sessionId) { toast('Generate a case first', 'err'); return; }
    // Reasoning panel no longer has a rail tab — Submit is its entry point.
    // If the resident hasn't filled in a working diagnosis yet, open the panel
    // so they can do so before the confirm modal appears.
    const dx = (document.getElementById('diagnosisGuess')?.value || '').trim();
    if (!dx) {
      wbOpenPanel('panel-reasoning');
      toast('Enter your working diagnosis to submit', 'err');
      setTimeout(() => document.getElementById('diagnosisGuess')?.focus(), 200);
      return;
    }
    document.getElementById('submitBackdrop').classList.add('on');
  }
  function hideSubmitConfirm() {
    document.getElementById('submitBackdrop').classList.remove('on');
  }
  async function confirmSubmitCase() {
    await evaluateCase();
  }
  async function saveAndExit() {
    // Session is already saved as 'evaluated' by evaluateCase() — don't call pause
    // (which would overwrite status back to 'paused')
    location.href = '/home';
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
      _ttsEnabled = true;
      document.getElementById('sessionBadge').hidden = false;
      document.getElementById('pauseBtn').hidden = false;
      document.getElementById('submitCaseBtn').hidden = false;

      // Case preview panels
      populateCasePreview(data.case_preview || {});

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
  let _patientAge = '';
  let _patientHR = null;
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

  /* ── Imaging panel state machine ──────────────────────────────────────
   * Cards hold their state in `data-state`: idle | pending | available |
   * disabled. The CSS reacts to that attribute; this helper updates the
   * card body text and the right-rail meta count.
   */
  const IMG_ACTIVE = ['cxr', 'ecg'];
  const IMG_LABELS = {
    cxr: { name: 'CXR',  order: 'Order',     pending: 'Acquiring…', available: 'Click to view', viewerId: 'panel-cxr' },
    ecg: { name: 'ECG',  order: 'Order',     pending: 'Acquiring…', available: 'Click to view', viewerId: 'panel-ecg' },
  };
  const IMG_STATUS_SUB = {
    idle: 'Not ordered',
    pending: 'Acquiring…',
    available: 'Available · click to view',
  };
  function _imgCard(modality) {
    return document.querySelector('.img-card[data-modality="' + modality + '"]');
  }
  function setImagingCardState(modality, state) {
    const card = _imgCard(modality);
    if (!card) return;
    card.dataset.state = state;
    const sub = card.querySelector('[data-role="status"]');
    const cta = card.querySelector('[data-role="cta"]');
    if (sub && IMG_STATUS_SUB[state]) sub.textContent = IMG_STATUS_SUB[state];
    if (cta) {
      const meta = IMG_LABELS[modality];
      if (meta) {
        cta.textContent = state === 'idle' ? meta.order
                        : state === 'pending' ? meta.pending
                        : state === 'available' ? meta.available
                        : cta.textContent;
      }
    }
    _refreshImagingRailMeta();
  }
  function _refreshImagingRailMeta() {
    let ready = 0;
    IMG_ACTIVE.forEach(m => {
      const c = _imgCard(m);
      if (c && c.dataset.state === 'available') ready++;
    });
    const meta = document.getElementById('imagingRailMeta');
    if (meta) meta.textContent = ready + '/' + IMG_ACTIVE.length;
    const sub = document.getElementById('imagingPanelSub');
    if (sub) sub.textContent = ready === 0
      ? 'Order studies · open results'
      : ready + ' of ' + IMG_ACTIVE.length + ' available';
  }
  function resetImagingPanel() {
    IMG_ACTIVE.forEach(m => setImagingCardState(m, 'idle'));
    _imagingPanelAutoOpenedThisCase = false;
  }
  let _imagingPanelAutoOpenedThisCase = false;
  function _maybeAutoOpenImagingPanel() {
    if (_imagingPanelAutoOpenedThisCase) return;
    _imagingPanelAutoOpenedThisCase = true;
    if (typeof wbOpenPanelKeepInvestigations === 'function') {
      wbOpenPanelKeepInvestigations('panel-imaging');
    }
  }
  function onImagingCardClick(modality) {
    const card = _imgCard(modality);
    if (!card) return;
    const state = card.dataset.state;
    if (state === 'disabled' || state === 'pending') return;
    if (state === 'available') {
      const meta = IMG_LABELS[modality];
      if (meta && meta.viewerId) wbOpenPanelKeepInvestigations(meta.viewerId);
      return;
    }
    // idle → fire the order
    if (!sessionId) { toast('Generate a case first.'); return; }
    setImagingCardState(modality, 'pending');
    const orderName = modality === 'cxr' ? 'CXR' : modality === 'ecg' ? 'ECG' : '';
    if (orderName) quickOrder(orderName);
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

  /* ── populateCasePreview — shared between generateCase and resumeCase ── */
  function populateCasePreview(cp) {
    const occ = cp.occupation||'';
    const gender = (cp.gender||'').toLowerCase();
    const patientName = cp.name||'';
    const v = cp.vitals||{};

    // Identity, persona, chips
    const race = cp.race||'';
    document.getElementById('ppCC').textContent = cp.chief_complaint||'—';
    document.getElementById('ppAge').textContent = cp.age||'—';
    document.getElementById('ppSex').textContent = cp.gender||'—';
    document.getElementById('ppRace').textContent = race;
    document.getElementById('ppRaceChip').hidden = !race;
    document.getElementById('ppOcc').textContent = occ;
    document.getElementById('ppOccChip').hidden = !occ;

    // Vitals — status-driven, two visual states (warn vs ok)
    function vitalTile(tileId, arrowId, flagId, okId, value, {critLow, cautLow, cautHigh, critHigh}={}, flagFn) {
      const tile = document.getElementById(tileId);
      const arrow = document.getElementById(arrowId);
      const flag = document.getElementById(flagId);
      const ok = document.getElementById(okId);
      if (!tile) return 'normal';
      let status = 'normal', dir = null;
      if (value == null) {
        tile.classList.remove('warn');
        if (arrow) arrow.hidden = true;
        if (flag) { flag.hidden = true; flag.textContent = ''; }
        if (ok) { ok.hidden = false; ok.textContent = '—'; }
        return 'unknown';
      }
      if (critHigh != null && value >= critHigh)      { status='critical'; dir='up'; }
      else if (critLow != null && value <= critLow)   { status='critical'; dir='down'; }
      else if (cautHigh != null && value >= cautHigh) { status='caution';  dir='up'; }
      else if (cautLow  != null && value <= cautLow)  { status='caution';  dir='down'; }
      const isWarn = status !== 'normal';
      tile.classList.toggle('warn', isWarn);
      if (arrow) {
        arrow.hidden = !isWarn;
        // Swap arrow direction by rotating the svg path if needed (HR/RR/BP arrows are "up"
        // in markup; SpO2 is "down" in markup — flip on the fly).
        if (isWarn && dir) {
          arrow.style.transform = dir==='down' ? 'rotate(180deg)' : '';
        }
      }
      const flagTxt = isWarn && flagFn ? flagFn(value, status, dir) : '';
      if (flag) { flag.textContent = flagTxt; flag.hidden = !flagTxt; }
      if (ok)   { ok.hidden = isWarn; }
      return status;
    }
    const hr = v.heart_rate, spo2 = v.oxygen_saturation, rr = v.respiratory_rate;
    const sbp = parseInt((v.blood_pressure||'').split('/')[0])||null;
    document.getElementById('ppHR').textContent   = hr   != null ? hr   : '—';
    document.getElementById('ppBP').textContent   = v.blood_pressure || '—';
    document.getElementById('ppSPO2').textContent = spo2 != null ? spo2 : '—';
    document.getElementById('ppRR').textContent   = rr   != null ? rr   : '—';
    const hrStatus = vitalTile('ppVitHR',  'ppArrowHR',  'ppFlagHR',  'ppOkHR',  hr,
      {critLow:40, cautLow:50, cautHigh:100, critHigh:150},
      (n) => n>=150?'marked tachycardia':n>=120?'sinus tachycardia':n>=100?'tachycardic':n<=40?'severe bradycardia':n<=50?'bradycardia':'');
    vitalTile('ppVitBP',  'ppArrowBP',  'ppFlagBP',  'ppOkBP',  sbp,
      {critLow:80, cautLow:100, cautHigh:160, critHigh:180},
      (n) => n>=180?'hypertensive crisis':n>=160?'hypertensive':n<=80?'hypotensive shock':n<=100?'borderline low':'');
    vitalTile('ppVitSPO2','ppArrowSPO2','ppFlagSPO2','ppOkSPO2',spo2,
      {critLow:88, cautLow:94},
      (n) => n<=88?'critical hypoxia':n<=91?'hypoxemic · room air':n<=94?'mild hypoxia':'');
    vitalTile('ppVitRR',  'ppArrowRR',  'ppFlagRR',  'ppOkRR',  rr,
      {cautHigh:22, critHigh:30},
      (n) => n>=30?'accessory muscle use':n>=22?'tachypneic':n<=10?'bradypneic':'');

    // Rhythm caption (purely informational)
    const rhythm = (hr == null) ? 'SINUS RHYTHM'
                 : (hr >= 150) ? 'MARKED TACHYCARDIA'
                 : (hr >= 100) ? 'SINUS TACHYCARDIA'
                 : (hr <= 40)  ? 'MARKED BRADYCARDIA'
                 : (hr <= 50)  ? 'SINUS BRADYCARDIA'
                 : 'NORMAL SINUS RHYTHM';
    const rhythmEl = document.getElementById('ppRhythm');
    if (rhythmEl) rhythmEl.textContent = rhythm;

    // Identity — name + descriptor + persona
    const persona = cp.persona_description||'';
    const statusMatch = persona.match(/^An?\s+(.+?)\s+who\s/i);
    const descriptor = statusMatch ? '— ' + statusMatch[1] : '';
    const fallbackName = patientName || (occ ? occ : (gender.startsWith('f') ? 'Female patient' : 'Male patient'));
    document.getElementById('ppName').textContent = fallbackName;
    document.getElementById('ppDesc').textContent = descriptor;
    document.getElementById('ppPersona').textContent = persona;

    // Avatar initials
    let avi;
    if (patientName) {
      const p = patientName.trim().split(/\s+/);
      avi = p.length>=2?(p[0][0]+p[p.length-1][0]).toUpperCase():p[0].substring(0,2).toUpperCase();
    } else if (occ) {
      const w = occ.trim().split(/\s+/).filter(Boolean);
      avi = w.length>=2?(w[0][0]+w[w.length-1][0]).toUpperCase():w[0].substring(0,2).toUpperCase();
    } else {
      avi = gender.startsWith('m')?'M':gender.startsWith('f')?'F':'?';
    }
    document.getElementById('ppAvatar').textContent = avi;

    // Chat bubble initials
    if (patientName) {
      const p = patientName.trim().split(/\s+/);
      _patientInitials = p.length>=2?(p[0][0]+p[p.length-1][0]).toUpperCase():p[0].substring(0,2).toUpperCase();
    } else {
      const w = occ.trim().split(/\s+/).filter(Boolean);
      _patientInitials = w.length>=2?(w[0][0]+w[w.length-1][0]).toUpperCase():w.length===1?w[0].substring(0,2).toUpperCase():(gender.startsWith('m')?'MP':'FP');
    }

    _patientGender = gender;
    _patientAge = cp.age || '';
    _patientHR = hr || null;

    // Open + pin the patient panel. The live ECG monitor stays hidden until
    // the resident orders an ECG investigation — see the ECG result handler.
    setPatientPanelHR(hr);
    _stopPatientEcg();
    const ppEcgWrap = document.getElementById('ppEcgWrap');
    if (ppEcgWrap) ppEcgWrap.hidden = true;
    if (typeof wbOpenPanel === 'function') {
      wbOpenPanel('panel-patient');
      wbPinPanel('panel-patient');
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
      document.getElementById('submitCaseBtn').hidden = false;
      window.speechSynthesis && window.speechSynthesis.cancel();
      _ttsEnabled = true;
      const track = document.getElementById('ttsTrack');
      if (track) track.classList.remove('off');
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

      populateCasePreview(data.case_preview);
      const voiceLbl = document.getElementById('ttsVoiceLabel');
      if (voiceLbl) voiceLbl.textContent = 'EN-US · ' + (_patientGender==='male'?'Adult Male':_patientGender==='female'?'Adult Female':'Auto');
      _chatTurnCount = 0;
      const turnBadge = document.getElementById('chatTurnBadge');
      if (turnBadge) turnBadge.hidden = true;

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
      document.querySelectorAll('#examSystems .pe-sys').forEach(r => {
        r.classList.remove('done', 'on');
      });
      const examBadge = document.getElementById('examDoneBadge');
      if (examBadge) examBadge.hidden = true;
      if (typeof peRender === 'function') peRender();
      document.getElementById('ecgContent').innerHTML = '';
      wbClosePanel('panel-ecg');
      document.getElementById('cxrImage').src = '';
      wbClosePanel('panel-cxr');
      // Imaging launchpad → reset every active card to idle, close the panel.
      if (typeof resetImagingPanel === 'function') resetImagingPanel();
      wbClosePanel('panel-imaging');
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
      // Prime the performance panel in the "In Progress" state but don't
      // auto-open it — the user opens it via its rail tab when they want.
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
    } catch(e){
      // Order failed — roll any pending Imaging cards back to idle.
      IMG_ACTIVE.forEach(m => {
        const c = _imgCard(m);
        if (c && c.dataset.state === 'pending') setImagingCardState(m, 'idle');
      });
      toast(e.message);
    } finally { spin('invBtn',false); }
  }
  function renderInv(result) {
    if(result.investigations){result.investigations.forEach(renderInv);return;}

    const vp = result.visual_payload;
    const isEcgImage = vp && vp.type === 'image' && Array.isArray(vp.leads) && vp.leads.length > 0;
    const isEcgSvg = vp && vp.svg;

    // ECG (either paper-image from PTB-XL or legacy SVG) → render in the ECG panel
    if(isEcgImage || isEcgSvg){
      if(isEcgImage){
        document.getElementById('ecgContent').innerHTML =
          `<img src="data:image/png;base64,${vp.data}" style="width:100%;display:block" alt="12-lead ECG"/>`;
      } else {
        document.getElementById('ecgContent').innerHTML = vp.svg;
      }
      const hr = _patientHR != null ? String(_patientHR) : '';
      const meta = document.getElementById('ecgMeta');
      const ecgNow = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
      document.getElementById('ecgMetaLeft').innerHTML = hr
        ? `RATE <strong>${hr}</strong> bpm &nbsp;·&nbsp; 25 mm/s &nbsp;·&nbsp; 10 mm/mV`
        : '25 mm/s &nbsp;·&nbsp; 10 mm/mV';
      document.getElementById('ecgMetaRight').innerHTML =
        `${(result.summary||'REVIEW TRACING FOR ABNORMALITIES').replace(/\.\s*$/, '').toUpperCase()} &nbsp;·&nbsp; ${ecgNow}`;
      meta.hidden = false;
      // Show the 100/500 Hz toggle only when the ECG came from PTB-XL
      // (the only path where re-rendering at a different rate is meaningful).
      const rateToggle = document.getElementById('ecgRateToggle');
      if (rateToggle) {
        const isPtbxl = vp.source_record && vp.source_record.includes('@');
        rateToggle.hidden = !isPtbxl;
        // Default selection = 500 Hz (matches the server's default render).
        rateToggle.querySelectorAll('.cb-rate-btn').forEach(b => {
          b.classList.toggle('active', parseInt(b.dataset.rate, 10) === 500);
        });
      }
      // The ECG viewer is no longer auto-opened — the user opens it by
      // clicking the (now green-bordered) ECG card in the Imaging panel.
      setImagingCardState('ecg', 'available');
      _maybeAutoOpenImagingPanel();

      // Bring the live Lead II monitor up in the Patient panel, driven by the
      // same morphology + rate the synthesizer used for the 12-lead.
      const morph = vp.lead_ii_morphology || null;
      if (morph) {
        setPatientPanelHR(morph.hr);
        setPatientPanelEcgMorphology(morph);
      }
      const ppEcgWrap = document.getElementById('ppEcgWrap');
      if (ppEcgWrap) ppEcgWrap.hidden = false;
      _initPatientEcg();

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
      document.getElementById('cxrImage').src = `data:image/jpeg;base64,${result.visual_payload.data}`;

      const finding = result.visual_payload.finding||'';
      document.getElementById('cxrFindingLabel').textContent = finding ? `Finding: ${finding}` : 'RoentGen-v2 · synth';

      const age = _patientAge || '—';
      const gender = (_patientGender || '').charAt(0).toUpperCase() || '—';
      const ovEl = document.getElementById('cxrOverlayPatient');
      if(ovEl) ovEl.innerHTML = `PATIENT<br/>MRN ----<br/>${gender} · ${age}`;

      const banner = document.getElementById('cxrFindingBanner');
      const textEl = document.getElementById('cxrFindingText');
      if(textEl) textEl.textContent = result.summary || (finding ? `Radiograph consistent with ${finding} pattern.` : '');
      if(banner) banner.hidden = !result.summary && !finding;

      // CXR viewer is no longer auto-opened — the user opens it via the
      // (now green-bordered) CXR card in the Imaging panel.
      setImagingCardState('cxr', 'available');
      _maybeAutoOpenImagingPanel();

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

  /* ── ECG sample-rate toggle (100 Hz ⇄ 500 Hz) ── */
  async function setEcgRate(rate) {
    const toggle = document.getElementById('ecgRateToggle');
    if (!toggle || toggle.hidden) return;
    const buttons = toggle.querySelectorAll('.cb-rate-btn');
    const current = toggle.querySelector('.cb-rate-btn.active');
    if (current && parseInt(current.dataset.rate, 10) === rate) return;
    buttons.forEach(b => { b.disabled = true; });
    try {
      const data = await postJson('/api/investigations/ecg-rerender', {
        session_id: sessionId,
        sample_rate_hz: rate,
      });
      document.getElementById('ecgContent').innerHTML = data.svg;
      buttons.forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.rate, 10) === rate);
      });
    } catch (e) {
      toast(e.message || 'Failed to re-render ECG');
    } finally {
      buttons.forEach(b => { b.disabled = false; });
    }
  }
  function togglePanel(id) {
    document.getElementById(id).classList.toggle('collapsed');
    if (id === 'panel-kg') setTimeout(graphFit, 300);
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
    if (!document.getElementById('panel-kg').closest('.grid-stack-item')) wbOpenPanel('panel-kg');
    document.getElementById('panel-kg').classList.add('kg-expanded');
    document.getElementById('kgBackdrop').classList.add('on');
    const btn = document.getElementById('kgExpandBtn');
    btn.title = 'Collapse';
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4.5 1v3.5H1M11 4.5H7.5V1M7.5 11V7.5H11M1 7.5h3.5V11"/></svg>`;
    setTimeout(_kgRerender, 320);
  }
  function collapseKg() {
    _kgExpanded = false;
    document.getElementById('panel-kg').classList.remove('kg-expanded');
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
    if (_examSelected.has(system)) {
      _examSelected.delete(system);
      row.classList.remove('on');
    } else {
      _examSelected.add(system);
      row.classList.add('on');
    }
    peRender();
  }

  function peSelectableRows() {
    return [...document.querySelectorAll('#examSystems .pe-sys')].filter(r => !r.classList.contains('done'));
  }

  function peToggleAll() {
    const rows = peSelectableRows();
    if (!rows.length) return;
    const allOn = rows.every(r => r.classList.contains('on'));
    rows.forEach(r => {
      const sys = r.getAttribute('data-name');
      if (allOn) { _examSelected.delete(sys); r.classList.remove('on'); }
      else       { _examSelected.add(sys);    r.classList.add('on'); }
    });
    peRender();
  }
  window.peToggleAll = peToggleAll;

  function peRender() {
    const rows = peSelectableRows();
    const allRowsCount = document.querySelectorAll('#examSystems .pe-sys').length || 9;
    const selected = [...document.querySelectorAll('#examSystems .pe-sys.on')];
    const n = selected.length;
    const countEl = document.getElementById('peCount');
    const badgeEl = document.getElementById('peBadge');
    const metaEl = document.getElementById('peMeta');
    const btn = document.getElementById('examBtn');
    const toggleAll = document.getElementById('peToggleAll');
    if (countEl) countEl.textContent = n;
    if (badgeEl) badgeEl.textContent = n;
    if (btn) btn.disabled = n === 0;
    if (metaEl) {
      if (n === 0) {
        metaEl.textContent = 'No systems selected yet.';
      } else {
        const names = selected.map(r => r.getAttribute('data-name'));
        const shown = names.slice(0, 2).join(', ').replace('&amp;', '&');
        const extra = n > 2 ? ` +${n - 2} more` : '';
        metaEl.innerHTML = `Examining <b>${n}</b> system${n > 1 ? 's' : ''} · ${shown}${extra}`;
      }
    }
    if (toggleAll) {
      // "Clear all" only when every still-selectable row is on; nothing left → disable
      if (!rows.length) {
        toggleAll.textContent = 'Select all';
        toggleAll.disabled = true;
      } else {
        toggleAll.disabled = false;
        toggleAll.textContent = rows.every(r => r.classList.contains('on')) ? 'Clear all' : 'Select all';
      }
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
      _examSelected.clear();
      document.querySelectorAll('#examSystems .pe-sys.on').forEach(el => el.classList.remove('on'));
      peRender();
    } catch(e) { toast(e.message); } finally { spin('examBtn', false); }
  }

  function renderExamResult(result) {
    const system = result.system;
    _examDone.add(system);
    _examResultMap[system] = result;
    // Per-system accent colour used by the result card icon below. Falls back
    // to teal if the system isn't in EXAM_COLORS for some reason.
    const sysColor = EXAM_COLORS[system] || 'var(--teal-600)';

    // Mark the matching .pe-sys card as done (still selected-looking, but locked).
    document.querySelectorAll('#examSystems .pe-sys').forEach(row => {
      if (row.getAttribute('data-name') === system) {
        row.classList.add('done');
        row.classList.remove('on');
      }
    });

    // Update done badge (rail-tab "X examined" pill)
    const badge = document.getElementById('examDoneBadge');
    badge.textContent = _examDone.size + ' examined';
    badge.hidden = false;

    // Add pin to body SVG if a body image is in the panel (legacy; safe no-op)
    addExamPin(system, result.abnormal);
    // Refresh the count pill / meta / button state since this row is no longer toggle-able
    peRender();

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
      el.textContent = '';
      const lbl = document.createElement('strong');
      lbl.style.fontWeight = '600';
      lbl.textContent = 'Submitted: ';
      el.appendChild(lbl);
      el.appendChild(document.createTextNode(data.diagnosis_guess));
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
    // Don't auto-open the panel — polling runs in the background, the panel
    // opens only when the user clicks its rail-tab icon.
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
      spin('submitConfirmBtn',true);
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

      wbOpenPanel('panel-perf');
      // Swap topbar: hide Pause + Submit, show Save & Exit
      hideSubmitConfirm();
      _stopTimer();
      document.getElementById('pauseBtn').hidden = true;
      document.getElementById('submitCaseBtn').hidden = true;
      document.getElementById('saveExitBtn').hidden = false;
      // Lock inputs
      document.querySelectorAll('#panel-reasoning input, #panel-reasoning textarea, #panel-reasoning button:not(.wb-pin-btn):not(.wb-fp-close)').forEach(el => { el.disabled = true; });
    } catch(e){ toast(e.message); } finally { spin('submitConfirmBtn',false); }
  }

  /* ── Workbench v3 — GridStack panel management ── */
  const WB_PANELS = {
    'panel-patient':   { x:9, y:0, w:3, h:14, minW:3, minH:8 },
    'panel-interview': { w:4, h:10, minW:3, minH:5 },
    'panel-phys-exam': { w:4, h:9,  minW:3, minH:5 },
    'panel-invest':    { w:4, h:8,  minW:3, minH:4 },
    'panel-reasoning': { w:4, h:10, minW:3, minH:5 },
    'panel-imaging':   { w:4, h:9,  minW:3, minH:5 },
    'panel-consults':  { w:4, h:8,  minW:3, minH:4 },
    'panel-kg':        { w:5, h:9,  minW:3, minH:5 },
    'panel-cxr':       { w:5, h:9,  minW:3, minH:5 },
    'panel-ecg':       { w:6, h:7,  minW:4, minH:4 },
    'panel-perf':      { w:4, h:11, minW:3, minH:5 },
  };
  let _gsGrid = null;
  const _pinnedPanels = new Set();

  function _initGrid() {
    _gsGrid = GridStack.init({
      column: 12,
      float: true,
      animate: true,
      cellHeight: 70,
      minRow: 1,
      resizable: { handles: 'se,s,e,sw,w' },
      draggable: { handle: '.wb-fp-head' },
    }, '#panelGrid');
  }

  /* ── Pin/unpin ── */
  function wbPinPanel(id) {
    _pinnedPanels.add(id);
    const panelEl = document.getElementById(id);
    if (panelEl) panelEl.classList.add('pinned');
    const btn = document.querySelector('.wb-pin-btn[data-pin-id="' + id + '"]');
    if (btn) { btn.classList.add('pinned'); btn.title = 'Unpin panel'; }
    const tab = document.querySelector('[data-panel="' + id + '"]');
    if (tab) tab.classList.add('pinned');
    const tabPinBtn = document.querySelector('.wb-tab-pin-dot[data-pin-tab="' + id + '"]');
    if (tabPinBtn) { tabPinBtn.classList.add('pinned'); tabPinBtn.title = 'Unpin panel'; }
  }
  function wbUnpinPanel(id) {
    _pinnedPanels.delete(id);
    const panelEl = document.getElementById(id);
    if (panelEl) panelEl.classList.remove('pinned');
    const btn = document.querySelector('.wb-pin-btn[data-pin-id="' + id + '"]');
    if (btn) { btn.classList.remove('pinned'); btn.title = 'Pin panel'; }
    const tab = document.querySelector('[data-panel="' + id + '"]');
    if (tab) tab.classList.remove('pinned');
    const tabPinBtn = document.querySelector('.wb-tab-pin-dot[data-pin-tab="' + id + '"]');
    if (tabPinBtn) { tabPinBtn.classList.remove('pinned'); tabPinBtn.title = 'Pin panel'; }
  }
  function wbTogglePin(id) {
    _pinnedPanels.has(id) ? wbUnpinPanel(id) : wbPinPanel(id);
  }

  /* ── Open / close ── */
  function wbOpenPanel(id) {
    // Close any open panel that is not pinned (single-panel mode)
    document.querySelectorAll('#panelGrid .grid-stack-item').forEach(gsi => {
      const pid = gsi.dataset.gsId;
      if (pid && pid !== id && !_pinnedPanels.has(pid)) wbClosePanel(pid);
    });
    const panelEl = document.getElementById(id);
    if (!panelEl || panelEl.closest('.grid-stack-item')) return;
    const cfg = WB_PANELS[id] || { w:4, h:8, minW:3, minH:4 };
    const wrapper = document.createElement('div');
    wrapper.className = 'grid-stack-item';
    wrapper.dataset.gsId = id;
    const inner = document.createElement('div');
    inner.className = 'grid-stack-item-content';
    inner.appendChild(panelEl);
    wrapper.appendChild(inner);
    _gsGrid.addWidget(wrapper, cfg);
    const tip = document.getElementById('wbEmptyTip');
    if (tip) tip.hidden = true;
    const tab = document.querySelector('[data-panel="' + id + '"]');
    if (tab) { tab.classList.add('active'); tab.classList.remove('has-update','has-alert'); }
    if (id === 'panel-kg') setTimeout(graphFit, 300);
  }
  function wbClosePanel(id) {
    const panelEl = document.getElementById(id);
    if (!panelEl) return;
    const gsiEl = panelEl.closest('.grid-stack-item');
    if (!gsiEl) return;
    // Also unpin on manual close
    wbUnpinPanel(id);
    if (id === 'panel-patient') _stopPatientEcg();
    document.getElementById('gs-store').appendChild(panelEl);
    _gsGrid.removeWidget(gsiEl);
    if (!document.querySelector('#panelGrid .grid-stack-item')) {
      const tip = document.getElementById('wbEmptyTip');
      if (tip) tip.hidden = false;
    }
    const tab = document.querySelector('[data-panel="' + id + '"]');
    if (tab) tab.classList.remove('active');
  }
  function wbTogglePanel(id) {
    const panelEl = document.getElementById(id);
    if (!panelEl) return;
    panelEl.closest('.grid-stack-item') ? wbClosePanel(id) : wbOpenPanel(id);
  }
  function wbOpenPanelKeepInvestigations(id) {
    // When a visual result (ECG/CXR) opens, keep panel-invest visible alongside it.
    // Temporarily pin it so wbOpenPanel's non-pinned sweep doesn't close it, then
    // unpin (same tick — no repaint) so the user can still close it manually.
    const invPanel = document.getElementById('panel-invest');
    const invOpen = !!(invPanel && invPanel.closest('.grid-stack-item'));
    const invWasPinned = _pinnedPanels.has('panel-invest');
    if (invOpen && !invWasPinned) wbPinPanel('panel-invest');
    wbOpenPanel(id);
    if (invOpen && !invWasPinned) wbUnpinPanel('panel-invest');
  }
  document.querySelectorAll('.wb-rail-tab').forEach(tab => {
    tab.addEventListener('click', () => wbTogglePanel(tab.dataset.panel));
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (_kgExpanded) collapseKg();
      document.querySelectorAll('#panelGrid .grid-stack-item').forEach(gsi => {
        const pid = gsi.dataset.gsId;
        if (pid) wbClosePanel(pid);
      });
    }
  });

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
    if (document.getElementById('submitBackdrop').classList.contains('on')) {
      if (e.key === 'Escape') { e.preventDefault(); hideSubmitConfirm(); }
      if (e.key === 'Enter')  { e.preventDefault(); confirmSubmitCase(); }
      return;
    }
    if (!document.getElementById('pauseBackdrop').classList.contains('on')) return;
    if (e.key === 'Enter')  { e.preventDefault(); resumeFromPause(); }
    if (e.key === 'Escape') { e.preventDefault(); location.href = '/home'; }
  });

  /* ── Patient panel: live Lead II ECG monitor ── */
  // Sweep-monitor canvas with a Lead II morphology derived from a 0..1 cardiac
  // phase. Rate-driven (beat interval = 60 / _ppBPM), resize-aware via
  // ResizeObserver, respects prefers-reduced-motion. Morphology mirrors the
  // synthesizer's per-lead findings — see backend/ecg_synthesizer._beat_template.
  let _ppBPM = 75;
  let _ppRAF = null;
  let _ppRO = null;
  let _ppCtx = null, _ppDPR = 1, _ppCssW = 0, _ppCssH = 0;
  let _ppVals = [];
  let _ppHeadF = 0, _ppLast = 0, _ppLastBeatIdx = -1;
  let _ppInited = false;
  const _ppReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Default Lead II morphology — overwritten by setPatientPanelEcgMorphology
  // once a real ECG is generated. Phase fractions follow the design prototype.
  let _ppMorph = {
    pr_frac: 0.18, qrs_frac: 0.10, qt_frac: 0.40,  // fractions of one beat phase
    p_absent: false, q_path: false, t_inv: false,
    qrs: 'normal', st_mv: 0.0, amp: 1.0,
  };

  function _ppEcg(phase) {
    // Map phase 0..1 → time within the beat, with intervals scaled like the
    // synthesizer's _beat_template. Centers are derived so that the R wave
    // lands near phase ~0.27 — matching the design's monitor pacing.
    const m = _ppMorph;
    const pr = m.pr_frac, qrs = m.qrs_frac, qt = m.qt_frac;
    const p_mu = pr * 0.50;
    const q_mu = pr;
    const r_mu = pr + qrs * 0.35;
    const s_mu = pr + qrs * 0.72;
    const t_mu = pr + qt * 0.65;
    const wide = (m.qrs === 'lbbb' || m.qrs === 'rbbb' || m.qrs === 'wide');
    const r_sig = wide ? qrs * 0.10 : qrs * 0.06;
    const s_sig = wide ? qrs * 0.08 : qrs * 0.055;
    const g = (mu, sig, amp) => amp * Math.exp(-0.5 * Math.pow((phase - mu) / Math.max(sig, 1e-4), 2));

    const sc = m.amp;
    let v = 0;
    // P wave
    if (!m.p_absent) v += g(p_mu, pr * 0.12, 0.15 * sc);
    // Q
    const q_amp = m.q_path ? -0.25 * sc : -0.08 * sc;
    v += g(q_mu, qrs * 0.06, q_amp);
    // R
    v += g(r_mu, r_sig, 0.90 * sc);
    // S
    v += g(s_mu, s_sig, -0.15 * sc);
    // LBBB notched R
    if (m.qrs === 'lbbb') v += g(r_mu + qrs * 0.30, r_sig * 0.8, 0.40 * sc);
    // RBBB RSR'
    if (m.qrs === 'rbbb') v += g(s_mu + qrs * 0.35, r_sig * 0.7, 0.45 * sc);
    // WPW delta
    if (m.qrs === 'delta') v += g(q_mu - qrs * 0.15, qrs * 0.12, 0.30 * sc);
    // ST offset between QRS end and T peak
    if (Math.abs(m.st_mv) > 0.005) {
      const st_start = s_mu + qrs * 0.08;
      const st_end = t_mu - (qt - qrs) * 0.25;
      if (phase >= st_start && phase <= st_end) v += m.st_mv * 0.30;
    }
    // T wave
    const t_amp = m.t_inv ? -0.20 * sc : (0.35 * sc + m.st_mv * 0.06);
    v += g(t_mu, (qt - qrs) * 0.18, t_amp);
    return v;
  }

  function setPatientPanelEcgMorphology(m) {
    if (!m) return;
    const hr  = Math.max(20, Math.min(300, parseInt(m.hr, 10) || _ppBPM));
    const beat_ms = 60000 / hr;
    const pr_ms  = Math.max(0,  Math.min(beat_ms * 0.6, +m.pr_ms  || 160));
    const qrs_ms = Math.max(40, Math.min(beat_ms * 0.4, +m.qrs_ms || 90));
    const qt_ms  = Math.max(qrs_ms + 40, Math.min(beat_ms * 0.95, +m.qt_ms || 420));
    _ppMorph = {
      pr_frac:  pr_ms  / beat_ms,
      qrs_frac: qrs_ms / beat_ms,
      qt_frac:  qt_ms  / beat_ms,
      p_absent: !!m.p_absent,
      q_path:   !!m.q_path,
      t_inv:    !!m.t_inv,
      qrs:      m.qrs || 'normal',
      st_mv:    +m.st_mv || 0,
      amp:      Math.max(0.1, Math.min(2, +m.amp_scale || 1)),
    };
    if (m.rhythm_label) {
      const r = document.getElementById('ppRhythm');
      if (r) r.textContent = String(m.rhythm_label).toUpperCase();
    }
    // If we're already streaming, redraw the static reference cycle so reduced-
    // motion users see the new morphology immediately.
    if (_ppReduced) { _ppDrawGrid(); _ppRenderStatic(); }
  }

  function _ppResizeCanvas() {
    const canvas = document.getElementById('ppEcg');
    if (!canvas) return;
    const wrap = canvas.parentElement;
    if (!wrap) return;
    _ppDPR = Math.min(window.devicePixelRatio || 1, 2);
    _ppCssW = Math.max(1, wrap.clientWidth);
    _ppCssH = Math.max(1, wrap.clientHeight);
    canvas.width  = Math.round(_ppCssW * _ppDPR);
    canvas.height = Math.round(_ppCssH * _ppDPR);
    _ppCtx = canvas.getContext('2d');
    _ppCtx.setTransform(_ppDPR, 0, 0, _ppDPR, 0, 0);
    _ppVals = new Array(Math.ceil(_ppCssW)).fill(NaN);
    _ppHeadF = 0; _ppLastBeatIdx = -1;
    _ppDrawGrid();
    if (_ppReduced) _ppRenderStatic();
  }

  function _ppDrawGrid() {
    if (!_ppCtx) return;
    _ppCtx.clearRect(0, 0, _ppCssW, _ppCssH);
    _ppCtx.lineWidth = 1;
    _ppCtx.strokeStyle = 'rgba(52,231,182,0.06)';
    _ppCtx.beginPath();
    for (let x = 0; x <= _ppCssW; x += 11) { _ppCtx.moveTo(x, 0); _ppCtx.lineTo(x, _ppCssH); }
    for (let y = 0; y <= _ppCssH; y += 11) { _ppCtx.moveTo(0, y); _ppCtx.lineTo(_ppCssW, y); }
    _ppCtx.stroke();
    _ppCtx.strokeStyle = 'rgba(52,231,182,0.11)';
    _ppCtx.beginPath();
    for (let x = 0; x <= _ppCssW; x += 55) { _ppCtx.moveTo(x, 0); _ppCtx.lineTo(x, _ppCssH); }
    for (let y = 0; y <= _ppCssH; y += 55) { _ppCtx.moveTo(0, y); _ppCtx.lineTo(_ppCssW, y); }
    _ppCtx.stroke();
  }

  function _ppRenderStatic() {
    // Reduced-motion: render a single representative cycle across the strip.
    if (!_ppCtx) return;
    const sweepSec = 4.2;
    const beat = 60 / Math.max(20, _ppBPM);
    const speed = _ppCssW / sweepSec;
    const mid = _ppCssH * 0.52, scale = _ppCssH * 0.34;
    _ppCtx.lineJoin = 'round'; _ppCtx.lineCap = 'round';
    _ppCtx.lineWidth = 2.1;
    _ppCtx.strokeStyle = '#34e7b6';
    _ppCtx.shadowColor = 'rgba(52,231,182,0.55)';
    _ppCtx.shadowBlur = 7;
    _ppCtx.beginPath();
    for (let x = 0; x < _ppCssW; x++) {
      const t = x / speed;
      const phase = ((t / beat) % 1 + 1) % 1;
      const y = mid - _ppEcg(phase) * scale;
      if (x === 0) _ppCtx.moveTo(x, y); else _ppCtx.lineTo(x, y);
    }
    _ppCtx.stroke();
    _ppCtx.shadowBlur = 0;
  }

  function _ppFrame(now) {
    let dt = (now - _ppLast) / 1000; _ppLast = now;
    if (dt > 0.05) dt = 0.05;
    const sweepSec = 4.2;
    const speed = _ppCssW / sweepSec;
    const beat = 60 / Math.max(20, _ppBPM);
    const cols = _ppVals.length;
    const newHeadF = _ppHeadF + speed * dt;
    for (let c = Math.ceil(_ppHeadF); c < newHeadF; c++) {
      const col = ((c % cols) + cols) % cols;
      const time = c / speed;
      const phase = (((time / beat) % 1) + 1) % 1;
      _ppVals[col] = _ppEcg(phase);
      for (let e = 1; e <= 9; e++) _ppVals[(col + e) % cols] = NaN;
    }
    _ppHeadF = newHeadF;
    const headCol = ((Math.floor(_ppHeadF) % cols) + cols) % cols;

    const beatIdx = Math.floor((_ppHeadF / speed) / beat);
    const phaseNow = ((((_ppHeadF / speed) / beat) % 1) + 1) % 1;
    if (phaseNow > 0.24 && phaseNow < 0.30 && beatIdx !== _ppLastBeatIdx) {
      _ppLastBeatIdx = beatIdx;
      const heart = document.getElementById('ppHeart');
      if (heart) {
        heart.classList.add('beat');
        setTimeout(() => heart.classList.remove('beat'), 90);
      }
    }

    _ppDrawGrid();
    const mid = _ppCssH * 0.52, scale = _ppCssH * 0.34;
    _ppCtx.lineJoin = 'round'; _ppCtx.lineCap = 'round';
    _ppCtx.lineWidth = 2.1;
    _ppCtx.strokeStyle = '#34e7b6';
    _ppCtx.shadowColor = 'rgba(52,231,182,0.55)';
    _ppCtx.shadowBlur = 7;
    _ppCtx.beginPath();
    let pen = false;
    for (let x = 0; x < cols; x++) {
      const v = _ppVals[x];
      if (v == null || isNaN(v)) { pen = false; continue; }
      const y = mid - v * scale;
      if (!pen) { _ppCtx.moveTo(x, y); pen = true; } else _ppCtx.lineTo(x, y);
    }
    _ppCtx.stroke();
    _ppCtx.shadowBlur = 0;

    const hv = _ppVals[(headCol - 1 + cols) % cols];
    if (hv != null && !isNaN(hv)) {
      const y = mid - hv * scale;
      _ppCtx.fillStyle = '#bafff0';
      _ppCtx.shadowColor = 'rgba(52,231,182,0.9)';
      _ppCtx.shadowBlur = 10;
      _ppCtx.beginPath(); _ppCtx.arc(headCol, y, 2.6, 0, Math.PI * 2); _ppCtx.fill();
      _ppCtx.shadowBlur = 0;
    }

    _ppRAF = requestAnimationFrame(_ppFrame);
  }

  function _initPatientEcg() {
    const canvas = document.getElementById('ppEcg');
    if (!canvas) return;
    // Lazy first-time setup
    if (!_ppInited) {
      _ppResizeCanvas();
      if (typeof ResizeObserver !== 'undefined') {
        _ppRO = new ResizeObserver(() => _ppResizeCanvas());
        _ppRO.observe(canvas.parentElement);
      } else {
        window.addEventListener('resize', _ppResizeCanvas);
      }
      _ppInited = true;
    } else {
      // Re-open after close: rebuild buffer at current size
      _ppResizeCanvas();
    }
    if (_ppReduced) return; // static cycle already rendered in resize
    if (_ppRAF) cancelAnimationFrame(_ppRAF);
    _ppLast = performance.now();
    _ppRAF = requestAnimationFrame(_ppFrame);
  }

  function _stopPatientEcg() {
    if (_ppRAF) { cancelAnimationFrame(_ppRAF); _ppRAF = null; }
  }

  function setPatientPanelHR(hr) {
    const n = parseInt(hr, 10);
    if (!isNaN(n) && n > 0) _ppBPM = n;
    const numEl = document.getElementById('ppHRMon');
    if (numEl) numEl.textContent = (!isNaN(n) && n > 0) ? n : '—';
  }

  /* ── Startup ── */
  _initGrid();
  (async function() {
    await checkAuth();
    const _resumeId = new URLSearchParams(window.location.search).get('resume');
    if (_resumeId) resumeCase(_resumeId);
  })();
</script>
<!--DRAWER-->
</body>
</html>
"""
