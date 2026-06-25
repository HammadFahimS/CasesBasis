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

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pathlib import Path

from pydantic import BaseModel, field_validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

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

    # Autonomous clock for the live voice sim (F1+F2). No-op unless LIVE_RUNTIME is
    # set, so existing deployments are unaffected until we opt in.
    from backend.session_runtime import start_runtime, stop_runtime
    await start_runtime()

    yield

    await stop_runtime()
    _faiss_index._model = None
    gc.collect()


# Shared web primitives live in the leaf web_deps module so routers can import
# them without an import cycle; re-exported here under their historical names.
from backend.web_deps import limiter as _limiter  # noqa: E402
from backend.web_deps import require_resident, set_auth_cookie as _set_auth_cookie  # noqa: E402

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

# ── Mounted route modules (F3) ─────────────────────────────────────────────────
# Routes extracted from this file live in backend.routers.*; mount them here and
# re-export their handlers + request models so existing imports and tests that
# reference them as web_app attributes keep working.
from backend.routers import auth as _auth_router  # noqa: E402
from backend.routers import cases as _cases_router  # noqa: E402

app.include_router(_auth_router.router)
app.include_router(_cases_router.router)

from backend.routers.auth import (  # noqa: E402,F401
    RequestLinkRequest,
    ResidentLoginRequest,
    login_resident,
    logout,
    me,
    request_magic_link,
    signup_resident,
    test_login,
    verify_magic_link,
)
from backend.routers.cases import (  # noqa: E402,F401
    api_create_simulated_case,
    api_delete_simulated_case,
    api_duplicate_simulated_case,
    api_get_simulated_case,
    api_list_simulated_cases,
    api_update_simulated_case,
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
    return _inject_drawer((Path(__file__).parent / "static" / "simulator.html").read_text())


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
        # Whether the autonomous clock is running — the client opens the live event
        # WebSocket only when it is, so the default (runtime-off) config stays quiet.
        "live_runtime": _live_runtime_enabled(),
    }


def _live_runtime_enabled() -> bool:
    from backend.session_runtime import runtime_enabled
    return runtime_enabled()


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


@app.websocket("/ws/voice/{session_id}")
async def ws_voice(websocket: WebSocket, session_id: str) -> None:
    """Live event channel for one session: streams the autonomous-clock events
    (deterioration, cues, state changes, case end) the Session Runtime publishes.

    Auth is the same signed cookie as the REST routes, and the socket only opens
    for the resident who owns the session. ``?since=<seq>`` replays missed events
    after a reconnect (the client dedups by ``seq``). Server→client only for now;
    turn replies still flow over the REST POST."""
    import asyncio

    from backend.session_runtime import get_bus
    from backend.voice_session import AccessDenied, CaseNotFound, get_session

    resident_id = get_current_resident_id(websocket)
    if resident_id is None:
        await websocket.close(code=4401)  # unauthenticated
        return
    try:
        # Authorises ownership and ensures the session is live (so the runtime ticks
        # it); the DB hit on a cold session runs off the event loop.
        await asyncio.to_thread(get_session, session_id, resident_id)
    except (CaseNotFound, AccessDenied):
        await websocket.close(code=4403)  # not found / not yours
        return

    await websocket.accept()
    bus = get_bus()
    queue = bus.subscribe(session_id)
    try:
        since = websocket.query_params.get("since")
        if since is not None:
            try:
                for ev in bus.replay(session_id, int(since)):
                    await websocket.send_json(ev)
            except ValueError:
                pass
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(session_id, queue)


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
