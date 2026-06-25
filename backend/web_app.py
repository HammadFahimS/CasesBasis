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
# Classic-simulator session core (state + service singletons + student-facing
# rendering) lives in backend.sim_core; re-export so the classic routes still in
# this file — and external importers (persistence, simulation_runner) — resolve
# them unchanged.
from backend.sim_core import (  # noqa: E402,F401
    CASE_CACHE,
    CASE_CACHE_TTL_SECONDS,
    SESSIONS,
    WebSession,
    cache_key,
    case_preview,
    clean_lab_value,
    get_doctor_consult_service,
    get_evaluator,
    get_graph,
    get_investigation_service,
    get_or_generate_cached_case,
    get_patient_service,
    get_physical_exam_agent,
    get_session,
    get_simulator,
    is_cxr_display,
    is_ecg_display,
    raw_investigation_report,
    redact_hidden_terms,
    student_facing_investigation_result,
    _full_state_response,
    _get_owned_session,
)

from backend.routers import auth as _auth_router  # noqa: E402
from backend.routers import cases as _cases_router  # noqa: E402
from backend.routers import dashboard as _dashboard_router  # noqa: E402
from backend.routers import voice as _voice_router  # noqa: E402

app.include_router(_auth_router.router)
app.include_router(_cases_router.router)
app.include_router(_dashboard_router.router)
app.include_router(_voice_router.router)

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
from backend.routers.dashboard import (  # noqa: E402,F401
    PauseRequest,
    dashboard_coaching_insight,
    dashboard_delete_session,
    dashboard_session_detail,
    dashboard_sessions,
    dashboard_stats,
    dashboard_voice_review,
    pause,
    resume,
)
from backend.routers.voice import (  # noqa: E402,F401
    _case_consultants,
    _case_roster,
    _live_runtime_enabled,
    _persist_turn,
    _turn_payload,
    _voice_session_payload,
    api_voice_action,
    api_voice_debrief,
    api_voice_llm,
    api_voice_patient,
    api_voice_pause,
    api_voice_sample,
    api_voice_session,
    api_voice_session_delete,
    api_voice_sessions,
    api_voice_stt,
    api_voice_tts,
    api_voice_unpause,
    api_voice_resume,
    get_graph_data,
    ws_voice,
)


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


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


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

