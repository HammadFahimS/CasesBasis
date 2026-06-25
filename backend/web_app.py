"""ASGI composition root for the Clinical Case Simulator.

After F3 this file is intentionally thin: it builds the FastAPI app, wires
startup/shutdown (lifespan), CORS and the rate limiter, then mounts the route
modules in ``backend.routers.*`` and re-exports their handlers, request models
and the ``backend.sim_core`` session-core symbols under their historical names
so existing imports and tests that reference them as ``web_app`` attributes keep
working. The only route defined inline is the ``/health`` ops check.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.db import run_migrations

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
from backend.routers import pages as _pages_router  # noqa: E402
from backend.routers import simulator as _simulator_router  # noqa: E402
from backend.routers import voice as _voice_router  # noqa: E402

app.include_router(_auth_router.router)
app.include_router(_cases_router.router)
app.include_router(_dashboard_router.router)
app.include_router(_pages_router.router)
app.include_router(_simulator_router.router)
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
from backend.routers.pages import (  # noqa: E402,F401
    _inject_drawer,
    _serve_with_chrome,
    analytics_page,
    body_image,
    crud_simulations_editor_page,
    crud_simulations_page,
    dashboard_redirect,
    ecg_algorithm_page,
    favicon,
    home_page,
    index,
    simulated_intro_page,
    simulator_page,
    voice_room_page,
)
from backend.routers.simulator import (  # noqa: E402,F401
    DiagnosisRequest,
    DoctorConsultRequest,
    ECGAlgorithmRequest,
    ECGRerenderRequest,
    EvaluateRequest,
    GenerateCaseRequest,
    InterpretationRequest,
    InvestigationRequest,
    QuestionRequest,
    ask_question,
    call_doctor,
    evaluate,
    generate_case,
    order_investigation,
    physical_exam,
    render_synthetic_ecg,
    rerender_ecg,
    submit_diagnosis,
    submit_interpretation,
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


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

