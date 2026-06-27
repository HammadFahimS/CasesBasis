"""Dashboard + session lifecycle routes: history, stats, coaching insight,
session detail/review/delete, and pause/resume.

Extracted from ``web_app`` (F3). Reads classic-simulator state via
``backend.sim_core`` and is owner-scoped via ``require_resident`` /
``_get_owned_session``. ``web_app`` mounts ``router`` and re-exports the handlers
+ ``PauseRequest`` so existing call sites and tests keep working.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.auth import get_current_resident_id
from backend.persistence import (
    delete_session_for_resident,
    get_recent_evaluations_for_coaching,
    get_resident_sessions,
    get_resident_stats,
    get_session_evaluation,
    get_session_for_resident,
    load_session,
    log_event,
    pause_session,
)
from backend.sim_core import (
    SESSIONS,
    _full_state_response,
    _get_owned_session,
    get_graph,
)
from backend.web_deps import require_resident

logger = logging.getLogger(__name__)

router = APIRouter()


class PauseRequest(BaseModel):
    elapsed_seconds: int = 0


@router.get("/api/dashboard/sessions")
def dashboard_sessions(http_request: Request) -> dict[str, Any]:
    resident_id = require_resident(http_request)
    from backend import voice_persistence
    sessions = get_resident_sessions(resident_id)
    sessions += voice_persistence.get_resident_evaluations(resident_id)
    # newest first across both sources (voice runs carry source="voice")
    sessions.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return {"sessions": sessions}


@router.get("/api/dashboard/stats")
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


@router.get("/api/dashboard/coaching-insight")
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


@router.get("/api/dashboard/session/{session_id}")
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


@router.get("/api/dashboard/voice-review/{session_id}")
def dashboard_voice_review(session_id: str, http_request: Request) -> dict[str, Any]:
    """The stored assessment for one concluded voice run (analytics review drawer)."""
    resident_id = require_resident(http_request)
    from backend import voice_persistence
    review = voice_persistence.get_evaluation(session_id, resident_id)
    if review is None:
        raise HTTPException(status_code=404, detail="voice run not found")
    return review


@router.delete("/api/dashboard/session/{session_id}")
def dashboard_delete_session(session_id: str, http_request: Request) -> dict[str, bool]:
    resident_id = require_resident(http_request)
    deleted = delete_session_for_resident(session_id, resident_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    SESSIONS.pop(session_id, None)
    log_event(session_id, "session.deleted", resident_id=resident_id)
    return {"ok": True}


@router.post("/api/sessions/{session_id}/pause")
def pause(session_id: str, body: PauseRequest, http_request: Request) -> dict[str, bool]:
    _get_owned_session(session_id, http_request)  # 404/403 if not the caller's
    graph = get_graph()
    snapshot = graph.get_context(session_id) if graph and graph.enabled else None
    pause_session(session_id, snapshot, body.elapsed_seconds)
    log_event(session_id, "session.paused", {"elapsed_seconds": body.elapsed_seconds})
    return {"ok": True}


@router.post("/api/sessions/{session_id}/resume")
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
