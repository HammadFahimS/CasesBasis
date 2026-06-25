"""Live voice-room routes: session start/resume, the autonomous-clock event
WebSocket, the grounded turn endpoints (nurse/action/patient), debrief,
pause/unpause, run listing/deletion, TTS, and the session knowledge graph.

Extracted from ``web_app`` (F3). The engine remains the single source of clinical
truth; these routes only carry audio/text and grounded turns to and from it.
``web_app`` mounts ``router`` and re-exports the handlers + the ``_voice_*`` /
``_case_*`` helpers so existing call sites and tests keep working.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

# Per-speaker TTS voices (each distinct so the room is tellable apart by ear) live
# in the cast registry, which resolves them from env at call time. The nurse
# default lives in voice_service; a rejected voice falls back there automatically.
from backend.actors import speaker_voice as _speaker_voice
from backend.auth import get_current_resident_id
from backend.sim_core import _get_owned_session
from backend.web_deps import limiter, require_resident

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/graph/{session_id}")
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


@router.post("/api/voice/session")
@limiter.limit("30/minute")
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


@router.post("/api/voice/resume")
@limiter.limit("30/minute")
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


@router.websocket("/ws/voice/{session_id}")
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


@router.post("/api/voice/stt")
@limiter.limit("60/minute")
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


@router.post("/api/voice/llm")
@limiter.limit("60/minute")
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


@router.post("/api/voice/action")
@limiter.limit("60/minute")
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


@router.post("/api/voice/llm-patient")
@limiter.limit("60/minute")
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


@router.post("/api/voice/debrief")
@limiter.limit("30/minute")
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


@router.post("/api/voice/pause")
@limiter.limit("60/minute")
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


@router.post("/api/voice/unpause")
@limiter.limit("60/minute")
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


@router.get("/api/voice/sessions")
@limiter.limit("60/minute")
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


@router.delete("/api/voice/sessions/{session_id}")
@limiter.limit("60/minute")
def api_voice_session_delete(session_id: str, request: Request) -> dict[str, Any]:
    """Delete one of the resident's voice runs (owner-scoped)."""
    resident_id = require_resident(request)
    from backend import voice_persistence, voice_session
    if not voice_persistence.delete(session_id, resident_id):
        raise HTTPException(status_code=404, detail="session not found")
    voice_session._SESSIONS.pop(session_id, None)   # drop any live copy too
    return {"ok": True}


@router.post("/api/voice/tts")
@limiter.limit("60/minute")
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


@router.get("/api/voice/sample")
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
