"""Question Bank API — UWorld-style practice over the seeded ``qbank_cases``.

All handlers are owner-scoped via ``require_resident`` and rate-limited like the
rest of the app. Content is read-only (served from the DB by
``backend.qbank_store``); only per-resident attempts are written
(``backend.qbank_persistence``). Attempt correctness is resolved server-side from
the stored case so the performance stats can't be spoofed by the client.

Mounted + re-exported by ``web_app`` alongside the other route modules.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend import qbank_persistence, qbank_store
from backend.web_deps import limiter, require_resident

router = APIRouter()

_MAX_SET = 100  # cap a single practice set so a "select all" stays manageable


@router.get("/api/qbank/facets")
def api_qbank_facets(http_request: Request) -> dict[str, Any]:
    """Filter options (subject/system/subtopic/exam/difficulty) with case counts,
    plus the unfiltered totals — everything the session builder needs to render."""
    require_resident(http_request)
    return {"facets": qbank_store.facets(), "totals": qbank_store.count_matching({})}


@router.post("/api/qbank/count")
@limiter.limit("120/minute")
def api_qbank_count(body: dict[str, Any], request: Request) -> dict[str, int]:
    """How many cases / checkpoints match the builder's current filters."""
    require_resident(request)
    return qbank_store.count_matching(body.get("filters") or {})


@router.post("/api/qbank/session")
@limiter.limit("60/minute")
def api_qbank_session(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Assemble a practice set: full case documents matching the filters.

    Returns the cases (stem + ordered checkpoints + every choice's explanation —
    the reveal needs them) capped at ``_MAX_SET``. Choice-order randomisation and
    Tutor/Test gating happen on the client.
    """
    resident_id = require_resident(request)
    filters = body.get("filters") or {}
    limit = min(int(body.get("limit") or _MAX_SET), _MAX_SET)
    rows = qbank_store.list_matching(filters, limit=limit)
    case_ids = [r["case_id"] for r in rows]
    cases = qbank_store.get_cases(case_ids)
    if not cases:
        raise HTTPException(status_code=404, detail="no cases match these filters")
    # Which checkpoints the resident has already answered (UWorld "seen" hint).
    seen = qbank_persistence.answered_checkpoint_ids(resident_id)
    return {
        "mode": (body.get("mode") or "tutor"),
        "cases": cases,
        "seen_checkpoint_ids": sorted(seen),
    }


@router.post("/api/qbank/attempt")
@limiter.limit("240/minute")
def api_qbank_attempt(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Record one answered checkpoint. Correctness is resolved server-side from
    the stored case (the client sends only which choice index it picked)."""
    resident_id = require_resident(request)
    case_id = str(body.get("case_id") or "")
    checkpoint_id = str(body.get("checkpoint_id") or "")
    chosen_index = body.get("chosen_index")
    mode = str(body.get("mode") or "tutor")

    case = qbank_store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    checkpoint = next(
        (c for c in case.get("checkpoints", []) if c.get("checkpoint_id") == checkpoint_id),
        None,
    )
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")

    choices = checkpoint.get("choices", [])
    try:
        idx = int(chosen_index)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="chosen_index required")
    if not (0 <= idx < len(choices)):
        raise HTTPException(status_code=422, detail="chosen_index out of range")

    correct = bool(choices[idx].get("correct"))
    correct_index = next((i for i, c in enumerate(choices) if c.get("correct")), None)
    tags = case.get("tags") or {}
    qbank_persistence.record_attempt(
        resident_id=resident_id,
        case_id=case_id,
        checkpoint_id=checkpoint_id,
        correct=correct,
        chosen_index=idx,
        mode=mode,
        subject=str(tags.get("subject") or ""),
        system=str(tags.get("system") or ""),
    )
    return {
        "correct": correct,
        "correct_index": correct_index,
        "triggers_remediation": bool(checkpoint.get("triggers_remediation")),
    }


@router.get("/api/qbank/stats")
def api_qbank_stats(http_request: Request) -> dict[str, Any]:
    """Cumulative performance for the resident: overall + by subject and system."""
    resident_id = require_resident(http_request)
    return qbank_persistence.stats(resident_id)
