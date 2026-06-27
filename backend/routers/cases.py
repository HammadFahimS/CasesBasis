"""CRUD API for authored simulated dynamic cases (the graph-editor backend).

Extracted from ``web_app`` (F3). All handlers are owner-scoped via
``require_resident``; the repository layer (``backend.simulated_case_repo``)
holds the actual persistence and gold-case protection rules.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.web_deps import limiter, require_resident

router = APIRouter()


@router.get("/api/simulated-cases")
def api_list_simulated_cases(http_request: Request) -> dict[str, Any]:
    require_resident(http_request)
    from backend.simulated_case_repo import list_cases as _list
    return {"cases": _list()}


@router.get("/api/simulated-cases/{case_id}")
def api_get_simulated_case(case_id: str, http_request: Request) -> dict[str, Any]:
    require_resident(http_request)
    from backend.simulated_case_repo import get_case
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@router.post("/api/simulated-cases")
@limiter.limit("30/minute")
def api_create_simulated_case(body: dict[str, Any], request: Request) -> dict[str, Any]:
    resident_id = require_resident(request)
    from backend.simulated_case_repo import CaseValidationError, create_case
    try:
        return create_case(body.get("case", {}), body.get("layout", {}), created_by=resident_id)
    except CaseValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.put("/api/simulated-cases/{case_id}")
@limiter.limit("60/minute")
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


@router.delete("/api/simulated-cases/{case_id}")
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


@router.post("/api/simulated-cases/{case_id}/duplicate")
@limiter.limit("30/minute")
def api_duplicate_simulated_case(case_id: str, request: Request) -> dict[str, Any]:
    resident_id = require_resident(request)
    from backend.simulated_case_repo import duplicate_case
    try:
        return duplicate_case(case_id, created_by=resident_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")
