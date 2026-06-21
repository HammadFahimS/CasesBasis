"""Load and list authored simulated dynamic cases.

Cases live as JSON documents under the repo-root ``cases/`` directory and are
validated against :class:`schemas.simulated_case.SimulatedCase` on load. This
is intentionally file-backed for now; a future CRUD layer can swap the storage
for the database without changing the :class:`SimulatedCase` contract or the
engine that consumes it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from schemas.simulated_case import SimulatedCase

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


class CaseNotFoundError(KeyError):
    """Raised when a requested case id has no JSON document."""


def _case_path(case_id: str) -> Path:
    # Guard against path traversal — case ids are flat slugs.
    if "/" in case_id or "\\" in case_id or case_id.startswith("."):
        raise CaseNotFoundError(case_id)
    return CASES_DIR / f"{case_id}.json"


@lru_cache(maxsize=64)
def load_case(case_id: str) -> SimulatedCase:
    """Load and validate one simulated case by id."""
    path = _case_path(case_id)
    if not path.is_file():
        raise CaseNotFoundError(case_id)
    data = json.loads(path.read_text())
    return SimulatedCase.model_validate(data)


def list_cases() -> list[dict[str, object]]:
    """Return a lightweight catalogue (id + meta) for every authored case."""
    catalogue: list[dict[str, object]] = []
    if not CASES_DIR.is_dir():
        return catalogue
    for path in sorted(CASES_DIR.glob("*.json")):
        try:
            case = load_case(path.stem)
        except Exception:
            # Skip malformed documents rather than breaking the whole listing.
            continue
        catalogue.append(
            {
                "id": case.id,
                "title": case.meta.title,
                "specialty": case.meta.specialty,
                "difficulty": case.meta.difficulty,
                "estimated_minutes": case.meta.estimated_minutes,
                "keywords": case.meta.keywords,
                "summary": case.meta.summary,
            }
        )
    return catalogue


# Seed-oriented aliases used by the DB repository. The "seed" names make intent
# explicit at call sites (these read the authored JSON files, not the DB).
load_seed_case = load_case
list_seed_cases = list_cases
