"""Question-bank content store: bundle → validate → seed → read.

The assessment content ships as a single bundled document
(``backend/data/qbank_cases.json``). On startup :func:`seed_qbank_cases` validates
every case against :class:`schemas.qbank.QBankCase` and upserts it into the
``qbank_cases`` table (the runtime source of truth, mirroring how gold simulated
cases are seeded). Malformed cases are logged and skipped — a bad entry never
breaks startup or the whole listing.

At runtime the session builder reads only the denormalised tag columns for
filtering/counting; the runner pulls a full case document (stem + checkpoints,
including every choice's explanation — the reveal needs them) via
:func:`get_case`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.db import get_db, utcnow
from schemas.qbank import QBankCase

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "data" / "qbank_cases.json"


# ── Bundled document → validated cases ───────────────────────────────────────

@lru_cache(maxsize=1)
def load_bundled_cases() -> list[QBankCase]:
    """Parse + validate the bundled JSON, dropping malformed cases (fail-soft).

    Returns the valid cases only; each rejected case is logged with the reason.
    A missing or unparseable file yields an empty list rather than raising, so a
    packaging slip degrades the feature instead of crashing the app.
    """
    try:
        raw = json.loads(DATA_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("qbank data unreadable at %s: %s", DATA_PATH, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("qbank data is not a list; ignoring")
        return []
    out: list[QBankCase] = []
    seen: set[str] = set()
    for entry in raw:
        try:
            case = QBankCase.model_validate(entry)
        except ValidationError as exc:
            cid = entry.get("case_id", "?") if isinstance(entry, dict) else "?"
            logger.warning("skipping malformed qbank case %s: %s", cid, exc)
            continue
        if case.case_id in seen:
            logger.warning("skipping duplicate qbank case id %s", case.case_id)
            continue
        seen.add(case.case_id)
        out.append(case)
    return out


# ── Seeding (startup) ────────────────────────────────────────────────────────

def seed_qbank_cases() -> int:
    """Upsert the bundled cases into ``qbank_cases``. Idempotent; returns the
    number of rows written. Safe to call on every startup.

    Unlike the insert-only gold-case seeder, this upserts: the qbank content is
    read-only reference data with no in-DB editing, so re-seeding should carry a
    corrected bundle forward. User progress lives in a separate table and is
    never touched.
    """
    cases = load_bundled_cases()
    conn = get_db()
    now = utcnow()
    written = 0
    for case in cases:
        t = case.tags
        conn.execute(
            "INSERT INTO qbank_cases "
            "(case_id, stem, subject, subtopic, system, exam, difficulty, high_yield, "
            " checkpoint_count, case_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(case_id) DO UPDATE SET "
            "  stem=excluded.stem, subject=excluded.subject, subtopic=excluded.subtopic, "
            "  system=excluded.system, exam=excluded.exam, difficulty=excluded.difficulty, "
            "  high_yield=excluded.high_yield, checkpoint_count=excluded.checkpoint_count, "
            "  case_json=excluded.case_json, updated_at=excluded.updated_at",
            (
                case.case_id, case.stem, t.subject, t.subtopic, t.system, t.exam,
                t.difficulty, 1 if t.high_yield else 0, len(case.checkpoints),
                case.model_dump_json(), now, now,
            ),
        )
        written += 1
    conn.commit()
    logger.info("seeded %d qbank cases", written)
    return written


# ── Filtering ────────────────────────────────────────────────────────────────

_ALLOWED_FILTERS = {"subject", "system", "subtopic", "exam"}


def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a parametrised WHERE clause from a filter dict.

    String filters (subject/system/subtopic/exam) accept a single value or a
    list (→ IN). ``difficulty`` accepts a single int or list of ints.
    ``high_yield`` truthy restricts to high-yield cases.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for key in _ALLOWED_FILTERS:
        val = filters.get(key)
        if val in (None, "", []):
            continue
        vals = val if isinstance(val, (list, tuple)) else [val]
        vals = [v for v in vals if v not in (None, "")]
        if not vals:
            continue
        clauses.append(f"{key} IN ({','.join('?' * len(vals))})")
        params.extend(vals)

    diff = filters.get("difficulty")
    if diff not in (None, "", []):
        dvals = diff if isinstance(diff, (list, tuple)) else [diff]
        dvals = [int(d) for d in dvals if str(d).strip() != ""]
        if dvals:
            clauses.append(f"difficulty IN ({','.join('?' * len(dvals))})")
            params.extend(dvals)

    if filters.get("high_yield"):
        clauses.append("high_yield = 1")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def count_matching(filters: dict[str, Any] | None = None) -> dict[str, int]:
    """How many cases and checkpoints match the current filters."""
    where, params = _where(filters or {})
    try:
        row = get_db().execute(
            "SELECT COUNT(*) AS cases, COALESCE(SUM(checkpoint_count), 0) AS checkpoints "
            f"FROM qbank_cases{where}",
            params,
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("qbank count failed: %s", exc)
        return {"cases": 0, "checkpoints": 0}
    return {"cases": int(row[0] or 0), "checkpoints": int(row[1] or 0)}


def list_matching(filters: dict[str, Any] | None = None, limit: int = 500) -> list[dict[str, Any]]:
    """Lightweight rows (id + tags + counts) for cases matching the filters."""
    where, params = _where(filters or {})
    try:
        rows = get_db().execute(
            "SELECT case_id, subject, subtopic, system, exam, difficulty, high_yield, "
            "       checkpoint_count FROM qbank_cases"
            f"{where} ORDER BY subject, subtopic, case_id LIMIT ?",
            [*params, limit],
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("qbank list failed: %s", exc)
        return []
    return [
        {
            "case_id": r[0], "subject": r[1], "subtopic": r[2], "system": r[3],
            "exam": r[4], "difficulty": r[5], "high_yield": bool(r[6]),
            "checkpoint_count": r[7],
        }
        for r in rows
    ]


def get_case(case_id: str) -> dict[str, Any] | None:
    """The full case document (stem + ordered checkpoints + choices) for the runner."""
    try:
        row = get_db().execute(
            "SELECT case_json FROM qbank_cases WHERE case_id=?", (case_id,)
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("qbank get_case failed for %s: %s", case_id, exc)
        return None
    if row is None:
        return None
    return json.loads(row[0])


def get_cases(case_ids: list[str]) -> list[dict[str, Any]]:
    """Full documents for a set of ids, returned in the requested order."""
    if not case_ids:
        return []
    placeholders = ",".join("?" * len(case_ids))
    try:
        rows = get_db().execute(
            f"SELECT case_id, case_json FROM qbank_cases WHERE case_id IN ({placeholders})",
            case_ids,
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("qbank get_cases failed: %s", exc)
        return []
    by_id = {r[0]: json.loads(r[1]) for r in rows}
    return [by_id[cid] for cid in case_ids if cid in by_id]


def facets() -> dict[str, list]:
    """Distinct filter values for the session builder, each with a case count."""
    conn = get_db()

    def _distinct(col: str) -> list[dict[str, Any]]:
        try:
            rows = conn.execute(
                f"SELECT {col} AS v, COUNT(*) AS n FROM qbank_cases "
                f"WHERE {col} != '' GROUP BY {col} ORDER BY {col}"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("qbank facets failed for %s: %s", col, exc)
            return []
        return [{"value": r[0], "count": r[1]} for r in rows]

    try:
        diffs = [
            {"value": r[0], "count": r[1]}
            for r in conn.execute(
                "SELECT difficulty AS v, COUNT(*) AS n FROM qbank_cases "
                "GROUP BY difficulty ORDER BY difficulty"
            ).fetchall()
        ]
    except sqlite3.Error:
        diffs = []

    return {
        "subject": _distinct("subject"),
        "system": _distinct("system"),
        "subtopic": _distinct("subtopic"),
        "exam": _distinct("exam"),
        "difficulty": diffs,
    }
