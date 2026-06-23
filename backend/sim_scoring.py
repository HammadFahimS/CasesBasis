"""Score a voice-sim run against the case — a small, reusable assessment seam.

The engine owns *what happened*; this module judges *how well*. It reads the
finished (or in-progress) run and emits a flat list of :class:`Criterion` — one
per thing the case cares about. The debrief renders them as a checklist, and
later phases (medication correctness, analytics) add criteria here rather than
bolting scoring onto each feature.

Pure and deterministic (no LLM, no DB) so it's unit-tested offline. Each scorer
fails *soft*: a criterion it can't evaluate is simply reported as unmet with a
plain reason, never an exception.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class Criterion:
    """One graded expectation for the run."""

    id: str
    label: str
    met: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())


def _score_recognition(session) -> Criterion:
    """Did the resident commit a working diagnosis that names the condition?

    Matched against the case's own diagnostic keywords (``meta.keywords``), which
    are the author's canonical names for the condition (e.g. "Myasthenic Crisis").
    Substring match on normalised text, so "myasthenia gravis" and "myasthenic
    crisis" both count.
    """
    case = session.case
    said = _norm(" ".join(session.engine.run.working_diagnoses))
    targets = [k for k in (case.meta.keywords or []) if k]
    hit = next((k for k in targets if _norm(k) and _norm(k) in said), None)
    if hit:
        return Criterion("recognition", "Recognised the diagnosis", True, f"Named “{hit}”.")
    if not said.strip():
        return Criterion("recognition", "Recognised the diagnosis", False, "No working diagnosis was stated.")
    return Criterion("recognition", "Recognised the diagnosis", False, "Working diagnosis didn't match the target condition.")


def _score_no_critical_error(session) -> Criterion:
    """Did the run avoid the case's authored critical-error paths?"""
    errors = list(session.engine.run.critical_errors)
    if errors:
        return Criterion("safety", "Avoided critical errors", False, f"{len(errors)} critical error(s).")
    return Criterion("safety", "Avoided critical errors", True, "No critical errors.")


# Scorers run in order; each returns exactly one Criterion. Add new scorers here
# (e.g. medication correctness) without touching callers.
_SCORERS = (
    _score_recognition,
    _score_no_critical_error,
)


def score_run(session) -> list[Criterion]:
    """Assess a run and return the criteria list (order is stable for the UI)."""
    out: list[Criterion] = []
    for scorer in _SCORERS:
        try:
            out.append(scorer(session))
        except Exception:  # a scorer must never break the debrief
            continue
    return out
