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
    """One graded expectation for the run.

    Two flavours, distinguished by ``guardrail``:

    * **credit** (``guardrail=False``) — something the resident must *actively do*
      to earn marks (name the diagnosis, give the right drug). ``weight`` is its
      share of the positive score.
    * **guardrail** (``guardrail=True``) — a pitfall to avoid (a critical error, a
      contraindicated drug). It is *penalty-only*: passing it earns nothing (you
      don't get points for simply not making a mistake), and failing it subtracts
      ``weight`` points. This is what stops a do-nothing run from banking free
      marks for "avoided errors".
    """

    id: str
    label: str
    met: bool
    detail: str = ""
    weight: int = 1
    guardrail: bool = False

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
        return Criterion("recognition", "Recognised the diagnosis", True, f"Named “{hit}”.", weight=REC_WEIGHT)
    if not said.strip():
        return Criterion("recognition", "Recognised the diagnosis", False, "No working diagnosis was stated.", weight=REC_WEIGHT)
    return Criterion("recognition", "Recognised the diagnosis", False, "Working diagnosis didn't match the target condition.", weight=REC_WEIGHT)


def _order_texts(session) -> list[str]:
    """Normalised text of every medication the resident ordered.

    Drawn from the Orders cards built in Phase A — each carries the drug, dose and
    the resident's own phrasing (so dose-direction words like 'reduced' survive)."""
    out = []
    for card in session.cards:
        if card.get("category") == "orders" or card.get("kind") == "order":
            out.append(_norm(f"{card.get('title', '')} {card.get('summary', '')}"))
    return out


def _avoid_violated(session, item, drugs: list[str]) -> bool:
    """An ``avoid`` item is tripped by a live intercept, or by an order carrying the
    drug (and, when the item names dose phrases, one of them). State-scoped items
    are judged on the engine's own action log, which carries each order's state."""
    from backend.medication import dose_phrase_matches
    log = session.engine.run.log
    if any(e.get("kind") == "intercept" and e.get("detail") == item.id for e in log):
        return True
    if item.states:
        texts = [str(e.get("detail") or "") for e in log
                 if e.get("kind") == "medicate" and e.get("state") in item.states]
    else:
        texts = _order_texts(session)
    for t in texts:
        if not any(d and d in _norm(t) for d in drugs):
            continue
        if not item.dose_keywords or dose_phrase_matches(t, item.dose_keywords):
            return True
    return False


def _score_medications(session) -> list[Criterion]:
    """Grade the resident's drug orders against the case's medication rubric.

    Returns one criterion per rubric item (``dose_adjust`` items are graded only
    when the drug was actually given, so choosing the other agent isn't punished).
    Cases without a ``medication_rubric`` yield nothing — fully backward-compatible.
    """
    rubric = getattr(session.case, "medication_rubric", None) or []
    texts = _order_texts(session)
    out: list[Criterion] = []
    for item in rubric:
        drugs = [_norm(d) for d in item.drugs if d]
        matched = [t for t in texts if any(d and d in t for d in drugs)]
        given = bool(matched)
        if item.expectation == "avoid":
            # A contraindicated drug is a pitfall, not an earnable point: penalty-only.
            given = _avoid_violated(session, item, drugs)
            out.append(Criterion(
                f"med-{item.id}", item.label, not given,
                item.note if given else "Correctly not given.",
                weight=AVOID_PENALTY, guardrail=True,
            ))
        elif item.expectation == "preferred":
            out.append(Criterion(
                f"med-{item.id}", item.label, given,
                "" if given else item.note,
                weight=MED_WEIGHT,
            ))
        elif item.expectation == "dose_adjust":
            if not given:
                continue  # not given → not applicable, skip rather than penalise
            kws = [_norm(k) for k in item.dose_keywords if k]
            tuned = any(any(k and k in t for k in kws) for t in matched)
            out.append(Criterion(
                f"med-{item.id}", item.label, tuned,
                "" if tuned else item.note,
                weight=MED_WEIGHT,
            ))
    return out


# --- scoring weights ---------------------------------------------------------
# Credit weights are relative shares of the positive score; guardrail weights are
# absolute point penalties subtracted when the pitfall is triggered.
REC_WEIGHT = 3          # naming the diagnosis is the biggest single credit
MED_WEIGHT = 2          # each preferred / correctly-dosed drug
WORKUP_WEIGHT = 2       # each expected exam / investigation performed
SAFETY_PENALTY = 40     # a critical error is severe
AVOID_PENALTY = 25      # giving a contraindicated drug


def _examined_systems(session) -> set[str]:
    """Body systems the resident examined. A full physical exam (logged as ``all``)
    counts as every documented system."""
    logged = {str(e.get("detail")) for e in session.engine.run.log if e.get("kind") == "exam"}
    if "all" in logged:
        return {e.system for e in session.case.exam}
    return logged


def _score_workup(session) -> list[Criterion]:
    """Grade the diagnostic workup against the case's ``workup_rubric``.

    One criterion per rubric item: an ``expected`` exam/investigation earns
    ``WORKUP_WEIGHT`` when any of its ``targets`` was performed/ordered; an
    ``avoid`` item is a penalty-only guardrail. Cases without a rubric yield
    nothing — fully backward-compatible."""
    rubric = getattr(session.case, "workup_rubric", None) or []
    if not rubric:
        return []
    examined = _examined_systems(session)
    ordered = set(session.engine.run.ordered_investigations)
    out: list[Criterion] = []
    for item in rubric:
        pool = examined if item.kind == "exam" else ordered
        did = any(t in pool for t in item.targets)
        if item.expectation == "avoid":
            out.append(Criterion(
                f"workup-{item.id}", item.label, not did,
                item.note if did else "Correctly not ordered.",
                weight=AVOID_PENALTY, guardrail=True,
            ))
        else:  # expected
            out.append(Criterion(
                f"workup-{item.id}", item.label, did,
                "" if did else item.note,
                weight=WORKUP_WEIGHT,
            ))
    return out


def _score_no_critical_error(session) -> Criterion:
    """Did the run avoid the case's authored critical-error paths?

    A guardrail: avoiding errors earns nothing on its own (otherwise doing nothing
    banks free marks), but triggering one costs ``SAFETY_PENALTY`` points."""
    errors = list(session.engine.run.critical_errors)
    if errors:
        return Criterion("safety", "Avoided critical errors", False, f"{len(errors)} critical error(s).",
                         weight=SAFETY_PENALTY, guardrail=True)
    return Criterion("safety", "Avoided critical errors", True, "No critical errors.",
                     weight=SAFETY_PENALTY, guardrail=True)


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
    try:
        out.extend(_score_medications(session))
    except Exception:
        pass
    try:
        out.extend(_score_workup(session))
    except Exception:
        pass
    return out


def overall_score(criteria: list[Criterion]) -> int:
    """A single 0-100 figure for analytics.

    The positive score is the weighted share of **credit** criteria the resident
    actually met (name the diagnosis, give the right drugs). **Guardrail** criteria
    don't contribute credit — they only subtract their ``weight`` when breached.

    This is what fixes the "50 for doing nothing" bug: an idle run meets no credit
    criteria (base 0) and, because it took no harmful action, breaches no guardrails
    (no penalty) — so it scores 0, not 50. Empty criteria → 0. Rounded to the
    nearest integer so it slots into the same charts as the manual-sim score."""
    if not criteria:
        return 0
    credits = [c for c in criteria if not c.guardrail]
    guards = [c for c in criteria if c.guardrail]
    if credits:
        cw = sum(max(1, c.weight) for c in credits)
        met = sum(max(1, c.weight) for c in credits if c.met)
        base = 100.0 * met / cw
    elif guards:
        base = 100.0  # nothing to actively earn — full marks unless a pitfall is hit
    else:
        return 0
    penalty = sum(c.weight for c in guards if not c.met)
    return max(0, min(100, round(base - penalty)))
