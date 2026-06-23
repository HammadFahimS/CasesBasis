"""Parse a spoken medication order into a structured, renderable form.

A resident says "give rocuronium 0.6 milligrams per kilo" or just "push some
succinylcholine". The nurse router classifies that as ``order_medication`` and
hands us the drug (``target``) and any dose/route paraphrase (``action_text``).
This module is the deterministic seam that turns that into a :class:`Med` — drug,
dose, detail, and whether it's a *high-alert* agent whose dose the nurse should
confirm before administration.

Pure and dependency-free (no LLM, no DB) so it's unit-tested offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# High-alert drug families whose dose the nurse confirms before giving. The
# motivating case is neuromuscular blockade in myasthenia gravis, where the
# paralytic dose is the crux (succinylcholine resistance, rocuronium sensitivity),
# but these are genuinely high-alert agents in any resuscitation.
HIGH_ALERT = (
    "succinylcholine", "suxamethonium", "rocuronium", "vecuronium",
    "cisatracurium", "pancuronium", "atracurium",            # paralytics
    "ketamine", "etomidate", "propofol", "midazolam",        # induction/sedation
    "epinephrine", "adrenaline", "norepinephrine",           # vasoactive
    "insulin", "heparin",                                    # classic high-alert
)

# A dose token: a number with a recognised unit (optionally per-kilo), e.g.
# "1.5 mg/kg", "0.6 milligrams per kilo", "100 mcg", "5 units".
_UNIT = r"(?:mg|milligrams?|micrograms?|mcg|microg|g|grams?|units?|iu|ml|millilitres?|milliliters?|mmol|meq)"
_PER_KG = r"(?:\s*(?:/|per)\s*(?:kg|kilo(?:gram)?s?))?"
_DOSE_RE = re.compile(
    rf"\b\d+(?:\.\d+)?\s*{_UNIT}{_PER_KG}\b",
    re.IGNORECASE,
)


@dataclass
class Med:
    """A parsed medication order."""

    drug: str
    dose: str | None = None
    detail: str = ""
    high_alert: bool = False

    @property
    def slug(self) -> str:
        """A stable card id for this drug, so re-orders upsert rather than dup."""
        return "med-" + re.sub(r"[^a-z0-9]+", "-", self.drug.lower()).strip("-")


def _extract_dose(*texts: str) -> str | None:
    for text in texts:
        if not text:
            continue
        match = _DOSE_RE.search(text)
        if match:
            return match.group(0).strip()
    return None


def _is_high_alert(drug: str) -> bool:
    low = drug.lower()
    return any(name in low for name in HIGH_ALERT)


def parse_med(target: str | None, action_text: str = "") -> Med:
    """Split a routed medication order into drug / dose / detail.

    ``target`` is the router's best guess at the drug; ``action_text`` is the
    fuller paraphrase (which usually carries the dose/route). Both are
    best-effort — we never reject an order, we just record what we could read.
    """
    detail = (action_text or "").strip()
    drug = (target or "").strip()
    if not drug:
        # No clean drug target — take the leading words of the paraphrase, minus
        # any dose token, as the drug name.
        stripped = _DOSE_RE.sub("", detail).strip()
        drug = stripped or detail or "medication"
    dose = _extract_dose(detail, drug)
    return Med(drug=drug, dose=dose, detail=detail, high_alert=_is_high_alert(drug))


def needs_dose_confirm(med: Med) -> bool:
    """Whether the nurse should confirm the dose before administering.

    True for high-alert agents — the dose-confirm loop is the teaching point for
    paralytics in myasthenic crisis, and a sensible safety habit generally."""
    return med.high_alert
