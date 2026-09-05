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


_ROUTE_RE = re.compile(
    r"\b(iv|im|io|po|sc|sl|pr|subcut(?:aneous)?|intramuscular(?:ly)?|intravenous(?:ly)?|"
    r"intraosseous|nebuli[sz]ed|neb|inhaled|topical)\b",
    re.IGNORECASE,
)


def extract_route(*texts: str) -> str | None:
    """The route word in an order ("IM", "IV"), if one was stated."""
    for text in texts:
        match = _ROUTE_RE.search(text or "")
        if match:
            return match.group(1).lower()
    return None


def _tokens(text: str) -> list[str]:
    # numbers keep their decimals so "0.1 mg" never reads as "1 mg"
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", (text or "").lower())


def dose_phrase_matches(text: str, phrases: list[str]) -> bool:
    """Whether any dose phrase appears in ``text`` as a whole-token sequence —
    "1 mg iv" matches "epinephrine 1 mg IV push" but not "0.1 mg IV"."""
    toks = _tokens(text)
    for phrase in phrases:
        want = _tokens(phrase)
        n = len(want)
        if n and any(toks[i:i + n] == want for i in range(len(toks) - n + 1)):
            return True
    return False


def intercept_item(case, med: "Med", state_id: str):
    """The medication-rubric item that says the nurse must withhold this order
    in this state, or None. An ``avoid`` item with ``intercept`` applies when the
    drug matches, the state is in scope, and (if it names dose phrases) one of them
    is in the order."""
    text = f"{med.drug} {med.detail} {med.dose or ''}".lower()
    for item in getattr(case, "medication_rubric", None) or []:
        if not getattr(item, "intercept", False) or item.expectation != "avoid":
            continue
        if item.states and state_id not in item.states:
            continue
        if not any(d.lower() in text for d in item.drugs if d):
            continue
        if item.dose_keywords and not dose_phrase_matches(text, item.dose_keywords):
            continue
        return item
    return None


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
