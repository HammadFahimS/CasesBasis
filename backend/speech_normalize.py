"""Speak-time normalisation of clinical shorthand for the TTS boundary.

The problem: ElevenLabs (like any TTS) pronounces clinical abbreviations and units
*character-by-character* — "SpO2" becomes "spoh-two", "mmHg" becomes "em-em-aitch-gee".
The fix belongs here, at the seam between text and speech, not in the LLM prompt:

* It is the single chokepoint every spoken line passes through (LLM actor replies,
  authored beats, cue content, vitals read-outs, nurse charting confirmations) — one
  function corrects all sources at once.
* It is deterministic — a curated table, not probabilistic model compliance, so speech
  correctness is decoupled from which LLM generates the text.
* It preserves display/speech separation: because it runs *inside* ``voice_service``'s
  synthesis calls, the transcript bubble and the chart keep the abbreviated form
  (``SpO₂ 94%``) while only the audio hears "oxygen saturation ninety-four percent".
* Context beats a generic dictionary here: ``mg`` (milligrams) vs ``Mg`` (magnesium),
  ``STEMI`` said as a word, ``120/80`` said "one twenty over eighty".

Pure stdlib (``re`` only) so it stays importable and unit-testable everywhere. Residual
*phonetic* misses (drug names, eponyms) are a separate concern — those want an ElevenLabs
Pronunciation Dictionary with phoneme rules, not this expander.
"""

from __future__ import annotations

import re

# ── ordered rules ────────────────────────────────────────────────────────────
# Applied top-to-bottom; earlier rules win. Multi-character acronyms that *contain*
# a shorter token (SpO2 ⊃ O2) MUST come before that token so we don't half-expand.

# 1 — symbols and sub/superscripts that would otherwise be dropped or misread.
_SYMBOL_RULES: list[tuple[str, str]] = [
    ("≤", " less than or equal to "),   # ≤
    ("≥", " greater than or equal to "), # ≥
    ("↑", " up "),                       # ↑ (trending up)
    ("↓", " down "),                     # ↓ (trending down)
    ("×", " by "),                        # × (e.g. "12 × 8")
    ("°", " degrees "),                    # ° when not caught by °C/°F below
]

# Compact the subscript-2 forms to plain "2" so the acronym rules below catch them.
_SUBSCRIPT_2 = ("₂", "2")   # ₂ → 2  (SpO₂ → SpO2)

# 2 — temperature (before the bare ° rule and before number handling).
_TEMP_RULES: list[tuple[str, str]] = [
    (r"°\s*C\b", " degrees Celsius"),
    (r"°\s*F\b", " degrees Fahrenheit"),
]

# 3 — clinical acronyms → spoken form. Word-boundaried, case-insensitive unless the
# casing is meaningful. Longest / most-specific first.
_ACRONYM_RULES: list[tuple[str, str]] = [
    (r"\bSpO2\b",   "oxygen saturation"),
    (r"\bSaO2\b",   "arterial oxygen saturation"),
    (r"\bPaCO2\b",  "arterial carbon dioxide"),
    (r"\bPaO2\b",   "arterial oxygen"),
    (r"\b[Ee]t?CO2\b", "end tidal carbon dioxide"),
    (r"\bCO2\b",    "carbon dioxide"),
    (r"\bO2\b",     "oxygen"),
    (r"\bBP\b",     "blood pressure"),
    (r"\bHR\b",     "heart rate"),
    (r"\bRR\b",     "respiratory rate"),
    (r"\bBPM\b",    "beats per minute"),
    (r"\bbpm\b",    "beats per minute"),
    (r"\bGCS\b",    "G C S"),
    (r"\bECG\b",    "E C G"),
    (r"\bEKG\b",    "E K G"),
    (r"\bIV\b",     "I V"),
    (r"\bIM\b",     "intramuscular"),
    (r"\bABG\b",    "A B G"),
    (r"\bJVP\b",    "J V P"),
]

# 4 — units. Compound (with a slash) before single tokens, so "mg/dL" isn't split by
# the vitals-ratio rule. Case-insensitive; these almost always trail a number.
_COMPOUND_UNIT_RULES: list[tuple[str, str]] = [
    (r"\bmg/dL\b",   "milligrams per deciliter"),
    (r"\bmg/dl\b",   "milligrams per deciliter"),
    (r"\bmEq/L\b",   "milliequivalents per liter"),
    (r"\bmmol/L\b",  "millimoles per liter"),
    (r"\bmcg/kg/min\b", "micrograms per kilogram per minute"),
    (r"\bmL/hr\b",   "milliliters per hour"),
    (r"\bmL/h\b",    "milliliters per hour"),
    (r"\bL/min\b",   "liters per minute"),
]

# 5 — vitals ratio: "120/80" → "120 over 80" (BP, I:E, etc.). Two runs of 2–3 digits.
_RATIO_RE = re.compile(r"\b(\d{2,3})/(\d{2,3})\b")

# 6 — single-token units. "L" and "%" need numeric context; the rest trail a number
# reliably enough to expand on a word boundary alone.
_UNIT_RULES: list[tuple[str, str]] = [
    (r"\bmmHg\b",  "millimeters of mercury"),
    (r"\bmcg\b",   "micrograms"),
    (r"\bµg\b", "micrograms"),   # µg
    (r"\bug\b",    "micrograms"),
    (r"\bmg\b",    "milligrams"),
    (r"\bmEq\b",   "milliequivalents"),
    (r"\bmmol\b",  "millimoles"),
    (r"\bkg\b",    "kilograms"),
    (r"\bmL\b",    "milliliters"),
    (r"\bml\b",    "milliliters"),
]

# Number-anchored units (avoid eating stray letters inside words).
_LITERS_RE = re.compile(r"(?<=\d)\s*L\b")          # "15 L" / "2L" → liters
_GRAMS_RE  = re.compile(r"(?<=\d)\s*g\b")           # "5 g" → grams
_PERCENT_RE = re.compile(r"\s*%")                    # "94%" → "94 percent"


def _compile(rules: list[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    return [(re.compile(p), r) for p, r in rules]


_SYMBOL_C   = _SYMBOL_RULES  # plain str replaces
_TEMP_C     = _compile(_TEMP_RULES)
_ACRONYM_C  = _compile(_ACRONYM_RULES)
_COMPOUND_C = _compile(_COMPOUND_UNIT_RULES)
_UNIT_C     = _compile(_UNIT_RULES)

_WS_RE = re.compile(r"[ \t]{2,}")


def normalize_for_speech(text: str) -> str:
    """Expand clinical abbreviations / units in ``text`` into their spoken form.

    Idempotent-ish and safe on plain prose — rules are word-boundaried, so ordinary
    words pass through untouched. Returns the input unchanged when it's empty."""
    if not text:
        return text

    out = text.replace(*_SUBSCRIPT_2)

    for src, dst in _TEMP_C:
        out = src.sub(dst, out)
    for src, dst in _SYMBOL_C:
        out = out.replace(src, dst)
    for src, dst in _ACRONYM_C:
        out = src.sub(dst, out)
    for src, dst in _COMPOUND_C:
        out = src.sub(dst, out)

    out = _RATIO_RE.sub(r"\1 over \2", out)

    for src, dst in _UNIT_C:
        out = src.sub(dst, out)
    out = _LITERS_RE.sub(" liters", out)
    out = _GRAMS_RE.sub(" grams", out)
    out = _PERCENT_RE.sub(" percent", out)

    return _WS_RE.sub(" ", out).strip()
