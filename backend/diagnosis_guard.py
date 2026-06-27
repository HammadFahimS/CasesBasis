"""Centralized guardrails against LLM-generated diagnosis leaks.

The student is supposed to figure out the diagnosis themselves. Every AI-facing
service that produces patient-visible text needs to scrub two failure modes:

1. **Name leaks** — the LLM mentions the diagnosis literally ("myocardial
   infarction") or by a common synonym/abbreviation ("heart attack", "MI",
   "STEMI", "ACS", "cardiac event").
2. **Interpretation leaks** — the LLM editorializes with phrases like
   "consistent with", "suggestive of", "probably", "raises concern for", etc.
   Even if the diagnosis word isn't named, "S3 gallop, suggesting LV failure"
   tells the student exactly what to think.

This module provides:
- An alias-aware `hidden_terms()` for word/phrase substitution
- A list of `INTERPRETIVE_PATTERNS` that signal LLM editorializing
- `scrub_finding_text()` — the full pipeline
- `contains_diagnosis_leak()` — boolean check for tests/logging

All services that produce student-visible text (physical exam, doctor consult,
investigations) should funnel their LLM output through `scrub_finding_text`
before persisting or returning to the client.
"""

from __future__ import annotations

import re

# ── Alias dictionary ─────────────────────────────────────────────────────────
#
# Each entry is a "concept" — a set of phrases that all refer to the same
# clinical condition. If the case's diagnosis matches any phrase in a concept
# (substring or exact match), every phrase in that concept becomes a hidden
# term.
#
# Keep entries lowercase. Word-boundary matching is added at substitution time
# (so two-letter abbreviations like "MI" or "PE" only fire on whole-word hits).

_DIAGNOSIS_CONCEPTS: list[set[str]] = [
    # ── Cardiac ──────────────────────────────────────────────────────────
    {
        "myocardial infarction", "heart attack", "mi", "stemi", "nstemi",
        "acute coronary syndrome", "acs", "cardiac event",
        "anterior mi", "inferior mi", "lateral mi", "posterior mi",
        "anterior stemi", "inferior stemi", "lateral stemi", "posterior stemi",
        "anteroseptal mi", "inferolateral mi", "infarction",
        "non-st elevation myocardial infarction",
        "st elevation myocardial infarction",
    },
    {
        "unstable angina", "preinfarction angina", "crescendo angina",
        "ua",
    },
    {
        "heart failure", "chf", "congestive heart failure",
        "cardiac failure", "left ventricular failure", "lvf",
        "right ventricular failure", "rvf", "biventricular failure",
        "decompensated heart failure",
    },
    {"atrial fibrillation", "afib", "atrial fib", "af with rvr"},
    {"atrial flutter", "aflutter", "atrial flut"},
    {"ventricular tachycardia", "vt", "v-tach", "vtach"},
    {"ventricular fibrillation", "vf", "v-fib", "vfib"},
    {"supraventricular tachycardia", "svt", "psvt", "paroxysmal svt"},
    {
        "aortic dissection", "ascending dissection", "descending dissection",
        "aortic tear", "type a dissection", "type b dissection",
    },
    {
        "abdominal aortic aneurysm", "aaa", "ruptured aaa", "leaking aaa",
        "ruptured abdominal aortic aneurysm",
    },
    {
        "pericarditis", "acute pericarditis", "pericardial inflammation",
        "viral pericarditis",
    },
    {"cardiac tamponade", "tamponade", "pericardial tamponade"},
    {
        "hypertensive emergency", "hypertensive crisis",
        "malignant hypertension", "hypertensive urgency",
    },
    {
        "complete heart block", "third-degree av block", "3rd degree av block",
        "third degree heart block", "complete av block", "chb",
    },
    {"wolff-parkinson-white", "wpw", "wpw syndrome", "pre-excitation"},
    # ── Pulmonary ────────────────────────────────────────────────────────
    {
        "pulmonary embolism", "pe", "lung clot", "blood clot in lung",
        "massive pe", "submassive pe", "saddle pe",
    },
    {
        "pneumonia", "lung infection", "chest infection", "lobar pneumonia",
        "bronchopneumonia", "community-acquired pneumonia", "cap",
        "hospital-acquired pneumonia", "hap", "aspiration pneumonia",
    },
    {
        "pneumothorax", "ptx", "tension pneumothorax", "collapsed lung",
        "simple pneumothorax", "spontaneous pneumothorax",
    },
    {
        "asthma", "asthma exacerbation", "bronchospasm",
        "acute asthma", "severe asthma", "status asthmaticus",
    },
    {
        "copd", "chronic obstructive pulmonary disease",
        "emphysema exacerbation", "copd exacerbation", "aecopd",
    },
    {"pleural effusion", "pleural fluid"},
    # ── Neuro ────────────────────────────────────────────────────────────
    {
        "stroke", "cerebrovascular accident", "cva", "brain attack",
        "ischemic stroke", "embolic stroke", "thrombotic stroke",
        "acute ischemic stroke", "ais",
    },
    {
        "hemorrhagic stroke", "intracerebral hemorrhage", "ich",
        "intracranial hemorrhage", "brain bleed", "intraparenchymal hemorrhage",
    },
    {
        "subarachnoid hemorrhage", "sah", "ruptured aneurysm",
        "ruptured berry aneurysm",
    },
    {
        "transient ischemic attack", "tia", "mini-stroke", "mini stroke",
    },
    {
        "meningitis", "bacterial meningitis", "viral meningitis",
        "meningococcal meningitis",
    },
    {"seizure", "status epilepticus", "convulsion", "tonic-clonic seizure"},
    # ── GI / abdominal ───────────────────────────────────────────────────
    {
        "appendicitis", "acute appendicitis", "appendix inflammation",
        "ruptured appendix",
    },
    {
        "cholecystitis", "gallbladder inflammation", "acute cholecystitis",
        "biliary colic", "gallstones",
    },
    {
        "pancreatitis", "acute pancreatitis", "necrotizing pancreatitis",
    },
    {
        "bowel obstruction", "small bowel obstruction", "sbo",
        "large bowel obstruction", "lbo", "intestinal obstruction",
        "mechanical obstruction",
    },
    {
        "diverticulitis", "acute diverticulitis",
        "complicated diverticulitis",
    },
    {
        "peptic ulcer", "peptic ulcer disease", "pud",
        "perforated ulcer", "perforated peptic ulcer",
    },
    {
        "mesenteric ischemia", "acute mesenteric ischemia", "ami",
    },
    {
        "ectopic pregnancy", "ruptured ectopic", "tubal pregnancy",
    },
    # ── Endo / metabolic ─────────────────────────────────────────────────
    {
        "diabetic ketoacidosis", "dka", "diabetic acidosis", "ketoacidosis",
    },
    {
        "hyperosmolar hyperglycemic state", "hhs",
        "hyperosmolar hyperglycemic syndrome", "hyperosmolar coma",
        "hyperglycemic hyperosmolar non-ketotic", "hhnk",
    },
    {
        "hypoglycemia", "low blood sugar", "insulin shock",
        "hypoglycemic episode",
    },
    {"thyroid storm", "thyrotoxic crisis", "thyrotoxicosis"},
    {"addisonian crisis", "adrenal crisis", "acute adrenal insufficiency"},
    # ── Infection / sepsis ───────────────────────────────────────────────
    {
        "sepsis", "septic shock", "septicemia", "blood poisoning",
        "severe sepsis",
    },
    {
        "urinary tract infection", "uti", "pyelonephritis",
        "acute pyelonephritis", "kidney infection",
    },
    # ── Allergic / immune ────────────────────────────────────────────────
    {
        "anaphylaxis", "anaphylactic shock", "severe allergic reaction",
        "anaphylactic reaction",
    },
    # ── Renal / urological ───────────────────────────────────────────────
    {
        "acute kidney injury", "aki", "acute renal failure", "arf",
    },
    {
        "kidney stone", "renal calculus", "renal colic", "ureteric stone",
        "nephrolithiasis", "urolithiasis",
    },
    # ── Hematological ────────────────────────────────────────────────────
    {
        "deep vein thrombosis", "dvt", "venous thromboembolism", "vte",
        "leg clot", "leg dvt",
    },
]


# ── Interpretive-phrase patterns ─────────────────────────────────────────────
#
# Sentences containing any of these phrases are LLM editorializing. We strip
# them so the student sees only observations, never the AI's reasoning.
# Patterns are case-insensitive at match time.

_INTERPRETIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bconsistent\s+with\b",
        r"\bsuggestive\s+of\b",
        r"\bsuggesting\b",
        r"\bsuggests\b",
        r"\bindicative\s+of\b",
        r"\bin\s+keeping\s+with\b",
        r"\bcompatible\s+with\b",
        r"\bappears?\s+to\s+be\b",
        r"\bseems?\s+(?:like|to\s+be)\b",
        r"\bcould\s+(?:be|represent)\b",
        r"\bmay\s+represent\b",
        r"\bmight\s+be\b",
        r"\b(?:most\s+)?likely\b",
        r"\bprobably\b",
        r"\bprobable\b",
        r"\bconcerning\s+for\b",
        r"\braises?\s+concern\s+for\b",
        r"\bdifferential\s+(?:diagnosis|includes?|consideration)\b",
        r"\bclinical\s+picture\s+(?:is|of|suggests?|consistent)\b",
        r"\bpresentation\s+(?:is|suggests?)\b",
        r"\bworking\s+diagnosis\b",
        r"\bprovisional\s+diagnosis\b",
        r"\b(?:this\s+)?looks\s+like\b",
        r"\bpoints?\s+(?:to|towards?)\b",
        r"\bin\s+favou?r\s+of\b",
        r"\bpathognomonic\s+(?:for|of)\b",
        # "diagnosis", "diagnose", "diagnoses" all leak the AI's reasoning.
        r"\bdiagnos(?:is|es|ed|ing|tic)\b",
    ]
]


# ── Public API ──────────────────────────────────────────────────────────────

def hidden_terms(diagnosis: str) -> list[str]:
    """Return all forbidden phrases for the given diagnosis.

    Always includes the literal diagnosis string. If the diagnosis maps to one
    of our known concepts, every alias in that concept is added too. Empty
    input returns an empty list.
    """
    literal = diagnosis.strip()
    if not literal:
        return []
    lower = literal.lower()
    out: list[str] = [literal]
    for concept in _DIAGNOSIS_CONCEPTS:
        if any(_concept_match(lower, term) for term in concept):
            out.extend(concept)
            break  # First matching concept wins (avoids cross-concept pollution).
    # De-dupe while preserving order; case-insensitive comparison.
    seen: set[str] = set()
    dedup: list[str] = []
    for term in out:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            dedup.append(term)
    return dedup


def _concept_match(diagnosis_lower: str, term: str) -> bool:
    """Whether the given concept term applies to this diagnosis.

    Matches with word boundaries on both sides so short abbreviations like
    "pe" or "mi" don't fire inside longer words ("ap**pe**ndicitis",
    "**mi**ningitis"). Either direction of containment counts: the term may
    appear *in* the diagnosis ("stemi" inside "anterior stemi"), or the
    diagnosis may appear *in* the term ("acs" matching the canonical
    "acute coronary syndrome").
    """
    t = term.lower().strip()
    if not t:
        return False
    # Term inside diagnosis (e.g. "stemi" in "anterior stemi")
    if re.search(rf"\b{re.escape(t)}\b", diagnosis_lower):
        return True
    # Diagnosis inside term (e.g. "acs" matches the canonical longer alias)
    if re.search(rf"\b{re.escape(diagnosis_lower)}\b", t):
        return True
    return False


def contains_diagnosis_leak(text: str, diagnosis: str) -> bool:
    """Return True if `text` mentions the diagnosis (literal or alias) OR
    contains any interpretive phrase that editorializes for the student.
    """
    if not text:
        return False
    lower = text.lower()
    for term in hidden_terms(diagnosis):
        if re.search(_term_pattern(term), text, flags=re.IGNORECASE):
            return True
    return any(pat.search(lower) for pat in _INTERPRETIVE_PATTERNS)


def scrub_finding_text(text: str, diagnosis: str) -> str:
    """Apply the full guard pipeline.

    1. Substitute every hidden term (literal + aliases) with "the condition".
    2. Drop any sentence containing an interpretive phrase — preserves the
       observation portion of mixed-content findings like
       "S3 gallop at the apex, suggesting LV failure" → "S3 gallop at the apex".
    """
    if not text:
        return text
    cleaned = text
    for term in hidden_terms(diagnosis):
        if len(term.strip()) < 2:
            continue
        cleaned = re.sub(
            _term_pattern(term),
            "the condition",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = _strip_interpretive_clauses(cleaned)
    return cleaned.strip()


# ── Internals ───────────────────────────────────────────────────────────────

def _term_pattern(term: str) -> str:
    """Word-boundary regex for substituting a hidden term."""
    escaped = re.escape(term.strip())
    if re.fullmatch(r"[A-Za-z0-9 \-]+", term.strip()):
        return rf"\b{escaped}\b"
    return escaped


_CLAUSE_SPLIT_RE = re.compile(r"[,;.!?—–]|\s+[-]\s+")


def _strip_interpretive_clauses(text: str) -> str:
    """Drop any sentence/clause that contains an interpretive phrase.

    Splits on all clause-separating punctuation (periods, semicolons, commas,
    em/en-dashes, hyphen-spaces) so a half-interpretive sentence like
    "S3 gallop at the apex, suggesting LV failure" loses just the interpretive
    half. Rejoins safe clauses with ", " — slightly flattens nested
    punctuation but preserves the clinical meaning.
    """
    parts = _CLAUSE_SPLIT_RE.split(text)
    kept = [p.strip() for p in parts if p.strip() and not _has_interpretive(p)]
    if not kept:
        # Everything was interpretive — return empty so caller can detect leak.
        return ""
    out = ", ".join(kept).strip().rstrip(",.;—– -")
    return out + "." if out else ""


def _has_interpretive(text: str) -> bool:
    return any(pat.search(text) for pat in _INTERPRETIVE_PATTERNS)


__all__ = [
    "hidden_terms",
    "contains_diagnosis_leak",
    "scrub_finding_text",
]
