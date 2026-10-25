"""ElevenLabs Pronunciation Dictionary — phonetic pass for the hard words.

The speak-time normaliser (:mod:`backend.speech_normalize`) fixes *abbreviations and
units* deterministically. This module handles the other class of TTS miss: words that
are spelled out in full but still mispronounced — drug names and eponyms (``prasugrel``,
``Kussmaul``, ``Guillain-Barré``). Those want a phonetic respelling applied by the voice
engine itself, which is exactly what an ElevenLabs Pronunciation Dictionary is for
(suggested by ElevenLabs when we hit the units issue).

How it works:

* :data:`RULES` is our curated lexicon — ``alias`` rules (a targeted respelling the
  engine speaks in place of the term). Alias rules are model-agnostic, so they apply on
  the Eleven v4 Turbo default *and* a Flash v2.5 / ``eleven_v3`` rollback. (Phoneme
  rules only work on ElevenLabs' v2 English models — v4 takes inline ``/IPA/`` instead —
  so we deliberately don't rely on them here.)
* :func:`tts_locators` returns the ``pronunciation_dictionary_locators`` list to attach to
  a TTS request. It resolves the dictionary once per process and caches it: an env-pinned
  id wins; otherwise we reuse an existing dictionary of the same name on the account;
  otherwise we create it from :data:`RULES`. Every failure path degrades to ``[]`` so TTS
  proceeds exactly as before — the dictionary is an enhancement, never a dependency.

Pin ``ELEVENLABS_PRONUNCIATION_DICTIONARY_ID`` (and optionally ``…_VERSION_ID``) in the
env to skip resolution entirely and avoid a first-turn lookup.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.elevenlabs.io/v1"
_DICT_NAME = "casesbasis-clinical"

# Curated respellings for terms ElevenLabs commonly mangles. Keep the alias close to how a
# clinician says the word; capitalised syllables mark the stress. Extend freely — new rows
# take effect on the next dictionary (re)build (delete the account dictionary or bump the
# name to force a rebuild after editing).
RULES: list[dict[str, str]] = [
    # ── antiplatelets / anticoagulants / thrombolytics ──
    {"string_to_replace": "prasugrel",     "type": "alias", "alias": "PRASS-oo-grell"},
    {"string_to_replace": "ticagrelor",    "type": "alias", "alias": "tye-CAG-rel-or"},
    {"string_to_replace": "tirofiban",     "type": "alias", "alias": "tye-roe-FYE-ban"},
    {"string_to_replace": "eptifibatide",  "type": "alias", "alias": "ep-tih-FYE-bah-tide"},
    {"string_to_replace": "dabigatran",    "type": "alias", "alias": "dab-ih-GAT-ran"},
    {"string_to_replace": "rivaroxaban",   "type": "alias", "alias": "riv-ah-ROX-ah-ban"},
    {"string_to_replace": "apixaban",      "type": "alias", "alias": "a-PIX-ah-ban"},
    {"string_to_replace": "edoxaban",      "type": "alias", "alias": "ee-DOX-ah-ban"},
    {"string_to_replace": "enoxaparin",    "type": "alias", "alias": "ee-nox-ah-PAR-in"},
    {"string_to_replace": "tenecteplase",  "type": "alias", "alias": "ten-ECK-teh-plase"},
    {"string_to_replace": "alteplase",     "type": "alias", "alias": "AL-teh-plase"},
    # ── cardiac / resus ──
    {"string_to_replace": "amiodarone",    "type": "alias", "alias": "am-ee-OH-dah-rone"},
    {"string_to_replace": "adenosine",     "type": "alias", "alias": "ah-DEN-oh-seen"},
    {"string_to_replace": "labetalol",     "type": "alias", "alias": "lah-BET-ah-lol"},
    {"string_to_replace": "metoprolol",    "type": "alias", "alias": "meh-TOE-proe-lol"},
    {"string_to_replace": "dobutamine",    "type": "alias", "alias": "doe-BYOO-tah-meen"},
    {"string_to_replace": "noradrenaline", "type": "alias", "alias": "nor-ah-DREN-ah-lin"},
    # ── neuro / NMJ (myasthenia gold case) ──
    {"string_to_replace": "levetiracetam", "type": "alias", "alias": "leh-veh-tih-RASS-eh-tam"},
    {"string_to_replace": "pyridostigmine","type": "alias", "alias": "peer-id-oh-STIG-meen"},
    {"string_to_replace": "neostigmine",   "type": "alias", "alias": "nee-oh-STIG-meen"},
    {"string_to_replace": "edrophonium",   "type": "alias", "alias": "ed-roe-FOH-nee-um"},
    {"string_to_replace": "rocuronium",    "type": "alias", "alias": "roe-cure-OH-nee-um"},
    # ── airway / foreign body ──
    {"string_to_replace": "Magill",        "type": "alias", "alias": "muh-GILL"},
    {"string_to_replace": "laryngoscopy",  "type": "alias", "alias": "lar-in-GOSS-kuh-pee"},
    {"string_to_replace": "laryngoscope",  "type": "alias", "alias": "luh-RING-guh-scope"},
    {"string_to_replace": "GlideScope",    "type": "alias", "alias": "GLIDE-scope"},
    {"string_to_replace": "amlodipine",    "type": "alias", "alias": "am-LOE-dih-peen"},
    {"string_to_replace": "atorvastatin",  "type": "alias", "alias": "a-TOR-va-stat-in"},
    {"string_to_replace": "suxamethonium", "type": "alias", "alias": "sux-ah-meh-THOH-nee-um"},
    {"string_to_replace": "succinylcholine","type": "alias","alias": "sux-in-ill-KOH-leen"},
    # ── abx / gi / other ──
    {"string_to_replace": "ceftriaxone",   "type": "alias", "alias": "sef-try-AX-one"},
    {"string_to_replace": "piperacillin",  "type": "alias", "alias": "pip-er-ah-SILL-in"},
    {"string_to_replace": "tazobactam",    "type": "alias", "alias": "taz-oh-BAK-tam"},
    {"string_to_replace": "vancomycin",    "type": "alias", "alias": "van-koe-MY-sin"},
    {"string_to_replace": "pantoprazole",  "type": "alias", "alias": "pan-TOE-prah-zole"},
    {"string_to_replace": "furosemide",    "type": "alias", "alias": "fur-OH-seh-mide"},
    {"string_to_replace": "naloxone",      "type": "alias", "alias": "nah-LOX-one"},
    # ── eponyms / signs ──
    {"string_to_replace": "Kussmaul",      "type": "alias", "alias": "KOOS-mowl"},
    {"string_to_replace": "Guillain-Barré","type": "alias", "alias": "GEE-yan bar-RAY"},
    {"string_to_replace": "Guillain-Barre","type": "alias", "alias": "GEE-yan bar-RAY"},
    {"string_to_replace": "Cheyne-Stokes", "type": "alias", "alias": "chain stokes"},
    {"string_to_replace": "Wernicke",      "type": "alias", "alias": "VER-nih-keh"},
    {"string_to_replace": "Wenckebach",    "type": "alias", "alias": "VENK-eh-bock"},
    {"string_to_replace": "Brugada",       "type": "alias", "alias": "broo-GAH-dah"},
    {"string_to_replace": "Takotsubo",     "type": "alias", "alias": "tah-koh-TSOO-boh"},
    {"string_to_replace": "pheochromocytoma","type": "alias","alias": "fee-oh-kroe-moe-sy-TOE-mah"},
]

# Cached resolution: unset → not tried; None → tried and unavailable; dict → the locator.
_UNSET = object()
_locator = _UNSET


def _elevenlabs_key() -> str | None:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip() or None


def _locator_from(pd_id: str | None, version_id: str | None) -> dict | None:
    if not pd_id:
        return None
    loc = {"pronunciation_dictionary_id": pd_id}
    if version_id:
        loc["version_id"] = version_id
    return loc


def _find_existing(key: str) -> dict | None:
    """Reuse a same-named dictionary already on the account (avoids duplicates on restart)."""
    resp = httpx.get(
        f"{_BASE_URL}/pronunciation-dictionaries",
        headers={"xi-api-key": key},
        params={"page_size": 100},
        timeout=10.0,
    )
    resp.raise_for_status()
    for d in resp.json().get("pronunciation_dictionaries", []):
        if d.get("name") == _DICT_NAME:
            return _locator_from(d.get("id"), d.get("latest_version_id") or d.get("version_id"))
    return None


def _create(key: str) -> dict | None:
    resp = httpx.post(
        f"{_BASE_URL}/pronunciation-dictionaries/add-from-rules",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={"name": _DICT_NAME, "rules": RULES},
        timeout=15.0,
    )
    resp.raise_for_status()
    body = resp.json()
    loc = _locator_from(body.get("id"), body.get("version_id"))
    if loc:
        logger.info("Created ElevenLabs pronunciation dictionary %s (%d rules). Pin "
                    "ELEVENLABS_PRONUNCIATION_DICTIONARY_ID=%s to skip lookup on restart.",
                    loc["pronunciation_dictionary_id"], len(RULES),
                    loc["pronunciation_dictionary_id"])
    return loc


def _resolve() -> dict | None:
    """Resolve the dictionary locator once. Env pin > reuse existing > create. Never raises."""
    env_id = os.environ.get("ELEVENLABS_PRONUNCIATION_DICTIONARY_ID", "").strip()
    if env_id:
        return _locator_from(env_id, os.environ.get("ELEVENLABS_PRONUNCIATION_VERSION_ID", "").strip() or None)
    key = _elevenlabs_key()
    if not key:
        return None
    try:
        return _find_existing(key) or _create(key)
    except Exception as exc:   # network / auth / quota — PD is optional, degrade quietly
        logger.warning("Pronunciation dictionary unavailable (continuing without it): %s", exc)
        return None


def tts_locators() -> list[dict]:
    """The ``pronunciation_dictionary_locators`` to attach to a TTS request (``[]`` if none).

    Resolved lazily and cached for the process. Safe to call on every synthesis."""
    global _locator
    if _locator is _UNSET:
        _locator = _resolve()
    return [_locator] if _locator else []
