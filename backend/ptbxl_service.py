"""PTB-XL → paper-style ECG image service.

Loads PTB-XL (PhysioNet) metadata once, indexes records by SCP superclass
(NORM, MI, STTC, CD, HYP), then on demand picks a record matching a clinical
diagnosis and renders it as a paper-style PNG via ECG-Image-Kit's ecg_plot.

The service is intentionally tolerant: if the dataset or ECG-Image-Kit isn't
available, `render_paper_ecg` returns None and the caller falls back to the
legacy numpy-synthesizer SVG path.
"""

from __future__ import annotations

import base64
import logging
import random
import re
import threading
from ast import literal_eval
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PTBXL_BASE = _PROJECT_ROOT / "data" / "ptb-xl"
_PTBXL_INDEX_DIR = _PROJECT_ROOT / "data" / "ptbxl_index"

# Minimum SCP-code confidence we will trust as a match. PTB-XL stores per-code
# confidence in 0..100. ~24 % of records carry low-confidence MI labels that
# look indistinguishable from normal — those make terrible teaching examples.
_MIN_CONFIDENCE = 80.0

# Cosine-similarity threshold for accepting a FAISS retrieval. Below this we
# treat the diagnosis as effectively non-cardiac and return NORM. Calibrated
# from sampling: asthma~0.43 (too low → NORM), pericarditis~0.64 (use match).
_FAISS_MIN_SCORE = 0.50
_FAISS_TOP_N_POOL = 10  # randomly pick from top-N hits above threshold

# PTB-XL infarction_stadium1 → acuity bucket. ~97 % of MI-labelled records are
# chronic/healed and look essentially normal except for residual Q waves —
# random picking shows the doctor a "normal" ECG for "heart attack". This map
# lets us pick acute records when the diagnosis text implies acute MI.
_ACUTE_STADIA = {"Stadium I", "Stadium I-II", "Stadium II"}
_CHRONIC_STADIA = {"Stadium III", "Stadium II-III"}

# Diagnosis-text → acuity preference. Only consulted when the matched SCP
# codes are MI codes (acuity is meaningless for AFIB, blocks, etc.).
_ACUTE_DX = re.compile(
    r"\b(stemi|nstemi|acute|heart\s+attack|fresh|new\s+(onset|mi)|evolving)\b",
    re.I,
)
_CHRONIC_DX = re.compile(
    r"\b(old|previous|prior|history\s+of|remote|chronic|healed|past)\b",
    re.I,
)


def _detect_mi_acuity(diagnosis: str, preferred_codes: tuple[str, ...]) -> str | None:
    """Return 'acute' / 'chronic' / None based on diagnosis text + matched codes.

    Acuity is only meaningful for MI codes — for AFIB or bundle blocks we don't
    filter by stadium. When the diagnosis mentions an MI but doesn't qualify the
    acuity, default to 'acute' (the clinical context that brought the patient in
    is almost always acute presentation).
    """
    if not any(code in _ANY_MI for code in preferred_codes):
        return None
    if _CHRONIC_DX.search(diagnosis):
        return "chronic"
    if _ACUTE_DX.search(diagnosis):
        return "acute"
    # Generic "myocardial infarction" without a qualifier — assume acute
    # presentation because that's almost always what the case generator intends.
    return "acute"

# Anatomic territory groups inside the MI superclass.
_AMI_CODES = ("AMI", "ASMI", "ALMI", "INJAS", "INJAL", "INJLA")  # anterior wall
_IMI_CODES = ("IMI", "ILMI", "IPMI", "IPLMI", "INJIN", "INJIL")  # inferior wall
_LMI_CODES = ("LMI",)
_PMI_CODES = ("PMI",)
_ANY_MI = _AMI_CODES + _IMI_CODES + _LMI_CODES + _PMI_CODES

# Ischemia subgroups inside STTC.
_ISC_ANT = ("ISCAN", "ISCAL", "ISCAS", "ISCLA")  # anterior/lateral ischemia
_ISC_INF = ("ISCIN", "ISCIL")
_ISC_ANY = _ISC_ANT + _ISC_INF + ("ISC_",)

# Diagnosis text patterns (regex, case-insensitive) → ordered list of SCP codes
# to prefer. First-match wins. If a record exists matching the first code at
# >= _MIN_CONFIDENCE we use it; otherwise we walk the list. The order matters:
# more specific codes come before generic fall-backs so anterior STEMI prefers
# AMI/ASMI over generic MI.
_PATTERNS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    # Territory-specific MIs ─────────────────────────────────────────────────
    (re.compile(r"\bantero?\s*-?\s*septal", re.I), ("ASMI", "AMI", "INJAS") + _AMI_CODES),
    (re.compile(r"\banterolateral", re.I), ("ALMI", "INJAL", "AMI") + _AMI_CODES),
    (re.compile(r"\banterior\s+wall|anterior\s+(stemi|mi|myocardial|infarct)", re.I),
        ("AMI", "ASMI", "ALMI") + _AMI_CODES),
    (re.compile(r"\binferolateral", re.I), ("ILMI", "INJIL") + _IMI_CODES),
    (re.compile(r"\binferoposterior", re.I), ("IPMI", "IPLMI") + _IMI_CODES),
    (re.compile(r"\binferior\s+wall|inferior\s+(stemi|mi|myocardial|infarct)", re.I),
        ("IMI", "ILMI", "IPMI", "IPLMI") + _IMI_CODES),
    (re.compile(r"\blateral\s+wall|\blateral\s+(stemi|mi|myocardial|infarct)", re.I),
        ("LMI", "ALMI", "ILMI")),
    (re.compile(r"\bposterior\s+wall|\bposterior\s+(stemi|mi|myocardial|infarct)", re.I),
        ("PMI", "IPMI", "IPLMI")),

    # Generic MI / heart attack (no territory) — pick any high-confidence MI.
    # `\bmi\b` catches the abbreviation alone ("history of MI", "post-MI", …).
    (re.compile(
        r"\b(stemi|nstemi|myocardial\s+infarction|heart\s+attack|acute\s+coronary|mi)\b",
        re.I,
    ), _ANY_MI),

    # Ischemia ───────────────────────────────────────────────────────────────
    (re.compile(r"\banterior\s+(ischemia|ischaemia)", re.I), _ISC_ANT + ("ISC_",)),
    (re.compile(r"\binferior\s+(ischemia|ischaemia)", re.I), _ISC_INF + ("ISC_",)),
    (re.compile(r"\b(ischemia|ischaemia|unstable\s+angina|stable\s+angina)", re.I), _ISC_ANY),

    # Conduction disturbances ───────────────────────────────────────────────
    (re.compile(r"\bcomplete\s+left\s+bundle|\bclbbb\b", re.I), ("CLBBB",)),
    (re.compile(r"\bcomplete\s+right\s+bundle|\bcrbbb\b", re.I), ("CRBBB",)),
    (re.compile(r"\bincomplete\s+left\s+bundle|\bilbbb\b", re.I), ("ILBBB",)),
    (re.compile(r"\bincomplete\s+right\s+bundle|\birbbb\b", re.I), ("IRBBB",)),
    (re.compile(r"\bleft\s+bundle\s+branch\s+block|\blbbb\b", re.I), ("CLBBB", "ILBBB")),
    (re.compile(r"\bright\s+bundle\s+branch\s+block|\brbbb\b", re.I), ("CRBBB", "IRBBB")),
    (re.compile(r"\bbundle\s+branch\s+block", re.I), ("CLBBB", "CRBBB", "ILBBB", "IRBBB")),
    (re.compile(r"\b(third|3rd|complete)\W*degree\W*(av\s+|atrioventricular\s+)?(block|heart\s+block)", re.I),
        ("3AVB",)),
    (re.compile(r"\b(second|2nd|mobitz)\W*degree\W*(av\s+|atrioventricular\s+)?(block|heart\s+block)", re.I),
        ("2AVB",)),
    (re.compile(r"\b(first|1st)\W*degree\W*(av\s+|atrioventricular\s+)?(block|heart\s+block)", re.I),
        ("1AVB",)),
    (re.compile(r"\b(av\s+block|atrioventricular\s+block|heart\s+block)", re.I),
        ("3AVB", "2AVB", "1AVB")),
    (re.compile(r"\b(wpw|wolff[\s-]?parkinson)", re.I), ("WPW",)),
    (re.compile(r"\bleft\s+anterior\s+fascicular|\blafb\b", re.I), ("LAFB",)),
    (re.compile(r"\bleft\s+posterior\s+fascicular|\blpfb\b", re.I), ("LPFB",)),

    # Hypertrophy ───────────────────────────────────────────────────────────
    (re.compile(r"\b(lvh|left\s+ventricular\s+hypertrophy)", re.I), ("LVH",)),
    (re.compile(r"\b(rvh|right\s+ventricular\s+hypertrophy)", re.I), ("RVH",)),
    (re.compile(r"\bleft\s+atrial\s+(enlargement|overload)|\blae\b", re.I), ("LAO/LAE",)),
    (re.compile(r"\bright\s+atrial\s+(enlargement|overload)|\brae\b", re.I), ("RAO/RAE",)),
    (re.compile(r"\bseptal\s+hypertrophy|\bhocm\b|hypertrophic\s+cardiomyopathy", re.I), ("SEHYP", "LVH")),
]


def map_diagnosis_to_codes(diagnosis: str) -> tuple[str, ...]:
    """Map free-text diagnosis to an ordered tuple of preferred SCP codes.

    Returns an empty tuple when no cardiac pattern matches — caller should
    fall back to NORM (a normal ECG is reasonable for a non-cardiac diagnosis).
    """
    if not diagnosis:
        return ()
    for pattern, codes in _PATTERNS:
        if pattern.search(diagnosis):
            return codes
    return ()


# Kept for backward compatibility with older callers and tests.
def map_diagnosis_to_superclass(diagnosis: str) -> str:
    """Coarse mapping to the 5-class PTB-XL superclass."""
    codes = map_diagnosis_to_codes(diagnosis)
    if not codes:
        return "NORM"
    # Infer superclass from first code in the preferred list.
    first = codes[0]
    if first in _ANY_MI:
        return "MI"
    if first in _ISC_ANY or first.startswith("ISC") or first in {"NDT", "NST_", "LNGQT", "DIG", "ANEUR", "EL"}:
        return "STTC"
    if first in {"LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP"}:
        return "HYP"
    return "CD"


# ── Service ───────────────────────────────────────────────────────────────────

class PTBXLService:
    """Index PTB-XL once, hand out paper-style PNGs matching the diagnosis."""

    def __init__(self, base_dir: Path | None = None, *, min_confidence: float = _MIN_CONFIDENCE) -> None:
        self._base_dir = base_dir or _PTBXL_BASE
        self._data_root: Path | None = None
        # Per-SCP-code list of (record_path, confidence).
        self._code_index: dict[str, list[tuple[str, float]]] = {}
        # Per-acuity-bucket, per-SCP-code list of records — only populated for
        # MI codes since stadium tagging is meaningless elsewhere. Keys are
        # 'acute' and 'chronic'; everything else falls through to _code_index.
        self._code_index_by_acuity: dict[str, dict[str, list[tuple[str, float]]]] = {
            "acute": {},
            "chronic": {},
        }
        # Per-superclass list of high-confidence records (for NORM fallback).
        self._norm_records: list[str] = []
        self._loaded = False
        self._lock = threading.Lock()
        self._min_confidence = min_confidence

    @property
    def enabled(self) -> bool:
        return bool(self._resolve_data_root())

    def _resolve_data_root(self) -> Path | None:
        """Locate the directory containing records100/ and ptbxl_database.csv.

        Handles both the unwrapped layout (data/ptb-xl/records100/) and the
        ZIP-extracted layout (data/ptb-xl/ptb-xl-a-large-…-1.0.3/records100/).
        """
        if self._data_root is not None:
            return self._data_root
        candidates = [self._base_dir]
        if self._base_dir.exists():
            candidates += [p for p in self._base_dir.iterdir() if p.is_dir()]
        for c in candidates:
            if (c / "ptbxl_database.csv").exists() and (c / "records100").exists():
                self._data_root = c
                return c
        return None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._code_index)
        with self._lock:
            if self._loaded:
                return bool(self._code_index)
            self._loaded = True
            root = self._resolve_data_root()
            if root is None:
                logger.info("PTB-XL not present at %s — disabled", self._base_dir)
                return False
            try:
                import pandas as pd
            except ImportError:
                logger.warning("pandas not installed — PTB-XL service disabled")
                return False

            db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
            db.scp_codes = db.scp_codes.apply(literal_eval)
            scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
            scp_diag = scp[scp.diagnostic == 1]

            train = db[(db.strat_fold >= 1) & (db.strat_fold <= 8)]
            code_index: dict[str, list[tuple[str, float]]] = {}
            acuity_index: dict[str, dict[str, list[tuple[str, float]]]] = {
                "acute": {}, "chronic": {},
            }
            norm_records: list[str] = []
            for ecg_id, row in train.iterrows():
                record_path = row.filename_lr
                stadium = row.infarction_stadium1 if isinstance(row.infarction_stadium1, str) else ""
                acuity = (
                    "acute" if stadium in _ACUTE_STADIA
                    else "chronic" if stadium in _CHRONIC_STADIA
                    else ""
                )
                for code, conf in row.scp_codes.items():
                    if code not in scp_diag.index:
                        continue
                    if conf < self._min_confidence:
                        continue
                    code_index.setdefault(code, []).append((record_path, float(conf)))
                    # Only index acuity for MI codes — for other codes (AFIB,
                    # blocks, hypertrophy, …) stadium is undefined.
                    if code in _ANY_MI and acuity:
                        acuity_index[acuity].setdefault(code, []).append(
                            (record_path, float(conf))
                        )
                    if code == "NORM":
                        norm_records.append(record_path)

            self._code_index = code_index
            self._code_index_by_acuity = acuity_index
            self._norm_records = norm_records
            top_counts = {k: len(v) for k, v in sorted(code_index.items(), key=lambda x: -len(x[1]))[:15]}
            acute_counts = {k: len(v) for k, v in acuity_index["acute"].items()}
            chronic_counts = {k: len(v) for k, v in acuity_index["chronic"].items()}
            logger.info(
                "PTB-XL indexed (conf>=%.0f): %d SCP codes, top: %s, NORM=%d, "
                "acute-MI per code: %s, chronic-MI per code: %s",
                self._min_confidence, len(code_index), top_counts, len(norm_records),
                acute_counts, chronic_counts,
            )
            return True

    def pick_record(
        self,
        preferred_codes: tuple[str, ...] | str,
        rng: random.Random | None = None,
        *,
        acuity: str | None = None,
    ) -> tuple[str, str] | None:
        """Pick a record matching the diagnosis.

        Accepts either an ordered tuple of preferred SCP codes (new path) or a
        single legacy superclass string (back-compat). For MI diagnoses, the
        optional `acuity` ('acute' / 'chronic') restricts to records tagged with
        the matching infarction_stadium — critical because ~97 % of PTB-XL MI
        records are old/healed and look normal on the ECG. Returns
        (record_path, matched_scp_code) or None when nothing fits.
        """
        if not self._ensure_loaded():
            return None
        rng = rng or random

        # Legacy: a single superclass string (used by `map_diagnosis_to_superclass`).
        if isinstance(preferred_codes, str):
            if preferred_codes == "NORM":
                if not self._norm_records:
                    return None
                return rng.choice(self._norm_records), "NORM"
            candidates = []
            for code, recs in self._code_index.items():
                if self._superclass_of(code) == preferred_codes:
                    candidates.extend((r, code) for r, _conf in recs)
            if not candidates:
                if self._norm_records:
                    return rng.choice(self._norm_records), "NORM"
                return None
            return rng.choice(candidates)

        # New: ordered list of preferred SCP codes.
        # First pass — try the acuity-filtered pool (e.g. only Stadium I/II
        # records for acute STEMI). This is critical correctness for MI cases.
        if acuity in ("acute", "chronic"):
            acuity_pool = self._code_index_by_acuity.get(acuity, {})
            for code in preferred_codes:
                recs = acuity_pool.get(code)
                if recs:
                    rec_path, _conf = rng.choice(recs)
                    return rec_path, code
            # No record matches the requested acuity — log so we know when this
            # happens, but fall through to the general pool rather than returning
            # NORM (a chronic-stadium MI is still better than a NORM ECG).
            logger.info(
                "PTB-XL: no %s records for codes %s — falling back to any stadium",
                acuity, preferred_codes,
            )

        # Second pass — any record matching the preferred codes.
        for code in preferred_codes:
            recs = self._code_index.get(code)
            if recs:
                rec_path, _conf = rng.choice(recs)
                return rec_path, code

        # Nothing matched — fall back to a normal ECG.
        if self._norm_records:
            return rng.choice(self._norm_records), "NORM"
        return None

    @staticmethod
    def _superclass_of(code: str) -> str:
        if code in _ANY_MI:
            return "MI"
        if code in _ISC_ANY or code.startswith("ISC") or code in {"NDT", "NST_", "LNGQT", "DIG", "ANEUR", "EL"}:
            return "STTC"
        if code in {"LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP"}:
            return "HYP"
        if code == "NORM":
            return "NORM"
        return "CD"

    def load_signal(
        self,
        relative_record: str,
        *,
        sample_rate_hz: int = 500,
    ) -> tuple[dict[str, "object"], int] | None:
        """Load PTB-XL waveform; returns ({lead → numpy.ndarray}, sample_rate_hz).

        sample_rate_hz selects between the 500 Hz `records500/*_hr` files (max
        signal fidelity — sharper QRS, finer ST morphology) and the 100 Hz
        `records100/*_lr` files (smaller, smoother). Falls back to whichever
        rate's file actually exists.
        """
        root = self._resolve_data_root()
        if root is None:
            return None
        try:
            import wfdb
        except ImportError:
            logger.warning("wfdb not installed — cannot load PTB-XL signal")
            return None

        # FAISS metadata stores `records100/.../<id>_lr` paths; rewrite to the
        # rate the caller asked for (both 100 Hz and 500 Hz files share the
        # same ecg_id and clinical content).
        if sample_rate_hz == 500:
            candidate = relative_record.replace("records100/", "records500/").replace("_lr", "_hr")
        else:
            candidate = relative_record  # already 100 Hz `_lr`
        if not (root / (candidate + ".hea")).exists():
            candidate = relative_record  # fall back to whichever rate exists

        record = wfdb.rdrecord(str(root / candidate))
        sig = record.p_signal.T  # (12, N)
        ecg: dict[str, "object"] = {
            _normalize_lead(name): sig[i] for i, name in enumerate(record.sig_name)
        }
        return ecg, int(record.fs)

    def rerender_record(
        self,
        source_record: str,
        *,
        sample_rate_hz: int = 500,
    ) -> str | None:
        """Re-render a previously-picked PTB-XL record at a different sample rate.

        Used by the ECG-panel rate toggle: takes a `source_record` value like
        "12388_lr@AMI" (the same string we stash in `ECGVisualPayload.source_record`),
        reconstructs the PTB-XL path from the ecg_id digits, loads the signal at
        the requested rate, and returns the SVG.
        """
        if not self._ensure_loaded():
            return None
        record_id = source_record.split("@", 1)[0]  # e.g. "12388_lr"
        try:
            ecg_id = int(record_id.split("_")[0])
        except ValueError:
            return None
        # PTB-XL path layout: records100/<ecg_id // 1000 * 1000 zero-padded>/<id>_lr
        bucket = f"{ecg_id // 1000 * 1000:05d}"
        relative = f"records100/{bucket}/{ecg_id:05d}_lr"
        loaded = self.load_signal(relative, sample_rate_hz=sample_rate_hz)
        if loaded is None:
            return None
        ecg_dict, sample_rate = loaded
        try:
            return _render_paper_ecg_native(ecg_dict, sample_rate, record_id=record_id)
        except Exception as exc:
            logger.warning("PTB-XL re-render failed for %s: %s", record_id, exc)
            return None

    def render_paper_ecg(
        self,
        diagnosis: str,
        *,
        heart_rate: int | None = None,
        seed: int | None = None,
        age: int | None = None,
        sex: str | None = None,
    ) -> tuple[str, str] | None:
        """Render a paper-style ECG SVG for the given diagnosis.

        Retrieval order:
          1. Regex maps diagnosis to specific SCP codes → exact anatomic match
             (best for common cardiac diagnoses where territory matters).
          2. FAISS similarity search over the embedded index → covers diagnoses
             my regex doesn't know (atrial fibrillation, sinus arrhythmia,
             pacemaker rhythm, etc.).
          3. NORM fallback when the score is too low to trust.

        Returns (png_bytes, source_id) where source_id encodes the record and
        the picker that matched it (e.g. "00123_lr@AMI" or "00123_lr@faiss").
        heart_rate is accepted for API symmetry but unused — stretching a real
        recording to fake an HR would corrupt morphology.
        """
        if not self._ensure_loaded():
            return None
        rng = random.Random(seed)

        # 1. Anatomically precise regex path (territory, AV-block degree, etc.)
        preferred = map_diagnosis_to_codes(diagnosis)
        # For MI codes, infer acuity from the diagnosis text so we don't pick a
        # chronic/healed MI when the case is an acute STEMI presentation.
        acuity = _detect_mi_acuity(diagnosis, preferred) if preferred else None
        picked: tuple[str, str] | None = None
        if preferred:
            picked = self.pick_record(preferred, rng=rng, acuity=acuity)

        # 2. FAISS retrieval for diagnoses the regex doesn't cover
        if picked is None or picked[1] == "NORM":
            faiss_pick = _get_faiss_index().search(
                query_text=_build_query_text(age, sex, diagnosis),
                rng=rng,
            )
            if faiss_pick is not None:
                rel_path, top_code, score = faiss_pick
                # Trust FAISS only when it found something genuinely related
                # OR when the regex returned nothing at all.
                if score >= _FAISS_MIN_SCORE or not preferred:
                    if score >= _FAISS_MIN_SCORE:
                        picked = (rel_path, top_code or "faiss")
                    elif picked is None:
                        # Score too low and regex empty — explicit NORM.
                        picked = self.pick_record("NORM", rng=rng)

        # 3. Final fallback (shouldn't normally trigger)
        if picked is None:
            picked = self.pick_record("NORM", rng=rng)
        if picked is None:
            return None

        relative, matched_code = picked
        loaded = self.load_signal(relative)
        if loaded is None:
            return None
        ecg_dict, sample_rate = loaded
        record_id = Path(relative).name
        try:
            svg = _render_paper_ecg_native(ecg_dict, sample_rate, record_id=record_id)
        except Exception as exc:
            logger.warning("PTB-XL SVG render failed for %s: %s", relative, exc)
            return None
        logger.info(
            "PTB-XL render: diagnosis=%r → code=%s → record=%s",
            diagnosis, matched_code, record_id,
        )
        return svg, f"{record_id}@{matched_code}"


def _normalize_lead(name: str) -> str:
    """Standardize lead casing for ECG-Image-Kit ('AVR' → 'aVR', etc.)."""
    u = name.upper()
    if u in ("AVR", "AVL", "AVF"):
        return "aV" + u[2]
    return u


_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _render_paper_ecg_native(
    ecg_dict: dict[str, "object"],
    sample_rate: int,
    *,
    record_id: str,
) -> str:
    """Render 12-lead ECG as a stacked-row paper-style SVG using matplotlib.

    One lead per row, full duration per strip, ±0.5 mV calibration grid.
    Returns the SVG as a string so it embeds inline in the page — vector
    graphics stay sharp at any zoom level, unlike a rasterized PNG.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_samples = max(len(np.asarray(ecg_dict[lead])) for lead in _LEAD_ORDER if lead in ecg_dict)
    duration = n_samples / sample_rate
    t = np.arange(n_samples) / sample_rate

    # Tight ±0.8 mV ylim that's sized for typical limb-lead amplitudes.
    # The ±0.5 mV major gridlines give the standard 10 mm/mV calibration.
    # Big-amplitude chest-lead R-waves may clip at the panel edge — that's
    # acceptable on a real ECG paper print and keeps the rendering readable.
    AMP = 0.8
    ticks_major = np.arange(-0.5, 0.51, 0.5)
    tick_labels = ["-0.5 mV", "0 mV", "+0.5 mV"]

    fig, axes = plt.subplots(12, 1, figsize=(11, 10), sharex=True)
    for ax, lead in zip(axes, _LEAD_ORDER):
        y = np.asarray(ecg_dict.get(lead, np.zeros(n_samples)))
        # Major grid only (every 0.2 s × 0.5 mV). Standard ECG paper has
        # minor gridlines too, but rendering 3,000+ minor lines per record
        # as individual SVG paths bloats the file to multi-MB without much
        # clinical value — doctors primarily read off the major boxes.
        ax.set_xticks(np.arange(0, duration + 1e-3, 0.2), minor=False)
        ax.set_yticks(ticks_major, minor=False)
        ax.grid(which="major", color="#f08a8a", linewidth=0.5)
        ax.set_facecolor("#fff5f5")
        ax.plot(t, y, color="#000000", linewidth=0.9)
        ax.set_ylim(-AMP, AMP)
        ax.set_yticklabels(tick_labels, fontsize=6)
        display = lead.upper() if lead.lower() not in ("avr", "avl", "avf") else lead
        ax.text(
            1.01, 0.5, display, transform=ax.transAxes, fontsize=9,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999"),
        )
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time in seconds", fontsize=11)
    plt.tight_layout()

    buf = io.StringIO()
    # dpi controls the resolution of rasterized elements (the grid only — the
    # trace is full vector). 200 dpi is plenty for grid lines that are 0.3-0.5px.
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=200)
    plt.close(fig)
    svg = buf.getvalue()
    # Strip XML declaration so the SVG can be embedded inline as innerHTML.
    if svg.startswith("<?xml"):
        svg = svg[svg.index("<svg") :]
    return svg


# ── FAISS semantic retrieval (mirrors backend.cxr_service._FaissIndex) ───────

def _build_query_text(age: int | None, sex: str | None, diagnosis: str) -> str:
    """Compose the query string in the same shape as the indexed embed texts."""
    parts: list[str] = []
    if age is not None:
        parts.append(f"{int(age)}yo")
    if sex:
        s = sex.strip().lower()
        if s in ("male", "m", "0"):
            parts.append("male")
        elif s in ("female", "f", "1"):
            parts.append("female")
    prefix = " ".join(parts)
    if prefix:
        return f"{prefix}. {diagnosis}".strip()
    return diagnosis.strip()


class _PTBXLFaiss:
    """Lazy-loaded FAISS index for diagnosis → similar PTB-XL record retrieval."""

    def __init__(self, index_dir: Path = _PTBXL_INDEX_DIR) -> None:
        self._index_dir = index_dir
        self._loaded = False
        self._index = None
        self._meta: dict | None = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        faiss_path = self._index_dir / "index.faiss"
        meta_path = self._index_dir / "metadata.parquet"
        return faiss_path.exists() and meta_path.exists()

    def _load(self) -> bool:
        if self._loaded:
            return self._index is not None
        with self._lock:
            if self._loaded:
                return self._index is not None
            self._loaded = True
            if not self.available:
                logger.info(
                    "PTB-XL FAISS index not found at %s — run scripts/build_ptbxl_index.py",
                    self._index_dir,
                )
                return False
            try:
                import faiss
                import pyarrow.parquet as pq
                from sentence_transformers import SentenceTransformer

                self._index = faiss.read_index(str(self._index_dir / "index.faiss"))
                self._meta = pq.read_table(self._index_dir / "metadata.parquet").to_pydict()
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info(
                    "PTB-XL FAISS index loaded: %d vectors",
                    self._index.ntotal,
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load PTB-XL FAISS index: %s", exc)
                self._index = None
                return False

    def search(
        self,
        query_text: str,
        rng: random.Random | None = None,
    ) -> tuple[str, str, float] | None:
        """Return (relative_record, top_scp_code, score) for a query, or None."""
        if not self._load() or self._index is None or self._model is None or self._meta is None:
            return None
        import faiss
        import numpy as np

        vec = self._model.encode([query_text], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(vec)

        k = min(_FAISS_TOP_N_POOL, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        rng = rng or random
        # Pick from the top-N pool — adds variety across repeated queries.
        pool = [
            (int(i), float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]
        if not pool:
            return None
        chosen_idx, chosen_score = rng.choice(pool)
        return (
            str(self._meta["filename_lr"][chosen_idx]),
            str(self._meta["top_code"][chosen_idx]),
            chosen_score,
        )


# Singleton — initialized on first use, indexes once.
_service: PTBXLService | None = None
_faiss: _PTBXLFaiss | None = None


def get_ptbxl_service() -> PTBXLService:
    global _service
    if _service is None:
        _service = PTBXLService()
    return _service


def _get_faiss_index() -> _PTBXLFaiss:
    global _faiss
    if _faiss is None:
        _faiss = _PTBXLFaiss()
    return _faiss


def encode_png(data: bytes) -> str:
    """Base64-encode PNG bytes for transport in the visual payload."""
    return base64.b64encode(data).decode("ascii")


# Re-export so callers can encode without importing base64 directly.
__all__ = [
    "PTBXLService",
    "get_ptbxl_service",
    "map_diagnosis_to_codes",
    "map_diagnosis_to_superclass",
    "encode_png",
]
