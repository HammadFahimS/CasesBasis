"""
CXR service: retrieves the most clinically relevant chest X-ray from the
locally indexed RoentGen-v2 dataset using FAISS embedding search.

Usage:
    1. python scripts/build_cxr_index.py        (one-time, ~30 min)
    2. Then the service loads automatically on first use.

Falls back to legacy random-from-folder behavior if the FAISS index has not
been built yet.
"""

from __future__ import annotations

import base64
import logging
import random
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

from schemas.investigation import CXRVisualPayload

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CXR_DATA_DIR = Path(__file__).parent.parent / "data" / "cxr"
INDEX_DIR = Path(__file__).parent.parent / "data" / "cxr_index"
FAISS_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.parquet"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── FAISS search parameters ───────────────────────────────────────────────────
SEARCH_K = 200       # candidates retrieved from FAISS
TOP_N_POOL = 20      # pick randomly from the top-N after filtering

# ── Diagnosis → finding column mapping ───────────────────────────────────────
_FINDING_RULES: list[tuple[list[str], str]] = [
    (["pneumothorax"],                              "Pneumothorax"),
    (["pleural effusion", "effusion"],              "Pleural Effusion"),
    (["pneumonia", "pneumonitis", "bronchopneumonia",
      "lobar pneumonia", "community-acquired"],     "Pneumonia"),
    (["consolidation"],                             "Consolidation"),
    (["pulmonary oedema", "pulmonary edema",
      "heart failure", "chf", "congestive",
      "flash pulmonary"],                           "Edema"),
    (["atelectasis", "collapse"],                   "Atelectasis"),
    (["cardiomegaly"],                              "Cardiomegaly"),
    (["ards", "acute respiratory distress",
      "lung opacity", "diffuse opacity"],           "Lung Opacity"),
    (["pericarditis", "pericardial effusion",
      "aortic dissection", "mediastin"],            "Enlarged Cardiomediastinum"),
    (["rib fracture", "fracture"],                  "Fracture"),
    (["lung lesion", "lung mass", "nodule",
      "malignancy", "cancer", "tumour", "tumor"],   "Lung Lesion"),
    (["mesothelioma", "pleural thickening"],        "Pleural Other"),
]

GENDER_MAP = {"male": "M", "female": "F", "m": "M", "f": "F"}


def _match_finding(diagnosis: str) -> str | None:
    lower = diagnosis.strip().lower()
    for keywords, finding in _FINDING_RULES:
        if any(kw in lower for kw in keywords):
            return finding
    return None


def _build_query_text(age: int, gender: str, race: str | None, diagnosis: str) -> str:
    gender_str = GENDER_MAP.get(gender.lower(), gender)
    race_str = (race or "unknown").lower()
    return f"{age}yo {gender_str} {race_str}. {diagnosis}"


# ── Lazy singleton ────────────────────────────────────────────────────────────

class _FaissIndex:
    """Lazily loaded FAISS index + metadata table."""

    def __init__(self) -> None:
        self._loaded = False
        self._index = None
        self._meta = None
        self._model: "SentenceTransformer | None" = None
        self._meta_dict: dict | None = None

    def _load(self) -> bool:
        if self._loaded:
            return self._index is not None

        self._loaded = True
        if not FAISS_PATH.exists() or not META_PATH.exists():
            logger.info("CXR FAISS index not found — run scripts/build_cxr_index.py")
            return False

        try:
            import faiss
            import pyarrow.parquet as pq
            from sentence_transformers import SentenceTransformer

            logger.info("Loading CXR FAISS index from %s ...", FAISS_PATH)
            self._index = faiss.read_index(str(FAISS_PATH))
            logger.info("  Loaded %d vectors", self._index.ntotal)

            logger.info("Loading CXR metadata from %s ...", META_PATH)
            table = pq.read_table(META_PATH)
            self._meta_dict = table.to_pydict()
            logger.info("  Loaded %d metadata rows", len(self._meta_dict["image_id"]))

            logger.info("Loading embedding model: %s ...", EMBED_MODEL_NAME)
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("  Model ready")
            return True
        except Exception as exc:
            logger.warning("Failed to load CXR FAISS index: %s", exc)
            self._index = None
            return False

    def search(
        self,
        query_text: str,
        finding_filter: str | None,
    ) -> dict | None:
        """Return a single metadata row dict for the best matching CXR."""
        if not self._load() or self._index is None or self._model is None:
            return None

        import faiss
        import numpy as np

        vec = self._model.encode([query_text], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(vec)

        k = min(SEARCH_K, self._index.ntotal)
        _scores, indices = self._index.search(vec, k)
        candidate_idxs: list[int] = [int(i) for i in indices[0] if i >= 0]

        meta = self._meta_dict
        assert meta is not None

        def row(idx: int) -> dict:
            return {col: meta[col][idx] for col in meta}

        # Filter by finding flag if we have one
        if finding_filter:
            filtered = [i for i in candidate_idxs if meta.get(finding_filter, [None])[i] == 1]
        else:
            filtered = candidate_idxs

        pool = filtered[:TOP_N_POOL] if filtered else candidate_idxs[:TOP_N_POOL]
        if not pool:
            return None

        chosen_idx = random.choice(pool)
        return row(chosen_idx)


_faiss_index = _FaissIndex()


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image_bytes_from_parquet(shard: str, row_idx: int) -> bytes | None:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(shard)
        rows_per_rg = pf.metadata.row_group(0).num_rows
        rg_idx = row_idx // rows_per_rg
        offset = row_idx % rows_per_rg
        table = pf.read_row_group(rg_idx, columns=["image"])
        img_dict = table.to_pydict()["image"][offset]
        raw: bytes = img_dict["bytes"]
        return raw
    except Exception as exc:
        logger.warning("Failed to load image from %s row %d: %s", shard, row_idx, exc)
        return None


def _bytes_to_jpeg_base64(raw: bytes) -> str:
    from PIL import Image

    img = Image.open(BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Legacy fallback (random from folder) ──────────────────────────────────────

_LEGACY_FINDING_RULES: list[tuple[list[str], str]] = [
    (["pneumothorax"],                              "Pneumothorax"),
    (["pleural effusion", "effusion"],              "Pleural_Effusion"),
    (["pneumonia", "pneumonitis", "bronchopneumonia",
      "lobar pneumonia", "community-acquired"],     "Pneumonia"),
    (["consolidation"],                             "Consolidation"),
    (["pulmonary oedema", "pulmonary edema",
      "heart failure", "chf", "congestive",
      "flash pulmonary"],                           "Edema"),
    (["atelectasis", "collapse"],                   "Atelectasis"),
    (["cardiomegaly"],                              "Cardiomegaly"),
    (["ards", "acute respiratory distress",
      "lung opacity", "diffuse opacity"],           "Lung_Opacity"),
    (["pericarditis", "pericardial effusion",
      "aortic dissection", "mediastin"],            "Enlarged_Cardiomediastinum"),
    (["rib fracture", "fracture"],                  "Fracture"),
    (["lung lesion", "lung mass", "nodule",
      "malignancy", "cancer", "tumour", "tumor"],   "Lung_Lesion"),
    (["mesothelioma", "pleural thickening"],        "Pleural_Other"),
]
_LEGACY_DEFAULT = "No_Finding"


def _legacy_match_finding(diagnosis: str) -> str:
    lower = diagnosis.strip().lower()
    for keywords, finding in _LEGACY_FINDING_RULES:
        if any(kw in lower for kw in keywords):
            return finding
    return _LEGACY_DEFAULT


def _legacy_payload(diagnosis: str) -> CXRVisualPayload | None:
    finding = _legacy_match_finding(diagnosis)
    d = CXR_DATA_DIR / finding
    images = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) if d.exists() else []
    if not images and finding != _LEGACY_DEFAULT:
        finding = _LEGACY_DEFAULT
        d = CXR_DATA_DIR / finding
        images = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) if d.exists() else []
    if not images:
        return None
    img_path = random.choice(images)
    try:
        data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return CXRVisualPayload(
            type="image",
            format="jpeg",
            data=data,
            finding=finding.replace("_", " "),
        )
    except Exception as exc:
        logger.warning("Legacy CXR load failed: %s", exc)
        return None


# ── Public service ─────────────────────────────────────────────────────────────

class CXRService:
    """Returns a CXRVisualPayload for a given diagnosis using FAISS embedding search."""

    def build_visual_payload(
        self,
        diagnosis: str,
        age: int = 40,
        gender: str = "male",
        race: str | None = None,
    ) -> CXRVisualPayload | None:
        finding_col = _match_finding(diagnosis)
        query_text = _build_query_text(age, gender, race, diagnosis)

        meta_row = _faiss_index.search(query_text, finding_col)
        if meta_row is None:
            logger.info("FAISS index unavailable — falling back to legacy CXR selection")
            return _legacy_payload(diagnosis)

        raw = _load_image_bytes_from_parquet(meta_row["shard"], meta_row["row_idx"])
        if raw is None:
            logger.warning("Image load failed — falling back to legacy")
            return _legacy_payload(diagnosis)

        try:
            data = _bytes_to_jpeg_base64(raw)
        except Exception as exc:
            logger.warning("JPEG conversion failed: %s — falling back", exc)
            return _legacy_payload(diagnosis)

        display_finding = finding_col if finding_col else "No Finding"
        return CXRVisualPayload(
            type="image",
            format="jpeg",
            data=data,
            finding=display_finding,
        )
