"""
Build a FAISS embedding index over PTB-XL train-fold ECG records.

For every record in folds 1–8 the script:
  1. Builds an English clinical text from age/sex/heart_axis plus the SCP
     code descriptions (confidence-ordered, ≥ 50% to skip noise labels).
  2. Encodes it with all-MiniLM-L6-v2 (L2-normalised for cosine similarity).
  3. Adds the vector to a FAISS IndexFlatIP.
  4. Records record path + metadata in a companion parquet file.

Output:
  data/ptbxl_index/index.faiss      — FAISS binary
  data/ptbxl_index/metadata.parquet — per-row metadata

Usage:
    python scripts/build_ptbxl_index.py
"""

from __future__ import annotations

import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PTBXL_BASE = PROJECT_ROOT / "data" / "ptb-xl"
INDEX_DIR = PROJECT_ROOT / "data" / "ptbxl_index"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Confidence floor for SCP codes to appear in the embed text. ~24% of MI
# labels are below 50 % — those are essentially noise and would poison the
# semantic space.
MIN_DESC_CONFIDENCE = 50.0


def _resolve_data_root() -> Path:
    """Find ptbxl_database.csv whether the user unpacked the ZIP or not."""
    candidates = [PTBXL_BASE]
    if PTBXL_BASE.exists():
        candidates += [p for p in PTBXL_BASE.iterdir() if p.is_dir()]
    for c in candidates:
        if (c / "ptbxl_database.csv").exists():
            return c
    raise SystemExit(
        f"ptbxl_database.csv not found under {PTBXL_BASE}. "
        "Did the PTB-XL download finish?"
    )


def _build_embed_text(row: pd.Series, scp_meta: pd.DataFrame) -> str:
    """Compose an English clinical description for a single PTB-XL row.

    Format:  "{age}yo {sex}, axis {axis}. {descriptions}."
    Example: "62yo male, axis LAD. anterior myocardial infarction; sinus rhythm."
    """
    age = "unknown" if pd.isna(row.age) else f"{int(row.age)}yo"
    sex = "female" if row.sex == 1 else "male"
    axis = row.heart_axis if isinstance(row.heart_axis, str) and row.heart_axis else ""

    descs: list[str] = []
    # row.scp_codes is a dict {code: confidence}; order by confidence desc.
    for code, conf in sorted(row.scp_codes.items(), key=lambda kv: -kv[1]):
        if conf < MIN_DESC_CONFIDENCE:
            continue
        if code not in scp_meta.index:
            continue
        desc = str(scp_meta.loc[code, "description"]).strip()
        if desc and desc not in descs:
            descs.append(desc)

    findings = "; ".join(descs) if descs else "normal ECG"
    prefix = f"{age} {sex}"
    if axis:
        prefix += f", axis {axis}"
    return f"{prefix}. {findings}."


def main() -> None:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers faiss-cpu pyarrow")
        sys.exit(1)

    root = _resolve_data_root()
    print(f"PTB-XL data root: {root}\n")

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)

    train = db[(db.strat_fold >= 1) & (db.strat_fold <= 8)].copy()
    print(f"Train rows (folds 1-8): {len(train):,}")

    texts = [_build_embed_text(row, scp) for _, row in train.iterrows()]
    print(f"Sample embed text: {texts[0]!r}")
    print(f"                   {texts[100]!r}\n")

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}\n")

    index = faiss.IndexFlatIP(dim)

    print("Embedding ...")
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        vecs = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  {batch_start + len(batch):,}/{len(texts):,}")

    print(f"\nFAISS index size: {index.ntotal:,} vectors")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss_path = INDEX_DIR / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved: {faiss_path}")

    # Pre-compute the matched-superclass and the top SCP code per record so
    # the runtime filter doesn't have to re-walk scp_statements.
    def top_code(scp_codes: dict) -> str:
        candidates = [
            (c, conf) for c, conf in scp_codes.items()
            if conf >= MIN_DESC_CONFIDENCE and c in scp.index
        ]
        if not candidates:
            return ""
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def superclass(scp_codes: dict) -> str:
        cs = [
            scp.loc[c, "diagnostic_class"]
            for c, conf in scp_codes.items()
            if conf >= MIN_DESC_CONFIDENCE and c in scp.index
            and scp.loc[c, "diagnostic"] == 1
        ]
        cs = [s for s in cs if isinstance(s, str)]
        return cs[0] if cs else ""

    meta_table = pa.table({
        "ecg_id": pa.array(train.index.tolist(), type=pa.int32()),
        "filename_lr": train.filename_lr.tolist(),
        "age": pa.array(
            [None if pd.isna(a) else int(a) for a in train.age.tolist()],
            type=pa.int32(),
        ),
        "sex": pa.array([int(s) for s in train.sex.tolist()], type=pa.int8()),
        "heart_axis": [str(a) if isinstance(a, str) else "" for a in train.heart_axis.tolist()],
        "top_code": [top_code(c) for c in train.scp_codes],
        "superclass": [superclass(c) for c in train.scp_codes],
        "strat_fold": pa.array([int(f) for f in train.strat_fold.tolist()], type=pa.int8()),
        "embed_text": texts,
    })
    meta_path = INDEX_DIR / "metadata.parquet"
    pq.write_table(meta_table, meta_path, compression="snappy")
    print(f"Saved: {meta_path}")
    print(f"\nDone. Index ready at {INDEX_DIR}/")


if __name__ == "__main__":
    main()
