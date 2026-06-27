"""
Build a FAISS embedding index over the locally cached RoentGen-v2 dataset.

For every PA-view row the script:
  1. Constructs an embed text:  "<age>yo <gender> <race>. <impression>"
  2. Encodes it with all-MiniLM-L6-v2 (L2-normalised for cosine similarity)
  3. Adds the vector to a FAISS IndexFlatIP
  4. Records shard path + row index + metadata in a companion parquet file

Output:
  data/cxr_index/index.faiss      — FAISS binary
  data/cxr_index/metadata.parquet — per-row metadata (no image bytes)

Usage:
    python scripts/build_cxr_index.py

Requires packages already installed by pip:
    sentence-transformers, faiss-cpu, pyarrow
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HF_CACHE_SNAPSHOT = Path(
    "/Users/hammad/.cache/huggingface/hub"
    "/datasets--stanfordmimi--RoentGen-v2-synthetic-dataset"
    "/snapshots/851fe4065ec30232933b54e4c17b0cd3dc66bddd"
)
DATA_DIR = HF_CACHE_SNAPSHOT / "data"
INDEX_DIR = Path(__file__).parent.parent / "data" / "cxr_index"

FINDING_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
GENDER_MAP = {"M": "male", "F": "female"}


def embed_text(row: dict) -> str:
    age = row.get("age") or "unknown"
    gender = GENDER_MAP.get(str(row.get("gender") or ""), str(row.get("gender") or "unknown"))
    race = str(row.get("race") or "unknown").lower()
    impression = str(row.get("impression") or "").strip()
    return f"{age}yo {gender} {race}. {impression}"


def main() -> None:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers faiss-cpu pyarrow")
        sys.exit(1)

    shards = sorted(DATA_DIR.glob("train-*.parquet"))
    if not shards:
        print(f"ERROR: No parquet files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(shards)} shards in {DATA_DIR}\n")

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}\n")

    index = faiss.IndexFlatIP(dim)

    # metadata lists (one entry per indexed row)
    meta_shard: list[str] = []
    meta_row_idx: list[int] = []
    meta_image_id: list[str] = []
    meta_age: list[int | None] = []
    meta_gender: list[str] = []
    meta_race: list[str] = []
    meta_findings: dict[str, list[int]] = {col: [] for col in FINDING_COLS}

    total_rows = 0
    total_pa = 0

    for shard_num, shard_path in enumerate(shards, 1):
        print(f"[{shard_num:02d}/{len(shards)}] {shard_path.name} ...", end=" ", flush=True)

        read_cols = ["image_id", "gender", "age", "race", "view", "impression"] + FINDING_COLS
        table = pq.read_table(shard_path, columns=read_cols)
        rows = table.to_pydict()
        n = len(rows["image_id"])
        total_rows += n

        # filter PA view
        pa_indices = [i for i in range(n) if str(rows["view"][i]).upper() == "PA"]
        total_pa += len(pa_indices)
        print(f"{n} rows → {len(pa_indices)} PA-view", flush=True)

        if not pa_indices:
            continue

        texts = [embed_text({col: rows[col][i] for col in read_cols}) for i in pa_indices]

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            batch_indices = pa_indices[batch_start : batch_start + BATCH_SIZE]

            vecs = model.encode(batch_texts, batch_size=BATCH_SIZE, show_progress_bar=False)
            vecs = vecs.astype(np.float32)
            faiss.normalize_L2(vecs)
            index.add(vecs)

            for i in batch_indices:
                meta_shard.append(str(shard_path))
                meta_row_idx.append(i)
                meta_image_id.append(str(rows["image_id"][i]))
                meta_age.append(rows["age"][i])
                meta_gender.append(str(rows["gender"][i]))
                meta_race.append(str(rows["race"][i]))
                for col in FINDING_COLS:
                    meta_findings[col].append(int(rows[col][i]) if rows[col][i] is not None else 0)

    print(f"\nTotal rows scanned: {total_rows:,} | PA-view indexed: {total_pa:,}")
    print(f"FAISS index size: {index.ntotal:,} vectors")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss_path = INDEX_DIR / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved: {faiss_path}")

    meta_schema = pa.schema(
        [
            ("shard", pa.string()),
            ("row_idx", pa.int32()),
            ("image_id", pa.string()),
            ("age", pa.int32()),
            ("gender", pa.string()),
            ("race", pa.string()),
        ]
        + [(col, pa.int8()) for col in FINDING_COLS]
    )
    meta_table = pa.table(
        {
            "shard": meta_shard,
            "row_idx": meta_row_idx,
            "image_id": meta_image_id,
            "age": pa.array(meta_age, type=pa.int32()),
            "gender": meta_gender,
            "race": meta_race,
            **{col: pa.array(meta_findings[col], type=pa.int8()) for col in FINDING_COLS},
        },
        schema=meta_schema,
    )
    meta_path = INDEX_DIR / "metadata.parquet"
    pq.write_table(meta_table, meta_path, compression="snappy")
    print(f"Saved: {meta_path}")
    print(f"\nDone. Index ready at {INDEX_DIR}/")


if __name__ == "__main__":
    main()
