"""
Download the full RoentGen-v2 dataset locally, then extract
representative chest X-ray images per finding category.

Steps:
  1. Downloads all 33 parquet files (~15.7 GB) via huggingface_hub
  2. Iterates locally (fast, no streaming) and saves IMAGES_PER_FINDING
     images per finding category to data/cxr/<Finding>/

Usage:
    python scripts/download_cxr_samples.py

Requires:
    HUGGINGFACE_TOKEN in .env with access to stanfordmimi/RoentGen-v2-synthetic-dataset
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATASET_ID = "stanfordmimi/RoentGen-v2-synthetic-dataset"
IMAGES_PER_FINDING = 20
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cxr"

FINDING_COLUMNS = [
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


def folder_name(finding: str) -> str:
    return finding.replace(" ", "_")


def main() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_TOKEN not set in .env")
        sys.exit(1)

    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: Run: pip install datasets huggingface_hub")
        sys.exit(1)

    # ── Step 1: Download all parquet files locally ─────────────────────────────
    print(f"Step 1: Downloading full dataset ({DATASET_ID}) ...")
    cache_dir = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=token,
        ignore_patterns=["*.md", "*.txt", "*.json"],
    )
    print(f"  Cached at: {cache_dir}\n")

    # ── Step 2: Load from local cache and extract images ──────────────────────
    print("Step 2: Loading from local cache ...")
    ds = load_dataset(
        DATASET_ID,
        split="train",
        cache_dir=str(Path(cache_dir).parent.parent),
        token=token,
    )
    print(f"  Total rows: {len(ds):,}\n")

    print(f"Step 3: Extracting {IMAGES_PER_FINDING} images per finding ...\n")
    counts: dict[str, int] = {f: 0 for f in FINDING_COLUMNS}
    done: set[str] = set()

    for row in ds:
        if len(done) == len(FINDING_COLUMNS):
            break

        for finding in FINDING_COLUMNS:
            if finding in done:
                continue
            if row.get(finding) != 1:
                continue

            out_dir = OUTPUT_DIR / folder_name(finding)
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"{row['image_id']}.jpg"

            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=85)
            counts[finding] += 1

            if counts[finding] >= IMAGES_PER_FINDING:
                print(f"  ✓ {finding:35s} {counts[finding]} images")
                done.add(finding)

    print(f"\nDone. Images saved to {OUTPUT_DIR}")
    total = sum(counts.values())
    print(f"Total: {total} images across {len(counts)} categories")
    for finding, count in counts.items():
        if count < IMAGES_PER_FINDING:
            print(f"  ! {finding}: only {count}/{IMAGES_PER_FINDING} (finding is rare in dataset)")


if __name__ == "__main__":
    main()
