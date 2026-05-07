"""
Download a representative sample of chest X-ray images from the
RoentGen-v2 synthetic dataset (stanfordmimi/RoentGen-v2-synthetic-dataset).

Run once to populate data/cxr/<finding>/*.jpg before starting the server.

Usage:
    python scripts/download_cxr_samples.py

Requires:
    - HUGGINGFACE_TOKEN in .env (must have accepted the dataset access agreement)
    - pip install datasets pillow python-dotenv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ID = "stanfordmimi/RoentGen-v2-synthetic-dataset"
IMAGES_PER_FINDING = 20
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cxr"

# All finding columns in the dataset that we care about
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

# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming dataset: {DATASET_ID}")
    print(f"Target: {IMAGES_PER_FINDING} images per finding → {OUTPUT_DIR}\n")

    ds = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        token=token,
    )

    # Track how many we've saved per finding
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
            if counts[finding] >= IMAGES_PER_FINDING:
                done.add(finding)
                continue

            finding_dir = OUTPUT_DIR / finding.replace(" ", "_")
            finding_dir.mkdir(parents=True, exist_ok=True)
            img_path = finding_dir / f"{row['image_id']}.jpg"

            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=85)
            counts[finding] += 1

            if counts[finding] == IMAGES_PER_FINDING:
                print(f"  ✓ {finding:35s} {counts[finding]} images")
                done.add(finding)

    print(f"\nDone. Images saved to {OUTPUT_DIR}")
    for finding, count in counts.items():
        if count < IMAGES_PER_FINDING:
            print(f"  ! {finding}: only {count}/{IMAGES_PER_FINDING} found in stream")


if __name__ == "__main__":
    main()
