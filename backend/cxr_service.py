"""
CXR service: picks a representative chest X-ray image for a given diagnosis
from the locally downloaded RoentGen-v2 sample set and returns it as a
base64-encoded JPEG payload.

Images must be downloaded first:
    python scripts/download_cxr_samples.py
"""

from __future__ import annotations

import base64
import logging
import random
from pathlib import Path

from schemas.investigation import CXRVisualPayload

logger = logging.getLogger(__name__)

CXR_DATA_DIR = Path(__file__).parent.parent / "data" / "cxr"

# ── Diagnosis → finding mapping ───────────────────────────────────────────────
# Keywords matched (case-insensitive, substring) against the diagnosis string.
# First match wins. Falls back to "No_Finding".

_FINDING_RULES: list[tuple[list[str], str]] = [
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

_DEFAULT_FINDING = "No_Finding"


def _match_finding(diagnosis: str) -> str:
    lower = diagnosis.strip().lower()
    for keywords, finding in _FINDING_RULES:
        if any(kw in lower for kw in keywords):
            return finding
    return _DEFAULT_FINDING


def _finding_dir(finding: str) -> Path:
    return CXR_DATA_DIR / finding


def _available_images(finding: str) -> list[Path]:
    d = _finding_dir(finding)
    if not d.exists():
        return []
    return list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))


class CXRService:
    """Returns a CXRVisualPayload for a given diagnosis."""

    def build_visual_payload(self, diagnosis: str) -> CXRVisualPayload | None:
        finding = _match_finding(diagnosis)
        images = _available_images(finding)

        # Fallback: if finding has no images, try No_Finding
        if not images and finding != _DEFAULT_FINDING:
            logger.warning(
                "No CXR images for finding %r (diagnosis: %r), falling back to No_Finding",
                finding,
                diagnosis,
            )
            finding = _DEFAULT_FINDING
            images = _available_images(finding)

        if not images:
            logger.warning(
                "CXR image directory empty or missing — run scripts/download_cxr_samples.py"
            )
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
            logger.warning("Failed to read CXR image %s: %s", img_path, exc)
            return None
