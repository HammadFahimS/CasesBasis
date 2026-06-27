from __future__ import annotations

from pathlib import Path

from backend.ptbxl_service import (
    PTBXLService,
    map_diagnosis_to_codes,
    map_diagnosis_to_superclass,
)


# ── Coarse superclass mapping (back-compat) ──────────────────────────────────

def test_diagnosis_to_superclass_known_cardiac_terms() -> None:
    assert map_diagnosis_to_superclass("myocardial infarction") == "MI"
    assert map_diagnosis_to_superclass("STEMI") == "MI"
    assert map_diagnosis_to_superclass("unstable angina") == "STTC"
    assert map_diagnosis_to_superclass("Left Bundle Branch Block") == "CD"
    assert map_diagnosis_to_superclass("LVH") == "HYP"


def test_diagnosis_to_superclass_falls_back_to_norm() -> None:
    assert map_diagnosis_to_superclass("appendicitis") == "NORM"
    assert map_diagnosis_to_superclass("") == "NORM"
    assert map_diagnosis_to_superclass("unrelated condition") == "NORM"


# ── Anatomic SCP-code mapping (fine-grained) ─────────────────────────────────

def test_anterior_mi_prefers_anterior_codes() -> None:
    codes = map_diagnosis_to_codes("anterior STEMI")
    assert codes[0] in {"AMI", "ASMI", "ALMI"}
    assert "IMI" not in codes[:3], "anterior STEMI must not prefer inferior codes"


def test_inferior_mi_prefers_inferior_codes() -> None:
    codes = map_diagnosis_to_codes("inferior wall MI")
    assert codes[0] in {"IMI", "ILMI", "IPMI", "IPLMI"}
    assert "AMI" not in codes[:3], "inferior MI must not prefer anterior codes"


def test_anteroseptal_specifically_picks_asmi_first() -> None:
    assert map_diagnosis_to_codes("anteroseptal infarction")[0] == "ASMI"


def test_av_blocks_are_degree_specific() -> None:
    assert map_diagnosis_to_codes("first-degree AV block")[0] == "1AVB"
    assert map_diagnosis_to_codes("second degree heart block")[0] == "2AVB"
    assert map_diagnosis_to_codes("complete heart block")[0] == "3AVB"


def test_bundle_blocks_distinguish_left_vs_right() -> None:
    assert map_diagnosis_to_codes("LBBB")[0] == "CLBBB"
    assert map_diagnosis_to_codes("complete right bundle branch block")[0] == "CRBBB"


def test_non_cardiac_diagnoses_return_empty_then_norm() -> None:
    assert map_diagnosis_to_codes("appendicitis") == ()
    assert map_diagnosis_to_codes("pneumonia") == ()


# ── Service ──────────────────────────────────────────────────────────────────

def test_service_disabled_when_dataset_missing(tmp_path: Path) -> None:
    svc = PTBXLService(base_dir=tmp_path)
    assert svc.enabled is False
    assert svc.pick_record(("AMI",)) is None
    assert svc.pick_record("MI") is None
    assert svc.render_paper_ecg("myocardial infarction") is None


# ── FAISS query text builder ─────────────────────────────────────────────────

def test_build_query_text_includes_age_and_sex() -> None:
    from backend.ptbxl_service import _build_query_text
    assert _build_query_text(58, "male", "anterior STEMI") == "58yo male. anterior STEMI"
    assert _build_query_text(72, "female", "complete heart block") == "72yo female. complete heart block"


def test_build_query_text_handles_missing_demographics() -> None:
    from backend.ptbxl_service import _build_query_text
    assert _build_query_text(None, None, "atrial fibrillation") == "atrial fibrillation"
    assert _build_query_text(40, None, "LVH") == "40yo. LVH"


def test_build_query_text_normalizes_sex_strings() -> None:
    from backend.ptbxl_service import _build_query_text
    assert "male" in _build_query_text(58, "M", "STEMI")
    assert "female" in _build_query_text(72, "F", "AF")


# ── FAISS index graceful fallback ────────────────────────────────────────────

def test_faiss_index_unavailable_when_files_missing(tmp_path: Path) -> None:
    from backend.ptbxl_service import _PTBXLFaiss
    idx = _PTBXLFaiss(index_dir=tmp_path)
    assert idx.available is False
    assert idx.search("anterior STEMI") is None
