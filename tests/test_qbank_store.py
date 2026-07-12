from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend import qbank_store
from schemas.qbank import QBankCase


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    """A migrated DB with the real bundled qbank cases seeded."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "qbank_store.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-qbank-store")
    from backend.db import run_migrations
    run_migrations()
    qbank_store.load_bundled_cases.cache_clear()
    qbank_store.seed_qbank_cases()
    yield


# ── schema validation ────────────────────────────────────────────────────────

def _case(**over):
    base = {
        "case_id": "c1", "stem": "A vignette.",
        "checkpoints": [{
            "checkpoint_id": "c1_cp1", "sequence": 1, "question": "Q?",
            "choices": [
                {"text": "right", "correct": True, "explanation": "because"},
                {"text": "wrong", "correct": False, "explanation": "no"},
            ],
        }],
        "tags": {"subject": "Cardiology", "system": "Cardiovascular", "difficulty": 3, "high_yield": True},
    }
    base.update(over)
    return base


def test_valid_case_parses_and_sorts_checkpoints():
    case = QBankCase.model_validate(_case(checkpoints=[
        {"checkpoint_id": "b", "sequence": 2, "question": "Q2",
         "choices": [{"text": "r", "correct": True}, {"text": "w", "correct": False}]},
        {"checkpoint_id": "a", "sequence": 1, "question": "Q1",
         "choices": [{"text": "r", "correct": True}, {"text": "w", "correct": False}]},
    ]))
    assert [c.checkpoint_id for c in case.checkpoints] == ["a", "b"]  # sorted by sequence


def test_checkpoint_must_have_exactly_one_correct_choice():
    with pytest.raises(ValidationError):
        QBankCase.model_validate(_case(checkpoints=[{
            "checkpoint_id": "x", "sequence": 1, "question": "Q",
            "choices": [{"text": "a", "correct": True}, {"text": "b", "correct": True}],
        }]))
    with pytest.raises(ValidationError):
        QBankCase.model_validate(_case(checkpoints=[{
            "checkpoint_id": "x", "sequence": 1, "question": "Q",
            "choices": [{"text": "a", "correct": False}, {"text": "b", "correct": False}],
        }]))


def test_duplicate_checkpoint_ids_rejected():
    with pytest.raises(ValidationError):
        QBankCase.model_validate(_case(checkpoints=[
            {"checkpoint_id": "dup", "sequence": 1, "question": "Q1",
             "choices": [{"text": "r", "correct": True}, {"text": "w", "correct": False}]},
            {"checkpoint_id": "dup", "sequence": 2, "question": "Q2",
             "choices": [{"text": "r", "correct": True}, {"text": "w", "correct": False}]},
        ]))


# ── loading is fail-soft ──────────────────────────────────────────────────────

def test_load_drops_malformed_cases_but_keeps_valid(tmp_path, monkeypatch):
    good = _case(case_id="good")
    bad = _case(case_id="bad", checkpoints=[{  # two correct → invalid
        "checkpoint_id": "bad_cp", "sequence": 1, "question": "Q",
        "choices": [{"text": "a", "correct": True}, {"text": "b", "correct": True}],
    }])
    dup = _case(case_id="good")  # duplicate id → dropped
    f = tmp_path / "mix.json"
    f.write_text(json.dumps([good, bad, dup]))
    monkeypatch.setattr(qbank_store, "DATA_PATH", f)
    qbank_store.load_bundled_cases.cache_clear()
    cases = qbank_store.load_bundled_cases()
    assert [c.case_id for c in cases] == ["good"]


def test_missing_file_yields_empty_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(qbank_store, "DATA_PATH", tmp_path / "does-not-exist.json")
    qbank_store.load_bundled_cases.cache_clear()
    assert qbank_store.load_bundled_cases() == []


# ── seeding + reads ───────────────────────────────────────────────────────────

def test_seed_is_idempotent_and_counts_match(seeded):
    totals = qbank_store.count_matching({})
    assert totals["cases"] == 30
    assert totals["checkpoints"] == 55
    # re-seeding writes the same rows, not extra ones
    qbank_store.seed_qbank_cases()
    assert qbank_store.count_matching({})["cases"] == 30


def test_filters_narrow_the_pool(seeded):
    endo = qbank_store.count_matching({"subject": "Endocrinology"})
    assert endo["cases"] == 4
    # multi-value OR across a facet
    multi = qbank_store.count_matching({"subject": ["Endocrinology", "Neurology"]})
    assert multi["cases"] == 6
    # difficulty + high_yield combine (AND)
    d4 = qbank_store.count_matching({"difficulty": 4})
    assert d4["cases"] == 2


def test_facets_expose_values_with_counts(seeded):
    fac = qbank_store.facets()
    assert {f["value"] for f in fac["subject"]} >= {"Endocrinology", "Neurology"}
    assert sum(f["count"] for f in fac["difficulty"]) == 30


def test_get_case_returns_full_document_with_choices(seeded):
    rows = qbank_store.list_matching({"subject": "Endocrinology"})
    assert rows
    case = qbank_store.get_case(rows[0]["case_id"])
    assert case["case_id"] == rows[0]["case_id"]
    assert case["checkpoints"]
    assert all("choices" in cp for cp in case["checkpoints"])
    # exactly one correct choice per checkpoint survives the round-trip
    for cp in case["checkpoints"]:
        assert sum(1 for ch in cp["choices"] if ch["correct"]) == 1


def test_get_case_unknown_id_is_none(seeded):
    assert qbank_store.get_case("nope") is None
