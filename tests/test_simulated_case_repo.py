from __future__ import annotations

import pytest

from backend.simulated_case_repo import (
    CaseProtectedError,
    CaseValidationError,
    create_case,
    delete_case,
    duplicate_case,
    get_case,
    list_cases,
    reseed_gold_case,
    seed_gold_cases,
    update_case,
)
from backend.simulated_case_store import load_seed_case


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "sim_repo_test.db")
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("APP_SECRET", "test-secret-sim-repo")
    from backend.db import run_migrations
    run_migrations()
    seed_gold_cases()
    yield path


def test_seed_inserts_gold_case(db_path):
    cases = list_cases()
    gold = next(c for c in cases if c["id"] == "myasthenic_crisis")
    assert gold["is_gold"] is True
    assert gold["state_count"] == 6


def test_seed_is_idempotent(db_path):
    before = len(list_cases())
    seed_gold_cases()
    assert len(list_cases()) == before


def test_reseed_overwrites_gold_row_from_file(db_path, monkeypatch):
    # seed_gold_cases never clobbers an existing row; reseed_gold_case deliberately does.
    # Deep-copy so we don't mutate load_seed_case's cached instance (shared across tests).
    edited = load_seed_case("myasthenic_crisis").model_copy(deep=True)
    edited.meta.title = "Edited Title"
    monkeypatch.setattr(
        "backend.simulated_case_repo.load_seed_case", lambda _id: edited
    )
    assert reseed_gold_case("myasthenic_crisis") is True
    assert get_case("myasthenic_crisis")["case"]["meta"]["title"] == "Edited Title"
    # is_gold protection is preserved through the reseed
    assert get_case("myasthenic_crisis")["is_gold"] is True


def test_reseed_returns_false_when_row_absent(db_path, monkeypatch):
    fake = load_seed_case("myasthenic_crisis").model_copy(deep=True)
    fake.id = "not_seeded_case"
    monkeypatch.setattr(
        "backend.simulated_case_repo.load_seed_case", lambda _id: fake
    )
    assert reseed_gold_case("not_seeded_case") is False


def test_gold_cannot_be_updated_or_deleted(db_path):
    payload = load_seed_case("myasthenic_crisis").model_dump()
    with pytest.raises(CaseProtectedError):
        update_case("myasthenic_crisis", payload)
    with pytest.raises(CaseProtectedError):
        delete_case("myasthenic_crisis")


def test_duplicate_creates_editable_copy(db_path):
    copy = duplicate_case("myasthenic_crisis")
    assert copy["is_gold"] is False
    assert copy["id"] != "myasthenic_crisis"
    assert copy["case"]["meta"]["title"].endswith("(copy)")

    # The copy is fully editable.
    payload = copy["case"]
    payload["meta"]["title"] = "My edited case"
    updated = update_case(copy["id"], payload, layout={"baseline": {"x": 10, "y": 20}})
    assert updated["case"]["meta"]["title"] == "My edited case"
    assert updated["layout"] == {"baseline": {"x": 10, "y": 20}}

    delete_case(copy["id"])
    assert get_case(copy["id"]) is None


def test_create_rejects_invalid_document(db_path):
    with pytest.raises(CaseValidationError):
        create_case({"id": "broken", "meta": {"title": "X"}})  # missing required sections


def test_create_assigns_unique_id(db_path):
    base = load_seed_case("myasthenic_crisis").model_dump()
    base["id"] = "mycase"
    a = create_case(base)
    b = create_case(base)
    assert a["id"] != b["id"]
