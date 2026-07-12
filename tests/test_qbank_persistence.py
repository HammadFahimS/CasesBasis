from __future__ import annotations

import pytest

from backend import qbank_persistence as P
from backend import qbank_store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Migrated DB, qbank cases seeded, and one real resident."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "qbank_persist.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-qbank-persist")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    run_migrations()
    qbank_store.load_bundled_cases.cache_clear()
    qbank_store.seed_qbank_cases()
    return get_or_create_resident("qbank@test.com")


def test_record_attempt_and_overall_stats(env):
    P.record_attempt(env, "case_a", "cp1", True, 0, "tutor", "Cardiology", "Cardiovascular")
    P.record_attempt(env, "case_a", "cp2", False, 1, "tutor", "Cardiology", "Cardiovascular")
    P.record_attempt(env, "case_b", "cp3", True, 2, "test", "Neurology", "Nervous")

    st = P.stats(env)
    assert st["total"] == 3
    assert st["correct"] == 2
    assert st["percent"] == 67
    assert st["answered_checkpoints"] == 3


def test_stats_break_down_by_subject_and_system(env):
    P.record_attempt(env, "c", "cp1", True, 0, "tutor", "Cardiology", "Cardiovascular")
    P.record_attempt(env, "c", "cp2", False, 0, "tutor", "Cardiology", "Cardiovascular")
    P.record_attempt(env, "c", "cp3", True, 0, "tutor", "Neurology", "Nervous")

    st = P.stats(env)
    subs = {r["key"]: r for r in st["by_subject"]}
    assert subs["Cardiology"]["total"] == 2 and subs["Cardiology"]["percent"] == 50
    assert subs["Neurology"]["percent"] == 100
    systems = {r["key"]: r for r in st["by_system"]}
    assert systems["Cardiovascular"]["total"] == 2


def test_answered_checkpoint_ids_are_distinct(env):
    P.record_attempt(env, "c", "cp1", True, 0)
    P.record_attempt(env, "c", "cp1", False, 1)   # re-answered the same checkpoint
    P.record_attempt(env, "c", "cp2", True, 0)
    assert P.answered_checkpoint_ids(env) == {"cp1", "cp2"}


def test_empty_stats_are_zeroed(env):
    st = P.stats(env)
    assert st == {
        "total": 0, "correct": 0, "percent": 0, "answered_checkpoints": 0,
        "by_subject": [], "by_system": [],
    }
