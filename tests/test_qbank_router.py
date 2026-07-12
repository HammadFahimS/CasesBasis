from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend import qbank_store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "qbank_router.db"))
    monkeypatch.setenv("APP_SECRET", "test-secret-qbank-router")
    from backend.db import run_migrations
    from backend.persistence import get_or_create_resident
    run_migrations()
    qbank_store.load_bundled_cases.cache_clear()
    qbank_store.seed_qbank_cases()
    return get_or_create_resident("router@test.com")


class Req:
    def __init__(self, resident_id):
        from backend.auth import COOKIE_NAME, make_session_cookie
        self.cookies = {COOKIE_NAME: make_session_cookie(resident_id)}


def test_facets_and_totals(env):
    from backend.routers.qbank import api_qbank_facets
    out = api_qbank_facets(Req(env))
    assert out["totals"] == {"cases": 30, "checkpoints": 55}
    assert out["facets"]["subject"]


def test_count_reflects_filters(env):
    from backend.routers.qbank import api_qbank_count
    out = api_qbank_count({"filters": {"subject": "Endocrinology"}}, Req(env))
    assert out == {"cases": 4, "checkpoints": 8}


def test_session_returns_full_cases_capped(env):
    from backend.routers.qbank import api_qbank_session
    out = api_qbank_session({"filters": {"subject": "Endocrinology"}, "mode": "tutor"}, Req(env))
    assert out["mode"] == "tutor"
    assert len(out["cases"]) == 4
    assert all("checkpoints" in c for c in out["cases"])
    assert "seen_checkpoint_ids" in out


def test_session_no_match_is_404(env):
    from backend.routers.qbank import api_qbank_session
    with pytest.raises(HTTPException) as exc:
        api_qbank_session({"filters": {"subject": "Nonexistent"}}, Req(env))
    assert exc.value.status_code == 404


def test_attempt_grades_server_side_and_records(env):
    from backend.routers.qbank import api_qbank_attempt, api_qbank_stats, api_qbank_session

    case = api_qbank_session({"filters": {"subject": "Endocrinology"}}, Req(env))["cases"][0]
    cp = case["checkpoints"][0]
    correct_idx = next(i for i, c in enumerate(cp["choices"]) if c["correct"])
    wrong_idx = next(i for i, c in enumerate(cp["choices"]) if not c["correct"])

    # a correct pick
    r1 = api_qbank_attempt(
        {"case_id": case["case_id"], "checkpoint_id": cp["checkpoint_id"],
         "chosen_index": correct_idx, "mode": "tutor"}, Req(env))
    assert r1["correct"] is True
    assert r1["correct_index"] == correct_idx

    # a wrong pick on a second checkpoint (or same) — grading is server-side,
    # so the client can't spoof it by omitting/forging a flag
    r2 = api_qbank_attempt(
        {"case_id": case["case_id"], "checkpoint_id": cp["checkpoint_id"],
         "chosen_index": wrong_idx, "mode": "tutor"}, Req(env))
    assert r2["correct"] is False

    st = api_qbank_stats(Req(env))
    assert st["total"] == 2
    assert st["correct"] == 1


def test_attempt_unknown_case_is_404(env):
    from backend.routers.qbank import api_qbank_attempt
    with pytest.raises(HTTPException) as exc:
        api_qbank_attempt({"case_id": "nope", "checkpoint_id": "x", "chosen_index": 0}, Req(env))
    assert exc.value.status_code == 404


def test_attempt_out_of_range_index_is_422(env):
    from backend.routers.qbank import api_qbank_attempt, api_qbank_session
    case = api_qbank_session({"filters": {"subject": "Endocrinology"}}, Req(env))["cases"][0]
    cp = case["checkpoints"][0]
    with pytest.raises(HTTPException) as exc:
        api_qbank_attempt(
            {"case_id": case["case_id"], "checkpoint_id": cp["checkpoint_id"], "chosen_index": 99},
            Req(env))
    assert exc.value.status_code == 422


def test_requires_authentication(env):
    from backend.routers.qbank import api_qbank_facets

    class Anon:
        cookies = {}

    with pytest.raises(HTTPException) as exc:
        api_qbank_facets(Anon())
    assert exc.value.status_code == 401
