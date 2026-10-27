"""Offline tests for the latency harness.

The probes themselves hit the network by design and are not exercised here. What
*is* tested is the part that decides whether an optimisation worked — because a
comparison tool that reports false wins is worse than no tool at all.
"""

from __future__ import annotations

import json

import pytest

from bench.compare import compare
from bench.stats import bootstrap_ci_median_diff, judge, percentile, summarize


# ── percentile ───────────────────────────────────────────────────────────────

def test_percentile_endpoints_and_median():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 0.5) == 3.0
    assert percentile(xs, 1.0) == 5.0


def test_percentile_interpolates_between_samples():
    assert percentile([0.0, 10.0], 0.5) == 5.0


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.5) == 0.0


def test_percentile_ignores_input_order():
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0


# ── summarize ────────────────────────────────────────────────────────────────

def test_summarize_reports_tail_not_just_centre():
    # One slow outlier must show up in p95/max without dragging p50 with it.
    s = summarize([10.0] * 19 + [500.0])
    assert s.n == 20
    assert s.p50 == 10.0
    assert s.maximum == 500.0
    assert s.p95 > s.p50


def test_summarize_empty_is_all_zero():
    s = summarize([])
    assert s.n == 0 and s.p50 == 0.0 and s.p95 == 0.0


def test_summarize_single_sample_has_no_spread():
    s = summarize([42.0])
    assert s.n == 1 and s.p50 == 42.0 and s.stdev == 0.0


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_bootstrap_ci_is_deterministic_for_a_given_seed():
    a = [100.0, 105.0, 98.0, 102.0, 110.0]
    b = [70.0, 72.0, 68.0, 75.0, 71.0]
    assert bootstrap_ci_median_diff(a, b, seed=7) == bootstrap_ci_median_diff(a, b, seed=7)


def test_bootstrap_ci_excludes_zero_for_a_clear_shift():
    before = [100.0 + i * 0.5 for i in range(20)]
    after = [40.0 + i * 0.5 for i in range(20)]
    lo, hi = bootstrap_ci_median_diff(before, after)
    assert hi < 0, "a 60ms speedup must produce a CI entirely below zero"


def test_bootstrap_ci_straddles_zero_for_identical_distributions():
    xs = [50.0, 55.0, 45.0, 60.0, 40.0, 52.0, 48.0, 58.0]
    lo, hi = bootstrap_ci_median_diff(xs, list(xs))
    assert lo <= 0.0 <= hi


def test_bootstrap_ci_of_empty_side_is_degenerate():
    assert bootstrap_ci_median_diff([], [1.0, 2.0]) == (0.0, 0.0)


# ── judge: the guard against false wins ──────────────────────────────────────

def test_judge_calls_a_real_speedup_improved():
    before = [200.0 + i for i in range(20)]
    after = [120.0 + i for i in range(20)]
    v = judge("llm_speak", before, after)
    assert v.label == "IMPROVED"
    assert v.delta_ms < 0


def test_judge_calls_a_real_slowdown_regressed():
    before = [120.0 + i for i in range(20)]
    after = [200.0 + i for i in range(20)]
    assert judge("llm_speak", before, after).label == "REGRESSED"


def test_judge_calls_noisy_identical_runs_noise():
    # Same underlying distribution, different draws — the common false-positive case.
    before = [100.0, 130.0, 90.0, 145.0, 85.0, 120.0, 110.0, 95.0, 140.0, 105.0]
    after = [95.0, 135.0, 100.0, 140.0, 88.0, 125.0, 115.0, 92.0, 138.0, 108.0]
    assert judge("tts_ttfb", before, after).label == "NOISE"


def test_judge_respects_the_absolute_effect_floor():
    """A tight, consistent 2ms shift is statistically real but operationally
    irrelevant — the floor is what stops us shipping it as a win."""
    before = [100.0 + i * 0.01 for i in range(40)]
    after = [98.0 + i * 0.01 for i in range(40)]
    v = judge("dispatch", before, after, min_effect_ms=5.0)
    assert v.label == "NOISE"
    assert "floor" in v.note
    # Same data, a floor low enough to care: now it counts.
    assert judge("dispatch", before, after, min_effect_ms=0.5).label == "IMPROVED"


def test_judge_reports_percentage_against_the_baseline():
    before = [100.0] * 10
    after = [50.0] * 10
    v = judge("turn_total", before, after)
    assert v.delta_pct == pytest.approx(-50.0)


# ── compare: run-to-run diffing ──────────────────────────────────────────────

def _run(label, probes):
    return {
        "label": label,
        "git": {"sha": "abc1234", "branch": "main", "dirty": False},
        "probes": [{"name": n, "tier": "network", "samples": s} for n, s in probes.items()],
    }


def test_compare_matches_probes_by_name():
    before = _run("baseline", {"tts_ttfb": [200.0 + i for i in range(20)]})
    after = _run("tuned", {"tts_ttfb": [120.0 + i for i in range(20)]})
    (v,) = compare(before, after)
    assert v.probe == "tts_ttfb" and v.label == "IMPROVED"


def test_compare_flags_probes_added_or_removed():
    before = _run("baseline", {"a": [10.0, 11.0]})
    after = _run("tuned", {"b": [10.0, 11.0]})
    labels = {v.probe: v.label for v in compare(before, after)}
    assert labels == {"a": "GONE", "b": "NEW"}


def test_compare_skips_probes_with_no_samples():
    """A skipped probe (missing API key) must not be diffed as if it were data."""
    before = {"label": "b", "probes": [{"name": "tts_ttfb", "samples": [], "skipped": "no key"}]}
    after = {"label": "a", "probes": [{"name": "tts_ttfb", "samples": [], "skipped": "no key"}]}
    assert compare(before, after) == []


def test_verdict_serialises_to_json():
    v = judge("turn_total", [200.0] * 10, [100.0] * 10)
    payload = json.dumps(v.to_dict())
    assert json.loads(payload)["label"] == "IMPROVED"


# ── probe registry wiring ────────────────────────────────────────────────────

def test_every_registered_probe_is_callable():
    from bench.probes import PROBES, PURE_PROBES
    assert PROBES, "registry must not be empty"
    assert all(callable(fn) for fn in PROBES.values())
    assert set(PURE_PROBES) <= set(PROBES)


def test_pure_probes_need_no_credentials():
    """The no-network tier must stay runnable in CI with no keys set."""
    from bench.probes import PURE_PROBES
    assert set(PURE_PROBES) == {"route", "dispatch", "engine_tick"}


def test_probe_result_reports_stats_for_its_samples():
    from bench.probes import ProbeResult
    d = ProbeResult("x", "pure", samples=[1.0, 2.0, 3.0]).to_dict()
    assert d["stats"]["n"] == 3 and d["stats"]["p50"] == 2.0


def test_probe_result_carries_skip_reason():
    from bench.probes import ProbeResult
    d = ProbeResult("x", "network", skipped="no key").to_dict()
    assert d["skipped"] == "no key" and d["stats"]["n"] == 0


# ── cross-probe mapping ──────────────────────────────────────────────────────

def test_map_compares_a_retired_probe_against_its_replacement():
    """Streaming STT retires stt_blocking rather than speeding it up. Without a
    mapping the two sides read as GONE/NEW and the real comparison never happens."""
    before = _run("baseline", {"stt_blocking": [520.0 + i for i in range(20)]})
    after = _run("streaming", {"stt_realtime": [110.0 + i for i in range(20)]})
    (v,) = compare(before, after, rename={"stt_blocking": "stt_realtime"})
    assert v.probe == "stt_blocking→stt_realtime"
    assert v.label == "IMPROVED"
    assert v.delta_ms < -350
    assert "cross-probe" in v.note


def test_map_is_ignored_when_either_side_is_absent():
    before = _run("baseline", {"stt_blocking": [520.0, 530.0]})
    after = _run("streaming", {"stt_blocking": [520.0, 530.0]})
    labels = {v.probe: v.label for v in compare(before, after, rename={"stt_blocking": "nope"})}
    assert "stt_blocking" in labels and "stt_blocking→nope" not in labels


def test_mapped_replacement_is_not_also_reported_as_new():
    before = _run("baseline", {"stt_blocking": [520.0] * 10, "tts_ttfb": [150.0] * 10})
    after = _run("streaming", {"stt_realtime": [110.0] * 10, "tts_ttfb": [150.0] * 10})
    names = [v.probe for v in compare(before, after, rename={"stt_blocking": "stt_realtime"})]
    assert names.count("stt_realtime") == 0, "the replacement must not double-report as NEW"
    assert "stt_blocking→stt_realtime" in names and "tts_ttfb" in names


def test_a_retired_probe_still_gets_its_own_row_when_still_measured():
    """stt_blocking survives as the fallback path. Mapping it to stt_realtime must
    not make its own before/after comparison vanish — or, worse, report it as NEW."""
    before = _run("baseline", {"stt_blocking": [520.0 + i for i in range(10)]})
    after = _run("streaming", {
        "stt_blocking": [521.0 + i for i in range(10)],
        "stt_realtime": [110.0 + i for i in range(10)],
    })
    labels = {v.probe: v.label for v in compare(before, after,
                                                rename={"stt_blocking": "stt_realtime"})}
    assert labels["stt_blocking→stt_realtime"] == "IMPROVED"
    assert labels["stt_blocking"] == "NOISE", "the surviving fallback keeps its own row"
    assert "NEW" not in labels.values()
