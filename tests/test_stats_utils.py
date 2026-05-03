from __future__ import annotations

from utils.common import most_common, most_common_values, normalize_history


def test_normalize_history_strips_and_drops_empty_values() -> None:
    assert normalize_history([" fever ", "", "  ", "dizziness"]) == ["fever", "dizziness"]


def test_most_common_returns_top_value_or_none() -> None:
    assert most_common([" fever ", "dizziness", "fever", " "]) == "fever"
    assert most_common([]) is None
    assert most_common([" ", ""]) is None


def test_most_common_preserves_first_seen_tie_order() -> None:
    assert most_common(["dizziness", "fever"]) == "dizziness"


def test_most_common_values_returns_limited_ranked_values() -> None:
    values = ["fever", "dizziness", "fever", "confusion", "dizziness", "fever"]

    assert most_common_values(values, limit=2) == ["fever", "dizziness"]
