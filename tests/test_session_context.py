from __future__ import annotations

from schemas.session_context import build_session_context
from tests.test_performance_tracker import make_evaluation


def test_build_session_context_summarizes_recent_evaluations() -> None:
    history = [
        make_evaluation(
            60,
            missed_symptoms=["fever"],
            missed_steps=["Clarify timing and progression of symptoms."],
        ),
        make_evaluation(
            70,
            missed_symptoms=["fever", "dizziness"],
            missed_steps=["Clarify timing and progression of symptoms."],
        ),
    ]

    context = build_session_context(history)

    assert context.recent_scores == [60, 70]
    assert context.repeated_missed_symptoms[0] == "fever"
    assert context.repeated_reasoning_weaknesses[0] == "Clarify timing and progression of symptoms."
    assert context.learning_focus_areas


def test_build_session_context_accepts_dict_and_json() -> None:
    evaluation = make_evaluation(45, missed_symptoms=["confusion"], missed_steps=["Check vitals."])

    context = build_session_context([evaluation.model_dump(), evaluation.model_dump_json()])

    assert context.recent_scores == [45, 45]
    assert context.repeated_missed_symptoms == ["confusion"]
    assert context.repeated_reasoning_weaknesses == ["Check vitals."]
