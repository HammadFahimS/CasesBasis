from __future__ import annotations

import subprocess
from typing import Any

import pytest

from backend.audit_fix_loop import (
    SCENARIOS,
    ensure_clean_worktree,
    run_audit_fix_loop,
    run_audited_scenarios,
)
from schemas.evaluation import (
    ClinicalEvaluation,
    DiagnosisGuessEvaluation,
    InvestigationInterpretationFeedback,
    LearningRecommendations,
    QuestionRelevanceEvaluation,
    ReasoningFeedback,
)
from schemas.simulation import SimulationResult
from schemas.simulation_audit import SimulationAudit


class FakeAuditLoopRunner:
    def __init__(self, audit: SimulationAudit | None = None) -> None:
        self.audit = audit or pass_audit()
        self.calls: list[dict[str, Any]] = []

    def run_once(self, **kwargs: Any) -> SimulationResult:
        self.calls.append(kwargs)
        return simulation_result(
            diagnosis=kwargs["diagnosis"],
            difficulty=kwargs["difficulty"],
            audit=self.audit,
        )


class CrashingRunner:
    def run_once(self, **_: Any) -> SimulationResult:
        raise RuntimeError("model returned invalid case")


def test_audit_fix_loop_refuses_dirty_worktree() -> None:
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        ensure_clean_worktree(
            allow_dirty=False,
            git_status_func=lambda: " M backend/file.py",
        )


def test_audit_fix_loop_allows_dirty_when_explicit() -> None:
    ensure_clean_worktree(
        allow_dirty=True,
        git_status_func=lambda: " M backend/file.py",
    )


def test_audit_fix_loop_runs_fast_audited_scenarios_and_checks() -> None:
    runner = FakeAuditLoopRunner()
    commands: list[list[str]] = []

    report = run_audit_fix_loop(
        runner=runner,
        allow_dirty=True,
        check_runner=fake_check_runner(commands),
    )

    assert report.iterations_completed == 1
    assert report.bugs_found == []
    assert report.bugs_fixed == []
    assert report.tests_added == []
    assert report.pytest_status == "passed"
    assert report.compile_status == "passed"
    assert len(runner.calls) == len(SCENARIOS)
    assert all(call["fast_mode"] is True for call in runner.calls)
    assert all(call["audit"] is True for call in runner.calls)
    assert all(call["timeout_seconds"] > 0 for call in runner.calls)
    assert commands == [
        ["pytest"],
        ["python", "-m", "compileall", "agents", "backend", "schemas"],
    ]


def test_audit_fix_loop_reports_confirmed_bugs_without_auto_fixing() -> None:
    report = run_audit_fix_loop(
        runner=FakeAuditLoopRunner(confirmed_bug_audit()),
        allow_dirty=True,
        run_checks=False,
    )

    assert report.iterations_completed == 1
    assert len(report.bugs_found) == len(SCENARIOS)
    assert all(bug["category"] == "confirmed_bug" for bug in report.bugs_found)
    assert report.bugs_fixed == []
    assert report.tests_added == []
    assert report.pytest_status == "not_run"
    assert report.compile_status == "not_run"


def test_audit_fix_loop_records_crashes_as_warnings() -> None:
    records = run_audited_scenarios(
        CrashingRunner(),  # type: ignore[arg-type]
        timeout_seconds=1,
        iteration=1,
    )

    assert len(records) == len(SCENARIOS)
    assert all(record.status == "crashed" for record in records)
    assert all(record.confirmed_bugs == [] for record in records)
    assert all(record.warnings for record in records)
    assert "RuntimeError" in records[0].error


def test_audit_fix_loop_failed_checks_are_reported() -> None:
    commands: list[list[str]] = []

    report = run_audit_fix_loop(
        runner=FakeAuditLoopRunner(),
        allow_dirty=True,
        check_runner=fake_check_runner(commands, failing_command="pytest"),
    )

    assert report.pytest_status == "failed"
    assert report.compile_status == "passed"


def fake_check_runner(
    commands: list[list[str]],
    *,
    failing_command: str | None = None,
) -> Any:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1 if command[0] == failing_command else 0,
            stdout="",
            stderr="",
        )

    return run


def simulation_result(
    *,
    diagnosis: str,
    difficulty: str,
    audit: SimulationAudit,
) -> SimulationResult:
    return SimulationResult(
        diagnosis=diagnosis,
        difficulty=difficulty,
        student_questions=[],
        patient_responses=[],
        investigations_ordered=[],
        student_interpretations=[],
        doctor_consults=[],
        diagnosis_guess=None,
        steps=[],
        evaluation=ClinicalEvaluation(
            correct_diagnosis_guess=DiagnosisGuessEvaluation(
                provided=False,
                guess=None,
                is_correct=None,
                diagnosis_match_level=None,
            ),
            key_symptoms_identified=[],
            key_symptoms_missed=["chest pressure"],
            relevance_of_questions=QuestionRelevanceEvaluation(
                relevant_count=0,
                irrelevant_count=0,
                relevant_questions=[],
                irrelevant_questions=[],
            ),
            overall_score=25,
            brief_feedback="feedback",
            reasoning_feedback=ReasoningFeedback(
                diagnostic_path="path",
                question_order="order",
                missed_reasoning_steps=["Ask about timing."],
            ),
            learning_recommendations=LearningRecommendations(
                primary_skill_gap="gap",
                focus_next="focus",
                action_items=["action"],
            ),
            investigation_interpretation_feedback=InvestigationInterpretationFeedback(),
        ),
        audit=audit,
    )


def pass_audit() -> SimulationAudit:
    return SimulationAudit.model_validate(
        {
            "overall_quality": "pass",
            "findings": [
                {
                    "category": "expected_behavior",
                    "severity": "low",
                    "issue": "No deterministic issue found.",
                    "evidence": "Run completed and evaluator output was coherent.",
                    "suggested_fix": "No fix needed.",
                }
            ],
            "should_fix": False,
        }
    )


def confirmed_bug_audit() -> SimulationAudit:
    return SimulationAudit.model_validate(
        {
            "overall_quality": "fail",
            "findings": [
                {
                    "category": "confirmed_bug",
                    "severity": "high",
                    "issue": "Hidden diagnosis leaked.",
                    "evidence": "Student-visible context included diagnosis.",
                    "suggested_fix": "Redact diagnosis before student prompt.",
                }
            ],
            "should_fix": True,
        }
    )
