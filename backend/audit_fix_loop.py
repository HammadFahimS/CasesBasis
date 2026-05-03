from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv

from backend.simulation_runner import FAST_MAX_STEPS, SimulationRunner
from schemas.simulation import SimulationResult
from schemas.simulation_audit import SimulationAuditFinding

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_ITERATIONS = 3
SCENARIOS = (
    ("myocardial infarction", "medium"),
    ("stroke", "medium"),
    ("asthma exacerbation", "medium"),
)


@dataclass
class LoopRunRecord:
    diagnosis: str
    difficulty: str
    status: str
    score: int | None = None
    confirmed_bugs: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


@dataclass
class LoopReport:
    iterations_completed: int
    bugs_found: list[dict[str, str]]
    bugs_fixed: list[str]
    tests_added: list[str]
    remaining_warnings: list[dict[str, str]]
    pytest_status: str
    compile_status: str
    runs: list[LoopRunRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "iterations_completed": self.iterations_completed,
            "bugs_found": self.bugs_found,
            "bugs_fixed": self.bugs_fixed,
            "tests_added": self.tests_added,
            "remaining_warnings": self.remaining_warnings,
            "pytest_status": self.pytest_status,
            "compile_status": self.compile_status,
            "runs": [
                {
                    "diagnosis": run.diagnosis,
                    "difficulty": run.difficulty,
                    "status": run.status,
                    "score": run.score,
                    "confirmed_bugs": run.confirmed_bugs,
                    "warnings": run.warnings,
                    "error": run.error,
                }
                for run in self.runs
            ],
        }


def run_audit_fix_loop(
    *,
    runner: SimulationRunner | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    allow_dirty: bool = False,
    run_checks: bool = True,
    git_status_func: Callable[[], str] | None = None,
    check_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> LoopReport:
    active_git_status = git_status_func or git_status_short
    active_check_runner = check_runner or run_command
    ensure_clean_worktree(allow_dirty=allow_dirty, git_status_func=active_git_status)
    active_runner = runner or SimulationRunner()
    all_runs: list[LoopRunRecord] = []
    bugs_found: list[dict[str, str]] = []
    remaining_warnings: list[dict[str, str]] = []
    bugs_fixed: list[str] = []
    tests_added: list[str] = []
    pytest_status = "not_run"
    compile_status = "not_run"

    for iteration in range(1, max_iterations + 1):
        iteration_runs = run_audited_scenarios(
            active_runner,
            timeout_seconds=timeout_seconds,
            iteration=iteration,
        )
        all_runs.extend(iteration_runs)
        confirmed = [
            finding
            for run in iteration_runs
            for finding in run.confirmed_bugs
        ]
        warnings = [
            finding
            for run in iteration_runs
            for finding in run.warnings
        ]
        bugs_found.extend(confirmed)
        remaining_warnings = warnings

        if not confirmed:
            if run_checks:
                pytest_status, compile_status = run_required_checks(active_check_runner)
            return LoopReport(
                iterations_completed=iteration,
                bugs_found=bugs_found,
                bugs_fixed=bugs_fixed,
                tests_added=tests_added,
                remaining_warnings=remaining_warnings,
                pytest_status=pytest_status,
                compile_status=compile_status,
                runs=all_runs,
            )

        # The auditor is advisory and must not modify code. Confirmed deterministic
        # bugs should be fixed by code changes plus regression tests, then this loop
        # can be rerun.
        if run_checks:
            pytest_status, compile_status = run_required_checks(active_check_runner)
        return LoopReport(
            iterations_completed=iteration,
            bugs_found=bugs_found,
            bugs_fixed=bugs_fixed,
            tests_added=tests_added,
            remaining_warnings=remaining_warnings,
            pytest_status=pytest_status,
            compile_status=compile_status,
            runs=all_runs,
        )

    if run_checks:
        pytest_status, compile_status = run_required_checks(active_check_runner)
    return LoopReport(
        iterations_completed=max_iterations,
        bugs_found=bugs_found,
        bugs_fixed=bugs_fixed,
        tests_added=tests_added,
        remaining_warnings=remaining_warnings,
        pytest_status=pytest_status,
        compile_status=compile_status,
        runs=all_runs,
    )


def run_audited_scenarios(
    runner: SimulationRunner,
    *,
    timeout_seconds: float,
    iteration: int,
) -> list[LoopRunRecord]:
    records: list[LoopRunRecord] = []
    for diagnosis, difficulty in SCENARIOS:
        try:
            result = runner.run_once(
                diagnosis=diagnosis,
                difficulty=difficulty,
                max_steps=FAST_MAX_STEPS,
                timeout_seconds=timeout_seconds,
                fast_mode=True,
                audit=True,
            )
            records.append(record_completed_run(result, iteration=iteration))
        except Exception as exc:
            records.append(
                LoopRunRecord(
                    diagnosis=diagnosis,
                    difficulty=difficulty,
                    status="crashed",
                    error=f"{type(exc).__name__}: {exc}",
                    warnings=[
                        {
                            "iteration": str(iteration),
                            "diagnosis": diagnosis,
                            "category": "possible_model_quality_issue",
                            "severity": "high",
                            "issue": "Simulation run crashed before audit completion.",
                            "evidence": f"{type(exc).__name__}: {exc}",
                            "suggested_fix": "Inspect whether this is deterministic code failure or model output quality before changing code.",
                        }
                    ],
                )
            )
    return records


def record_completed_run(result: SimulationResult, *, iteration: int) -> LoopRunRecord:
    confirmed: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if result.audit:
        for finding in result.audit.findings:
            payload = finding_payload(finding, result=result, iteration=iteration)
            if finding.category == "confirmed_bug":
                confirmed.append(payload)
            elif finding.category == "possible_model_quality_issue":
                warnings.append(payload)
    return LoopRunRecord(
        diagnosis=result.diagnosis,
        difficulty=result.difficulty,
        status="completed",
        score=result.evaluation.overall_score,
        confirmed_bugs=confirmed,
        warnings=warnings,
    )


def finding_payload(
    finding: SimulationAuditFinding,
    *,
    result: SimulationResult,
    iteration: int,
) -> dict[str, str]:
    return {
        "iteration": str(iteration),
        "diagnosis": result.diagnosis,
        "category": finding.category,
        "severity": finding.severity,
        "issue": finding.issue,
        "evidence": finding.evidence,
        "suggested_fix": finding.suggested_fix,
    }


def ensure_clean_worktree(
    *,
    allow_dirty: bool,
    git_status_func: Callable[[], str],
) -> None:
    status = git_status_func().strip()
    if status and not allow_dirty:
        raise RuntimeError(
            "Working tree has uncommitted changes. Commit/stash them or rerun with --allow-dirty."
        )


def git_status_short() -> str:
    return run_command(["git", "status", "--short"]).stdout


def run_required_checks(
    check_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> tuple[str, str]:
    pytest_result = check_runner(["pytest"])
    compile_result = check_runner(["python", "-m", "compileall", "agents", "backend", "schemas"])
    return status_from_returncode(pytest_result.returncode), status_from_returncode(
        compile_result.returncode
    )


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )


def status_from_returncode(returncode: int) -> str:
    return "passed" if returncode == 0 else "failed"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a safe simulation audit-fix loop.")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    try:
        report = run_audit_fix_loop(
            max_iterations=args.max_iterations,
            timeout_seconds=args.timeout_seconds,
            allow_dirty=args.allow_dirty,
            run_checks=not args.skip_checks,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error": f"{type(exc).__name__}: {exc}",
                    "bugs_found": [],
                    "bugs_fixed": [],
                    "tests_added": [],
                    "remaining_warnings": [],
                    "pytest_status": "not_run",
                    "compile_status": "not_run",
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.pytest_status != "failed" and report.compile_status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
