from __future__ import annotations

import argparse
import json
from collections import Counter
from time import monotonic
from typing import Any

from dotenv import load_dotenv

from agents.simulation_auditor import SimulationAuditorAgent
from agents.student import StudentAgent
from backend.clinical_evaluator import ClinicalEvaluator
from backend.doctor_consult import DoctorConsultService
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.service import ClinicalCaseSimulator
from backend.web_app import case_preview, student_facing_investigation_result
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import ClinicalEvaluation
from schemas.investigation import InvestigationResult
from schemas.simulation import (
    SimulationResult,
    SimulationStep,
    SimulationSummary,
    StudentAction,
)

DEFAULT_MAX_STEPS = 8
FAST_MAX_STEPS = 4


class SimulationRunner:
    """Runs repeated internal end-to-end simulations with an AI student."""

    def __init__(
        self,
        *,
        simulator: ClinicalCaseSimulator | None = None,
        patient_service: PatientResponseService | None = None,
        investigation_service: InvestigationService | None = None,
        doctor_consult_service: DoctorConsultService | None = None,
        evaluator: ClinicalEvaluator | None = None,
        student_agent: StudentAgent | None = None,
        simulation_auditor: SimulationAuditorAgent | None = None,
    ) -> None:
        self.simulator = simulator or ClinicalCaseSimulator()
        self.patient_service = patient_service or PatientResponseService()
        self.investigation_service = investigation_service or InvestigationService()
        self.doctor_consult_service = doctor_consult_service
        self.evaluator = evaluator or ClinicalEvaluator()
        self.student_agent = student_agent or StudentAgent()
        self.simulation_auditor = simulation_auditor

    def run_once(
        self,
        *,
        diagnosis: str,
        difficulty: str = "medium",
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float | None = None,
        fast_mode: bool = False,
        audit: bool = False,
        skip_audit: bool = False,
        session_history: list[ClinicalEvaluation | dict[str, Any] | str] | None = None,
    ) -> SimulationResult:
        started_at = monotonic()
        case = self.simulator.generate_case(
            diagnosis,
            difficulty=difficulty,
            session_history=session_history or [],
        )
        check_timeout(started_at, timeout_seconds)
        state = SimulationState(
            case=case,
            difficulty=difficulty,
            fast_mode=fast_mode,
            max_steps=max_steps,
        )

        for step_number in range(1, max_steps + 1):
            check_timeout(started_at, timeout_seconds)
            action = self.student_agent.next_action(
                build_student_visible_context(state, step_number=step_number),
                difficulty=difficulty,
                step_number=step_number,
            )
            check_timeout(started_at, timeout_seconds)
            if action.action == "finish":
                state.steps.append(
                    SimulationStep(step_number=step_number, action="finish")
                )
                break
            self.execute_action(state, action, step_number)
            check_timeout(started_at, timeout_seconds)
            if state.diagnosis_guess:
                break

        evaluation_questions = list(state.student_questions)
        evaluation_responses = list(state.patient_responses)
        if state.diagnosis_guess:
            evaluation_questions.append(f"Diagnosis guess: {state.diagnosis_guess}")
            evaluation_responses.append("I'm not sure.")

        original_reasoning_review = getattr(self.evaluator, "enable_reasoning_review", None)
        if fast_mode and original_reasoning_review is not None:
            self.evaluator.enable_reasoning_review = False
        try:
            evaluation = self.evaluator.evaluate(
                case,
                evaluation_questions,
                evaluation_responses,
                investigations_ordered=state.investigations_ordered,
                student_interpretations=state.student_interpretations,
                session_history=session_history or [],
            )
        finally:
            if fast_mode and original_reasoning_review is not None:
                self.evaluator.enable_reasoning_review = original_reasoning_review

        result = SimulationResult(
            diagnosis=diagnosis,
            difficulty=difficulty,
            student_questions=state.student_questions,
            patient_responses=state.patient_responses,
            investigations_ordered=state.investigations_ordered,
            student_interpretations=state.student_interpretations,
            doctor_consults=state.doctor_consults,
            diagnosis_guess=state.diagnosis_guess,
            steps=state.steps,
            evaluation=evaluation,
        )
        if audit and not skip_audit:
            result = result.model_copy(
                update={
                    "audit": self.get_simulation_auditor().audit(
                        build_audit_context(case, state, result)
                    )
                }
            )
        return result

    def run_many(
        self,
        *,
        runs: int,
        diagnosis: str,
        difficulty: str = "medium",
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float | None = None,
        fast_mode: bool = False,
        audit: bool = False,
        skip_audit: bool = False,
        print_progress: bool = False,
    ) -> tuple[list[SimulationResult], SimulationSummary]:
        results: list[SimulationResult] = []
        for run_number in range(1, runs + 1):
            started_at = monotonic()
            result = self.run_once(
                diagnosis=diagnosis,
                difficulty=difficulty,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                fast_mode=fast_mode,
                audit=audit,
                skip_audit=skip_audit,
            )
            results.append(result)
            if print_progress:
                elapsed = monotonic() - started_at
                print(
                    f"run={run_number} diagnosis={diagnosis} "
                    f"elapsed_seconds={elapsed:.2f} score={result.evaluation.overall_score}",
                    flush=True,
                )
        return results, summarize_simulation_results(results)

    def execute_action(
        self,
        state: "SimulationState",
        action: StudentAction,
        step_number: int,
    ) -> None:
        if action.action == "ask_patient_question":
            question = str(action.patient_question or "").strip()
            answer = self.patient_service.answer_question(state.case, question)
            state.student_questions.append(question)
            state.patient_responses.append(answer)
            state.steps.append(
                SimulationStep(
                    step_number=step_number,
                    action=action.action,
                    request=question,
                    response=answer,
                )
            )
            return

        if action.action == "order_investigation":
            investigation = str(action.investigation or "").strip()
            results = self.investigation_service.get_results(state.case, investigation)
            student_facing_results = [
                student_facing_investigation_result(result, state.case)
                for result in results
            ]
            state.investigations_ordered.extend(results)
            state.raw_investigation_results.extend(student_facing_results)
            state.steps.append(
                SimulationStep(
                    step_number=step_number,
                    action=action.action,
                    request=investigation,
                    response=student_facing_results[0]
                    if len(student_facing_results) == 1
                    else student_facing_results,
                )
            )
            return

        if action.action == "submit_interpretation":
            interpretation = str(action.interpretation or "").strip()
            state.student_interpretations.append(interpretation)
            state.steps.append(
                SimulationStep(
                    step_number=step_number,
                    action=action.action,
                    request=interpretation,
                )
            )
            return

        if action.action == "submit_diagnosis":
            state.diagnosis_guess = str(action.diagnosis_guess or "").strip()
            state.steps.append(
                SimulationStep(
                    step_number=step_number,
                    action=action.action,
                    request=state.diagnosis_guess,
                )
            )
            return

        if action.action == "call_doctor":
            question = str(action.consult_question or "").strip()
            if state.fast_mode:
                state.steps.append(
                    SimulationStep(
                        step_number=step_number,
                        action=action.action,
                        request=question,
                        response={"status": "skipped", "reason": "fast mode disables doctor consult"},
                    )
                )
                return
            consult = self.get_doctor_consult_service().consult(
                state.case,
                state.student_questions,
                state.patient_responses,
                state.investigations_ordered,
                question,
            )
            state.doctor_consults.append(consult)
            state.steps.append(
                SimulationStep(
                    step_number=step_number,
                    action=action.action,
                    request=question,
                    response=consult.model_dump(),
                )
            )

    def get_simulation_auditor(self) -> SimulationAuditorAgent:
        if self.simulation_auditor is None:
            self.simulation_auditor = SimulationAuditorAgent()
        return self.simulation_auditor

    def get_doctor_consult_service(self) -> DoctorConsultService:
        if self.doctor_consult_service is None:
            self.doctor_consult_service = DoctorConsultService()
        return self.doctor_consult_service


class SimulationState:
    """Mutable state used while executing one simulated encounter."""

    def __init__(
        self,
        *,
        case: ClinicalCase,
        difficulty: str,
        fast_mode: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.case = case
        self.difficulty = difficulty
        self.fast_mode = fast_mode
        self.max_steps = max_steps
        self.student_questions: list[str] = []
        self.patient_responses: list[str] = []
        self.investigations_ordered: list[InvestigationResult] = []
        self.raw_investigation_results: list[dict[str, Any]] = []
        self.student_interpretations: list[str] = []
        self.doctor_consults: list[DoctorConsultResponse] = []
        self.diagnosis_guess: str | None = None
        self.steps: list[SimulationStep] = []


def build_student_visible_context(
    state: SimulationState,
    *,
    step_number: int | None = None,
) -> dict[str, Any]:
    steps_remaining = state.max_steps - step_number + 1 if step_number is not None else None
    return {
        "case_intro": case_preview(state.case),
        "difficulty": state.difficulty,
        "simulation_control": {
            "fast_mode": state.fast_mode,
            "max_steps": state.max_steps,
            "current_step": step_number,
            "steps_remaining": steps_remaining,
            "near_end": steps_remaining is not None and steps_remaining <= 2,
        },
        "patient_conversation": [
            {"question": question, "answer": answer}
            for question, answer in zip(
                state.student_questions,
                state.patient_responses,
                strict=True,
            )
        ],
        "raw_investigation_results": state.raw_investigation_results,
        "student_interpretations": state.student_interpretations,
        "doctor_consults": [consult.model_dump() for consult in state.doctor_consults],
        "diagnosis_guess": state.diagnosis_guess,
    }


def build_audit_context(
    case: ClinicalCase,
    state: SimulationState,
    result: SimulationResult,
) -> dict[str, Any]:
    return {
        "generated_case": case.model_dump(),
        "student_visible_case_intro": case_preview(case),
        "transcript": [
            {"question": question, "patient_response": answer}
            for question, answer in zip(
                state.student_questions,
                state.patient_responses,
                strict=True,
            )
        ],
        "student_actions": [step.model_dump() for step in state.steps],
        "investigations_ordered": [
            investigation.model_dump()
            for investigation in state.investigations_ordered
        ],
        "student_facing_investigation_outputs": state.raw_investigation_results,
        "student_interpretations": state.student_interpretations,
        "doctor_consults": [consult.model_dump() for consult in state.doctor_consults],
        "diagnosis_guess": state.diagnosis_guess,
        "evaluator_output": result.evaluation.model_dump(),
    }


def summarize_simulation_results(results: list[SimulationResult]) -> SimulationSummary:
    if not results:
        return SimulationSummary(
            runs=0,
            average_score=0.0,
            common_missed_symptoms=[],
            common_reasoning_weaknesses=[],
            diagnosis_accuracy_rate=0.0,
            audit_summary=None,
        )
    scores = [result.evaluation.overall_score for result in results]
    missed_symptoms = Counter(
        symptom
        for result in results
        for symptom in result.evaluation.key_symptoms_missed
    )
    weaknesses = Counter(
        weakness
        for result in results
        for weakness in result.evaluation.reasoning_feedback.missed_reasoning_steps
    )
    correct_diagnoses = sum(
        1
        for result in results
        if result.evaluation.correct_diagnosis_guess.is_correct is True
    )
    return SimulationSummary(
        runs=len(results),
        average_score=round(sum(scores) / len(scores), 2),
        common_missed_symptoms=[item for item, _ in missed_symptoms.most_common(5)],
        common_reasoning_weaknesses=[item for item, _ in weaknesses.most_common(5)],
        diagnosis_accuracy_rate=round(correct_diagnoses / len(results), 2),
        audit_summary=summarize_audits(results),
    )


def summarize_audits(results: list[SimulationResult]) -> dict[str, object] | None:
    audits = [result.audit for result in results if result.audit is not None]
    if not audits:
        return None
    categories = Counter(
        finding.category
        for audit in audits
        for finding in audit.findings
    )
    severities = Counter(
        finding.severity
        for audit in audits
        for finding in audit.findings
    )
    return {
        "audited_runs": len(audits),
        "should_fix_runs": sum(1 for audit in audits if audit.should_fix),
        "overall_quality_counts": dict(Counter(audit.overall_quality for audit in audits)),
        "finding_category_counts": dict(categories),
        "finding_severity_counts": dict(severities),
    }


def check_timeout(started_at: float, timeout_seconds: float | None) -> None:
    if timeout_seconds is None:
        return
    if monotonic() - started_at >= timeout_seconds:
        raise TimeoutError(f"simulation run exceeded {timeout_seconds} seconds")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run AI-student clinical simulations.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--diagnosis", default="myocardial infarction")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    max_steps = min(args.max_steps, FAST_MAX_STEPS) if args.fast else args.max_steps
    runner = SimulationRunner()
    results, summary = runner.run_many(
        runs=args.runs,
        diagnosis=args.diagnosis,
        difficulty=args.difficulty,
        max_steps=max_steps,
        timeout_seconds=args.timeout_seconds,
        fast_mode=args.fast,
        audit=args.audit,
        skip_audit=args.skip_audit,
        print_progress=True,
    )
    payload = {
        "summary": summary.model_dump(),
        "results": [result.model_dump() for result in results],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
