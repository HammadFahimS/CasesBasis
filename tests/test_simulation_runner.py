from __future__ import annotations

import json
from typing import Any

import pytest

from backend.investigations import InvestigationService
from backend.simulation_runner import (
    DEFAULT_MAX_STEPS,
    FAST_MAX_STEPS,
    SimulationRunner,
    build_student_visible_context,
    summarize_simulation_results,
)
from agents.simulation_auditor import SimulationAuditorAgent
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.evaluation import (
    ClinicalEvaluation,
    DiagnosisGuessEvaluation,
    InvestigationInterpretationFeedback,
    LearningRecommendations,
    QuestionRelevanceEvaluation,
    ReasoningFeedback,
)
from schemas.simulation import SimulationResult, StudentAction
from schemas.simulation_audit import SimulationAudit
from tests.test_investigations import mi_case


class FakeSimulator:
    def __init__(self, case: ClinicalCase | None = None) -> None:
        self.case = case or mi_case()
        self.calls: list[tuple[str, str]] = []

    def generate_case(
        self,
        diagnosis: str,
        *,
        difficulty: str,
        session_history: list[object],
    ) -> ClinicalCase:
        self.calls.append((diagnosis, difficulty))
        return self.case


class FakePatientService:
    def answer_question(self, case: ClinicalCase, question: str) -> str:
        if "nausea" in question.lower():
            return "Yes, I feel nauseated and sweaty."
        if "chest" in question.lower():
            return "It feels like heavy chest pressure."
        return "I'm not sure."


class FakeDoctorConsultService:
    def __init__(self) -> None:
        self.calls = 0

    def consult(
        self,
        case: ClinicalCase,
        student_questions: list[str],
        patient_responses: list[str],
        investigations_ordered: list[object],
        consult_question: str,
    ) -> DoctorConsultResponse:
        self.calls += 1
        return DoctorConsultResponse(
            urgency_level="urgent",
            concise_advice="I only have a minute: get an ECG and troponin.",
            recommended_next_steps=["Order ECG", "Send troponin"],
            missing_information=["Symptom onset time"],
            safety_warnings=["Escalate if unstable"],
            should_escalate=True,
        )


class ScriptedStudentAgent:
    def __init__(self, actions: list[StudentAction]) -> None:
        self.actions = actions
        self.contexts: list[dict[str, Any]] = []

    def next_action(
        self,
        visible_context: dict[str, Any],
        *,
        difficulty: str,
        step_number: int,
    ) -> StudentAction:
        self.contexts.append(visible_context)
        index = min(step_number - 1, len(self.actions) - 1)
        return self.actions[index]


class StepBasedStudentAgent:
    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def next_action(
        self,
        visible_context: dict[str, Any],
        *,
        difficulty: str,
        step_number: int,
    ) -> StudentAction:
        self.contexts.append(visible_context)
        if step_number == 1:
            return StudentAction(action="ask_patient_question", patient_question="Can you describe the chest pain?")
        if step_number == 2:
            return StudentAction(action="submit_diagnosis", diagnosis_guess="heart attack")
        return StudentAction(action="finish")


class FakeSimulationAuditor:
    def __init__(self, audit: SimulationAudit | None = None) -> None:
        self.contexts: list[dict[str, Any]] = []
        self.audit_result = audit or warning_audit()

    def audit(self, audit_context: dict[str, Any]) -> SimulationAudit:
        self.contexts.append(audit_context)
        return self.audit_result


class StubSimulationAuditorAgent(SimulationAuditorAgent):
    def __init__(self) -> None:
        self.last_user_prompt: str | None = None

    def parse_response(self, **kwargs: object) -> SimulationAudit:
        self.last_user_prompt = str(kwargs["user_prompt"])
        return SimulationAudit.model_validate(
            {
                "overall_quality": "fail",
                "findings": [
                    {
                        "category": "confirmed_bug",
                        "severity": "high",
                        "issue": "Hidden diagnosis leaked.",
                        "evidence": "The transcript exposed the diagnosis.",
                        "suggested_fix": "Redact the diagnosis from student-visible context.",
                    }
                ],
                "should_fix": True,
            }
        )


def test_student_action_requires_payload_for_action() -> None:
    with pytest.raises(ValueError):
        StudentAction(action="ask_patient_question")

    assert StudentAction(
        action="order_investigation",
        investigation="ECG",
    ).investigation == "ECG"


def test_default_max_steps_is_eight() -> None:
    assert DEFAULT_MAX_STEPS == 8
    assert FAST_MAX_STEPS < DEFAULT_MAX_STEPS


def test_simulation_runner_does_not_pass_hidden_case_data_to_student() -> None:
    student = ScriptedStudentAgent([StudentAction(action="finish")])
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=student,
    )

    runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=1,
    )

    assert student.contexts
    serialized_context = json.dumps(student.contexts[0]).lower()
    assert "myocardial infarction" not in serialized_context
    assert "if_treated" not in serialized_context
    assert "if_untreated" not in serialized_context
    assert "outcome" not in serialized_context
    assert "nausea" not in serialized_context
    assert "left arm discomfort" not in serialized_context


def test_simulation_runner_executes_end_to_end_with_mocked_student() -> None:
    student = ScriptedStudentAgent(
        [
            StudentAction(
                action="ask_patient_question",
                patient_question="Can you describe the chest pain?",
            ),
            StudentAction(action="order_investigation", investigation="ECG"),
            StudentAction(
                action="submit_interpretation",
                interpretation="The ECG has ST elevation and looks abnormal.",
            ),
            StudentAction(action="call_doctor", consult_question="What should I do next?"),
            StudentAction(action="submit_diagnosis", diagnosis_guess="heart attack"),
        ]
    )
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=student,
    )

    result = runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=5,
    )

    assert result.diagnosis_guess == "heart attack"
    assert result.student_questions == ["Can you describe the chest pain?"]
    assert result.patient_responses == ["It feels like heavy chest pressure."]
    assert result.investigations_ordered
    assert result.student_interpretations == ["The ECG has ST elevation and looks abnormal."]
    assert result.doctor_consults
    assert result.evaluation.correct_diagnosis_guess.is_correct is True
    assert result.steps[-1].action == "submit_diagnosis"


def test_fast_mode_skips_doctor_consult_and_restores_reasoning_review() -> None:
    student = ScriptedStudentAgent(
        [
            StudentAction(action="call_doctor", consult_question="Can you advise?"),
            StudentAction(action="finish"),
        ]
    )
    doctor_service = FakeDoctorConsultService()
    from backend.clinical_evaluator import ClinicalEvaluator

    evaluator = ClinicalEvaluator(enable_reasoning_review=True, reasoning_reviewer=object())
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=doctor_service,
        evaluator=evaluator,
        student_agent=student,
    )

    result = runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
        fast_mode=True,
    )

    assert doctor_service.calls == 0
    assert result.doctor_consults == []
    assert result.steps[0].response == {
        "status": "skipped",
        "reason": "fast mode disables doctor consult",
    }
    assert evaluator.enable_reasoning_review is True


def test_simulation_timeout_guard_raises() -> None:
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=ScriptedStudentAgent([StudentAction(action="finish")]),
    )

    with pytest.raises(TimeoutError):
        runner.run_once(
            diagnosis="myocardial infarction",
            difficulty="medium",
            max_steps=1,
            timeout_seconds=0,
        )


def test_run_many_prints_progress_when_enabled(capsys: pytest.CaptureFixture[str]) -> None:
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=StepBasedStudentAgent(),
    )

    runner.run_many(
        runs=1,
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
        print_progress=True,
    )

    output = capsys.readouterr().out
    assert "run=1 diagnosis=myocardial infarction" in output
    assert "elapsed_seconds=" in output
    assert "score=" in output


def test_repeated_runs_produce_aggregate_summary() -> None:
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=StepBasedStudentAgent(),
    )

    results, summary = runner.run_many(
        runs=3,
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
    )

    assert len(results) == 3
    assert summary.runs == 3
    assert summary.average_score > 0
    assert "nausea" in summary.common_missed_symptoms
    assert summary.common_reasoning_weaknesses
    assert summary.diagnosis_accuracy_rate == 1.0


def test_audit_is_optional_and_disabled_by_default() -> None:
    auditor = FakeSimulationAuditor()
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=StepBasedStudentAgent(),
        simulation_auditor=auditor,
    )

    result = runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
    )

    assert result.audit is None
    assert auditor.contexts == []


def test_auditor_receives_full_internal_qa_context() -> None:
    auditor = FakeSimulationAuditor()
    student = StepBasedStudentAgent()
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=student,
        simulation_auditor=auditor,
    )

    result = runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
        audit=True,
    )

    assert result.audit is not None
    assert auditor.contexts
    auditor_context = auditor.contexts[0]
    assert auditor_context["generated_case"]["diagnosis"] == "myocardial infarction"
    assert "nausea" in auditor_context["generated_case"]["symptoms"]
    assert "diagnosis" not in auditor_context["student_visible_case_intro"]

    assert student.contexts
    student_context_text = json.dumps(student.contexts[0]).lower()
    assert "myocardial infarction" not in student_context_text


def test_confirmed_bug_audit_sets_should_fix_true_and_summary_counts_it() -> None:
    auditor = FakeSimulationAuditor(confirmed_bug_audit())
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=StepBasedStudentAgent(),
        simulation_auditor=auditor,
    )

    results, summary = runner.run_many(
        runs=1,
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
        audit=True,
    )

    assert results[0].audit is not None
    assert results[0].audit.should_fix is True
    assert summary.audit_summary is not None
    assert summary.audit_summary["should_fix_runs"] == 1
    assert summary.audit_summary["finding_category_counts"]["confirmed_bug"] == 1


def test_skip_audit_prevents_audit_even_when_requested() -> None:
    auditor = FakeSimulationAuditor(confirmed_bug_audit())
    runner = SimulationRunner(
        simulator=FakeSimulator(),
        patient_service=FakePatientService(),
        investigation_service=InvestigationService(),
        doctor_consult_service=FakeDoctorConsultService(),
        student_agent=StepBasedStudentAgent(),
        simulation_auditor=auditor,
    )

    result = runner.run_once(
        diagnosis="myocardial infarction",
        difficulty="medium",
        max_steps=2,
        audit=True,
        skip_audit=True,
    )

    assert result.audit is None
    assert auditor.contexts == []


def test_simulation_auditor_agent_parses_output_into_schema() -> None:
    auditor = StubSimulationAuditorAgent()

    audit = auditor.audit({"generated_case": {"diagnosis": "myocardial infarction"}})

    assert isinstance(audit, SimulationAudit)
    assert audit.overall_quality == "fail"
    assert audit.should_fix is True
    assert audit.findings[0].category == "confirmed_bug"
    assert auditor.last_user_prompt is not None
    assert "myocardial infarction" in auditor.last_user_prompt


def test_summarize_simulation_results_uses_evaluations() -> None:
    results = [
        simulation_result(score=50, correct=True, missed=["nausea"], weakness="Ask about risk."),
        simulation_result(score=70, correct=False, missed=["nausea"], weakness="Ask about timing."),
    ]

    summary = summarize_simulation_results(results)

    assert summary.runs == 2
    assert summary.average_score == 60.0
    assert summary.common_missed_symptoms[0] == "nausea"
    assert summary.common_reasoning_weaknesses == ["Ask about risk.", "Ask about timing."]
    assert summary.diagnosis_accuracy_rate == 0.5


def test_student_visible_context_uses_raw_investigation_results() -> None:
    from backend.simulation_runner import SimulationState
    from backend.web_app import student_facing_investigation_result

    case = mi_case()
    state = SimulationState(case=case, difficulty="medium")
    ecg = InvestigationService().get_result(case, "ECG")
    state.investigations_ordered.append(ecg)
    state.raw_investigation_results.append(student_facing_investigation_result(ecg, case))

    context = build_student_visible_context(state)
    serialized = json.dumps(context).lower()

    assert "ecg completed" in serialized
    assert "st elevation" not in serialized
    assert "myocardial infarction" not in serialized


def simulation_result(
    *,
    score: int,
    correct: bool,
    missed: list[str],
    weakness: str,
) -> SimulationResult:
    return SimulationResult(
        diagnosis="myocardial infarction",
        difficulty="medium",
        student_questions=[],
        patient_responses=[],
        investigations_ordered=[],
        student_interpretations=[],
        doctor_consults=[],
        diagnosis_guess="heart attack" if correct else "pneumonia",
        steps=[],
        evaluation=ClinicalEvaluation(
            correct_diagnosis_guess=DiagnosisGuessEvaluation(
                provided=True,
                guess="heart attack" if correct else "pneumonia",
                is_correct=correct,
            ),
            key_symptoms_identified=[],
            key_symptoms_missed=missed,
            relevance_of_questions=QuestionRelevanceEvaluation(
                relevant_count=1,
                irrelevant_count=0,
                relevant_questions=["question"],
                irrelevant_questions=[],
            ),
            overall_score=score,
            brief_feedback="feedback",
            reasoning_feedback=ReasoningFeedback(
                diagnostic_path="path",
                question_order="order",
                missed_reasoning_steps=[weakness],
            ),
            learning_recommendations=LearningRecommendations(
                primary_skill_gap="gap",
                focus_next="focus",
                action_items=["action"],
            ),
            investigation_interpretation_feedback=InvestigationInterpretationFeedback(),
        ),
    )


def warning_audit() -> SimulationAudit:
    return SimulationAudit.model_validate(
        {
            "overall_quality": "warning",
            "findings": [
                {
                    "category": "possible_model_quality_issue",
                    "severity": "medium",
                    "issue": "Student missed subtle clues.",
                    "evidence": "The transcript did not include onset timing.",
                    "suggested_fix": "No code fix; review model behavior.",
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
                    "evidence": "Student-visible context included diagnosis text.",
                    "suggested_fix": "Redact hidden diagnosis before student prompt.",
                }
            ],
            "should_fix": True,
        }
    )
