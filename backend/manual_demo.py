from __future__ import annotations

import json
import sys
from typing import Any

from dotenv import load_dotenv

from backend.clinical_evaluator import ClinicalEvaluator
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.service import ClinicalCaseSimulator
from schemas.clinical_case import ClinicalCase

DIAGNOSIS = "myocardial infarction"
DIFFICULTY = "medium"

STUDENT_QUESTIONS = [
    "Can you describe what you are feeling right now?",
    "Where is the discomfort, and does it go anywhere else?",
    "When did this start, and has it changed over time?",
    "Are you short of breath, nauseated, or sweating?",
    "Do you have any history of high blood pressure, cholesterol problems, or smoking?",
    "I think this could be myocardial infarction.",
]

REQUESTED_INVESTIGATIONS = [
    "ECG",
    "troponin",
]

STUDENT_INTERPRETATIONS = [
    "The ECG is abnormal with ST elevation and reciprocal changes, so this looks like an acute ischemic pattern.",
    "The troponin is elevated above the reference range, which supports cardiac muscle injury.",
]


def main() -> int:
    load_dotenv()
    try:
        simulator = ClinicalCaseSimulator()
        patient_service = PatientResponseService()
        investigation_service = InvestigationService()
        evaluator = ClinicalEvaluator()

        generated_case = simulator.generate_case(
            DIAGNOSIS,
            difficulty=DIFFICULTY,
            session_history=[],
        )
        patient_responses = [
            patient_service.answer_question(generated_case, question)
            for question in STUDENT_QUESTIONS
        ]
        investigation_results = [
            investigation_service.get_result(generated_case, request)
            for request in REQUESTED_INVESTIGATIONS
        ]
        evaluation = evaluator.evaluate(
            generated_case,
            STUDENT_QUESTIONS,
            patient_responses,
            session_history=[],
            investigations_ordered=investigation_results,
            student_interpretations=STUDENT_INTERPRETATIONS,
        )
    except Exception as exc:
        sys.stderr.write(f"Manual demo failed: {exc}\n")
        return 1

    print_section("Scenario", scenario_payload())
    print_section("Generated Case", generated_case)
    print_section("Student Questions And Patient Responses", interaction_payload(patient_responses))
    print_section("Investigations Ordered", investigation_results)
    print_section("Student Interpretations", STUDENT_INTERPRETATIONS)
    print_section("Evaluation", evaluation)
    print_section(
        "Investigation Interpretation Feedback",
        evaluation.investigation_interpretation_feedback,
    )
    print_section("Learning Recommendations", evaluation.learning_recommendations)
    return 0


def scenario_payload() -> dict[str, str]:
    return {
        "diagnosis": DIAGNOSIS,
        "difficulty": DIFFICULTY,
    }


def interaction_payload(patient_responses: list[str]) -> list[dict[str, str]]:
    return [
        {
            "student_question": question,
            "patient_response": response,
        }
        for question, response in zip(STUDENT_QUESTIONS, patient_responses, strict=True)
    ]


def print_section(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, ClinicalCase):
        print(payload.model_dump_json(indent=2))
        return
    if hasattr(payload, "model_dump_json"):
        print(payload.model_dump_json(indent=2))
        return
    if isinstance(payload, list) and payload and hasattr(payload[0], "model_dump"):
        print(json.dumps([item.model_dump() for item in payload], indent=2))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
