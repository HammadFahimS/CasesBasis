from __future__ import annotations

from agents.generator import GeneratorAgent, MedicalFacts
from backend.clinical_evaluator import ClinicalEvaluator
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.performance_tracker import StudentPerformanceTracker
from backend.service import ClinicalCaseSimulator
from schemas.clinical_case import ClinicalCase
from schemas.evaluation import ClinicalEvaluation, ReasoningFeedback
from tests.test_performance_tracker import make_evaluation


class DemoGenerator:
    def generate(
        self,
        diagnosis: str,
        difficulty: str = "medium",
        session_history: list[object] | None = None,
    ) -> ClinicalCase:
        assert diagnosis == "myocardial infarction"
        assert difficulty == "medium"
        assert session_history == []
        return ClinicalCase.model_validate(
            {
                "diagnosis": "myocardial infarction",
                "patient": {
                    "age": 58,
                    "gender": "male",
                    "history": ["hypertension", "current smoker", "high cholesterol"],
                },
                "symptoms": [
                    "chest pressure",
                    "shortness of breath",
                    "nausea",
                    "sweating",
                    "left arm discomfort",
                ],
                "vitals": {
                    "heart_rate": 104,
                    "blood_pressure": "148/92",
                    "respiratory_rate": 22,
                    "oxygen_saturation": 95,
                },
                "progression": [
                    {
                        "time_min": 0,
                        "description": "Pressure starts while walking upstairs and does not fully improve with rest.",
                    },
                    {
                        "time_min": 20,
                        "description": "Pressure becomes more intense with sweating, nausea, and mild breathlessness.",
                    },
                    {
                        "time_min": 45,
                        "description": "Discomfort persists and radiates toward the left arm.",
                    },
                ],
                "risk_factors": ["smoking", "hypertension", "high cholesterol"],
                "exam_findings": ["diaphoretic", "anxious", "no focal neurologic deficit"],
                "outcome": {
                    "if_treated": "Rapid emergency treatment improves the chance of preserving heart muscle.",
                    "if_untreated": "Ongoing ischemia can lead to arrhythmia, heart failure, or death.",
                },
            }
        )


class DemoValidator:
    def recover(self, diagnosis: str, failure_reason: str | None = None, **_: object) -> ClinicalCase:
        raise AssertionError("validator recovery should not be used in demo")

    def correct(self, case: ClinicalCase, failure_reason: str | None = None) -> ClinicalCase:
        raise AssertionError("validator correction should not be used in demo")


class DemoPatientAgent:
    def answer(self, case: ClinicalCase, question: str) -> str:
        normalized = question.lower()
        if "describe" in normalized or "chest" in normalized:
            return "It feels like a heavy pressure in the middle of my chest."
        if "short of breath" in normalized or "breath" in normalized:
            return "A little, yes. I feel like I can't quite catch a full breath."
        if "go anywhere" in normalized or "radiate" in normalized:
            return "It kind of goes into my left arm, I think."
        if "nausea" in normalized or "sweating" in normalized:
            return "Yes, I feel nauseated and I'm sweating more than usual."
        if "myocardial infarction" in normalized:
            return "I'm not sure."
        return "I'm not sure."


class SessionCapturingGenerator(GeneratorAgent):
    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.received_session_history: list[object] | None = None

    def extract_medical_facts(self, diagnosis: str) -> MedicalFacts:
        return MedicalFacts(
            core_symptoms=["productive cough", "fever", "shortness of breath"],
            possible_atypical_symptoms=["fatigue", "mild confusion"],
            key_risk_factors=["older age", "smoking"],
        )

    def parse_response(self, **kwargs: object) -> ClinicalCase:
        self.last_prompt = str(kwargs["user_prompt"])
        return ClinicalCase.model_validate(
            {
                "diagnosis": "community-acquired pneumonia",
                "patient": {
                    "age": 67,
                    "gender": "female",
                    "history": ["former smoker", "type 2 diabetes"],
                },
                "symptoms": [
                    "cough",
                    "fever",
                    "shortness of breath",
                    "fatigue",
                ],
                "vitals": {
                    "heart_rate": 102,
                    "blood_pressure": "118/72",
                    "respiratory_rate": 24,
                    "oxygen_saturation": 93,
                },
                "progression": [
                    {
                        "time_min": 0,
                        "description": "Cough and fatigue have been present for two days.",
                    },
                    {
                        "time_min": 60,
                        "description": "Breathing feels harder and fever becomes more noticeable.",
                    },
                ],
                "risk_factors": ["former smoker", "diabetes"],
                "exam_findings": ["crackles at right lung base", "mild increased work of breathing"],
                "outcome": {
                    "if_treated": "Symptoms improve with timely antibiotics and supportive care.",
                    "if_untreated": "Respiratory status can worsen and require hospital care.",
                },
            }
        )

    def generate(
        self,
        diagnosis: str,
        difficulty: str = "medium",
        session_history: list[object] | None = None,
    ) -> ClinicalCase:
        self.received_session_history = session_history
        return super().generate(
            diagnosis,
            difficulty=difficulty,
            session_history=session_history,
        )


class OfflineLLMGenerator(GeneratorAgent):
    def __init__(self) -> None:
        self.calls: list[type[object]] = []

    def parse_response(self, **kwargs: object) -> object:
        response_model = kwargs["response_model"]
        self.calls.append(response_model)
        if response_model is MedicalFacts:
            return MedicalFacts(
                core_symptoms=["chest pressure", "shortness of breath", "sweating"],
                possible_atypical_symptoms=["nausea", "fatigue"],
                key_risk_factors=["smoking", "hypertension", "high cholesterol"],
            )
        if response_model is ClinicalCase:
            return ClinicalCase.model_validate(
                {
                    "diagnosis": "myocardial infarction",
                    "patient": {
                        "age": "58",
                        "gender": " male ",
                        "history": [" hypertension ", "current smoker", "high cholesterol"],
                    },
                    "symptoms": [
                        " chest pressure ",
                        "shortness of breath",
                        "nausea",
                        "sweating",
                        "left arm discomfort",
                    ],
                    "vitals": {
                        "heart_rate": "104 bpm",
                        "blood_pressure": "148 over 92",
                        "respiratory_rate": "22",
                        "oxygen_saturation": "95%",
                    },
                    "progression": [
                        {
                            "time_min": "0 min",
                            "description": "Pressure starts while walking upstairs.",
                        },
                        {
                            "time_min": "20",
                            "description": "Pressure intensifies with sweating and nausea.",
                        },
                        {
                            "time_min": "45",
                            "description": "Discomfort persists and radiates toward the left arm.",
                        },
                    ],
                    "risk_factors": ["smoking", "hypertension", "high cholesterol"],
                    "exam_findings": ["diaphoretic", "anxious"],
                    "outcome": {
                        "if_treated": "Rapid emergency treatment improves outcomes.",
                        "if_untreated": "Ongoing ischemia can lead to instability.",
                    },
                }
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class OfflineValidator:
    def recover(self, *_: object, **__: object) -> ClinicalCase:
        raise AssertionError("offline generator should not require recovery")

    def correct(self, *_: object, **__: object) -> ClinicalCase:
        raise AssertionError("offline generated case should pass medical validation")


class OfflinePatientAgent:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, case: ClinicalCase, question: str) -> str:
        self.questions.append(question)
        normalized = question.lower()
        if "feeling" in normalized or "chest" in normalized:
            return "It feels like a heavy pressure in the center of my chest."
        if "go anywhere" in normalized or "arm" in normalized:
            return "It maybe goes into my left arm."
        if "started" in normalized or "changed" in normalized:
            return "It started around half an hour ago and has been getting worse."
        if "breath" in normalized or "nauseated" in normalized or "sweating" in normalized:
            return "I feel a little short of breath, nauseated, and sweaty."
        if "history" in normalized or "smoking" in normalized:
            return "I have high blood pressure and cholesterol, and I smoke."
        if case.diagnosis.lower() in normalized:
            return "I'm not sure."
        return "I'm not sure."


class OfflineReasoningReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review(
        self,
        *,
        case: ClinicalCase,
        evaluation: ClinicalEvaluation,
        student_questions: list[str],
        patient_responses: list[str],
    ) -> ReasoningFeedback:
        self.calls += 1
        return evaluation.reasoning_feedback


class FailingGenericInvestigationGenerator:
    def generate(self, *_: object, **__: object) -> object:
        raise AssertionError("generic investigation generation should not be needed")


def test_end_to_end_demo_for_myocardial_infarction() -> None:
    simulator = ClinicalCaseSimulator(
        generator=DemoGenerator(),
        validator=DemoValidator(),
    )
    patient_service = PatientResponseService(agent=DemoPatientAgent())
    evaluator = ClinicalEvaluator()
    student_questions = [
        "Can you describe the chest discomfort?",
        "Are you short of breath?",
        "Does the pain go anywhere?",
        "Do you have nausea or sweating?",
        "Could this be myocardial infarction?",
    ]

    generated_case = simulator.generate_case(
        "myocardial infarction",
        difficulty="medium",
        session_history=[],
    )
    patient_responses = [
        patient_service.answer_question(generated_case, question)
        for question in student_questions
    ]
    evaluation = evaluator.evaluate(
        generated_case,
        student_questions,
        patient_responses,
        session_history=[],
    )

    assert generated_case.diagnosis == "myocardial infarction"
    assert evaluation.correct_diagnosis_guess.is_correct is True
    assert evaluation.overall_score >= 70
    assert evaluation.reasoning_feedback.missed_reasoning_steps
    assert evaluation.overall_score < 100


def test_end_to_end_negative_demo_for_poor_student_behavior() -> None:
    simulator = ClinicalCaseSimulator(
        generator=DemoGenerator(),
        validator=DemoValidator(),
    )
    patient_service = PatientResponseService(agent=DemoPatientAgent())
    evaluator = ClinicalEvaluator()
    student_questions = [
        "What is your favorite movie?",
        "Do you like coffee?",
        "What kind of music do you listen to?",
        "Could this be food poisoning?",
    ]

    generated_case = simulator.generate_case(
        "myocardial infarction",
        difficulty="medium",
        session_history=[],
    )
    patient_responses = [
        patient_service.answer_question(generated_case, question)
        for question in student_questions
    ]
    evaluation = evaluator.evaluate(
        generated_case,
        student_questions,
        patient_responses,
        session_history=[],
    )

    assert evaluation.overall_score < 40
    assert evaluation.key_symptoms_missed
    assert len(evaluation.key_symptoms_missed) == len(generated_case.symptoms)
    assert "scattered" in evaluation.reasoning_feedback.diagnostic_path.lower()
    assert evaluation.correct_diagnosis_guess.provided is True
    assert evaluation.correct_diagnosis_guess.is_correct is False
    assert evaluation.learning_recommendations.primary_skill_gap in {
        "foundational data gathering",
        "key symptom elicitation",
        "question relevance",
    }
    assert evaluation.learning_recommendations.focus_next
    assert evaluation.learning_recommendations.action_items
    evaluation_without_guess = evaluator.evaluate(
        generated_case,
        student_questions[:-1],
        patient_responses[:-1],
        session_history=[],
    )
    assert generated_case.diagnosis not in evaluation_without_guess.model_dump_json().lower()


def test_end_to_end_session_adaptation_demo() -> None:
    session_history = [
        make_evaluation(
            42,
            missed_symptoms=["cough"],
            missed_steps=["Clarify timing and progression of symptoms."],
        ),
        make_evaluation(
            38,
            missed_symptoms=["fever"],
            missed_steps=["Ask about relevant history or risk factors."],
        ),
        make_evaluation(
            47,
            missed_symptoms=["cough"],
            missed_steps=["Clarify timing and progression of symptoms."],
        ),
    ]
    generator = SessionCapturingGenerator()
    simulator = ClinicalCaseSimulator(
        generator=generator,
        validator=DemoValidator(),
    )
    evaluator = ClinicalEvaluator()
    tracker = StudentPerformanceTracker()

    generated_case = simulator.generate_case(
        "community-acquired pneumonia",
        difficulty="medium",
        session_history=session_history,
    )
    evaluation = evaluator.evaluate(
        generated_case,
        ["Do you have cough?", "Do you have fever?", "Are you short of breath?"],
        ["Yes.", "Yes.", "A little."],
        session_history=session_history,
    )
    performance = tracker.summarize(
        student_id="demo-student",
        previous_results=session_history,
    )

    assert generator.received_session_history is session_history
    assert generator.last_prompt is not None
    assert "SESSION CONTEXT" in generator.last_prompt
    assert "Clarify timing and progression of symptoms." in generator.last_prompt
    assert "Ask about relevant history or risk factors." in generator.last_prompt
    assert "Avoid repeating an identical scenario" in generator.last_prompt
    assert evaluation.learning_recommendations.primary_skill_gap == "foundational data gathering"
    assert "consistent opening sequence" in evaluation.learning_recommendations.focus_next
    assert performance.recommended_difficulty == "easy"


def test_offline_end_to_end_demo_without_live_llm_calls() -> None:
    generator = OfflineLLMGenerator()
    patient_agent = OfflinePatientAgent()
    reasoning_reviewer = OfflineReasoningReviewer()
    simulator = ClinicalCaseSimulator(
        generator=generator,
        validator=OfflineValidator(),
    )
    patient_service = PatientResponseService(agent=patient_agent)
    investigation_service = InvestigationService(
        generic_generator=FailingGenericInvestigationGenerator()
    )
    evaluator = ClinicalEvaluator(
        enable_reasoning_review=True,
        reasoning_reviewer=reasoning_reviewer,
    )
    student_questions = [
        "Can you describe what you are feeling in your chest?",
        "Does the discomfort go anywhere, like your arm?",
        "When did it start and has it changed?",
        "Are you short of breath, nauseated, or sweating?",
        "Do you have a history of blood pressure, cholesterol issues, or smoking?",
        "I think this could be myocardial infarction.",
    ]
    student_interpretations = [
        "The ECG is abnormal with ST elevation and reciprocal changes.",
        "The troponin is elevated above the reference range.",
    ]

    generated_case = simulator.generate_case(
        "myocardial infarction",
        difficulty="medium",
        session_history=[],
    )
    patient_responses = [
        patient_service.answer_question(generated_case, question)
        for question in student_questions
    ]
    investigations = [
        investigation_service.get_result(generated_case, "ECG"),
        investigation_service.get_result(generated_case, "troponin"),
    ]
    evaluation = evaluator.evaluate(
        generated_case,
        student_questions,
        patient_responses,
        investigations_ordered=investigations,
        student_interpretations=student_interpretations,
        session_history=[],
    )

    assert generator.calls == [MedicalFacts, ClinicalCase]
    assert isinstance(generated_case, ClinicalCase)
    assert generated_case.vitals.blood_pressure == "148/92"
    assert generated_case.vitals.heart_rate == 104
    assert len(patient_responses) == len(student_questions)
    assert all(response for response in patient_responses)
    assert patient_agent.questions == student_questions
    assert len(investigations) == 2
    assert investigations[0].investigation == "ECG"
    assert investigations[0].visual_payload is not None
    assert investigations[1].investigation == "Troponin"
    assert investigations[1].structured_result["flag"] == "high"
    assert isinstance(evaluation.overall_score, int)
    assert 0 <= evaluation.overall_score <= 100
    assert evaluation.learning_recommendations.primary_skill_gap
    assert evaluation.learning_recommendations.focus_next
    assert evaluation.learning_recommendations.action_items
    assert evaluation.investigation_interpretation_feedback.correctly_interpreted_findings
    assert evaluation.correct_diagnosis_guess.is_correct is True
