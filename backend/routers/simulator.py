"""Classic (manual) simulator workbench API: case generation, history Q&A,
investigations (+ ECG re-render and the doctor-facing synthetic-ECG generator),
interpretations, diagnosis, doctor consult, evaluation and physical exam.

Extracted from ``web_app`` (F3). State and service singletons come from
``backend.sim_core``; the engine/services remain the single source of clinical
truth. ``web_app`` mounts ``router`` and re-exports the handlers + request models
so existing call sites and tests keep working.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from backend.auth import get_current_resident_id
from backend.persistence import (
    create_session,
    log_event,
    save_evaluation,
    save_session,
)
from backend.sim_core import (
    SESSIONS,
    WebSession,
    _get_owned_session,
    case_preview,
    get_doctor_consult_service,
    get_evaluator,
    get_graph,
    get_investigation_service,
    get_or_generate_cached_case,
    get_patient_service,
    get_physical_exam_agent,
    student_facing_investigation_result,
)
from backend.web_deps import limiter
from schemas.physical_exam import PhysicalExamRequest

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateCaseRequest(BaseModel):
    diagnosis: str
    difficulty: str = "medium"

    @field_validator("diagnosis")
    @classmethod
    def validate_diagnosis(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("diagnosis is required")
        if len(v) > 200:
            raise ValueError("diagnosis must be under 200 characters")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")
        return v


class QuestionRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is required")
        if len(v) > 500:
            raise ValueError("question must be under 500 characters")
        return v


class InvestigationRequest(BaseModel):
    session_id: str
    investigation: str

    @field_validator("investigation")
    @classmethod
    def validate_investigation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("investigation is required")
        if len(v) > 200:
            raise ValueError("investigation must be under 200 characters")
        return v


class InterpretationRequest(BaseModel):
    session_id: str
    interpretation: str

    @field_validator("interpretation")
    @classmethod
    def validate_interpretation(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 2000:
            raise ValueError("interpretation must be under 2000 characters")
        return v


class DiagnosisRequest(BaseModel):
    session_id: str
    diagnosis: str

    @field_validator("diagnosis")
    @classmethod
    def validate_diagnosis(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("diagnosis is required")
        if len(v) > 200:
            raise ValueError("diagnosis must be under 200 characters")
        return v


class DoctorConsultRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is required")
        if len(v) > 500:
            raise ValueError("question must be under 500 characters")
        return v


class EvaluateRequest(BaseModel):
    session_id: str
    elapsed_seconds: int = 0


class ECGRerenderRequest(BaseModel):
    """Re-render the most recent ECG investigation at a different sample rate."""
    session_id: str
    sample_rate_hz: int = 500  # 100 or 500


class ECGAlgorithmRequest(BaseModel):
    """Inputs for the doctor-facing synthetic-ECG generator."""
    diagnosis: str
    heart_rate: int | None = None
    age: int | None = None
    gender: str | None = None
    blood_pressure: str | None = None


@router.post("/api/ecg-algorithm/render")
@limiter.limit("20/minute")
def render_synthetic_ecg(body: ECGAlgorithmRequest, request: Request) -> dict[str, Any]:
    """Force the legacy LLM+numpy synthetic ECG path, regardless of PTB-XL.

    Returns the SVG, the LLM-produced ECGParameters, and a markdown trace that
    explains how the SVG was constructed. Used by the algorithm-review page so
    clinicians can audit each step.
    """
    from agents.ecg import ECGParameterAgent
    from agents.prompt_loader import load
    from backend.ecg_algorithm import build_algorithm_markdown
    from backend.ecg_synthesizer import synthesize_ecg_svg

    diagnosis = (body.diagnosis or "").strip()
    if not diagnosis:
        raise HTTPException(status_code=400, detail="diagnosis is required")

    agent = ECGParameterAgent()
    params = agent.generate(diagnosis, heart_rate=body.heart_rate)
    svg = synthesize_ecg_svg(params)
    system_prompt = load("ecg", "system")
    markdown = build_algorithm_markdown(
        diagnosis=diagnosis,
        heart_rate=body.heart_rate,
        params=params,
        system_prompt=system_prompt,
        provider=agent.provider,
        model=agent.model,
        case_summary={
            "age": body.age,
            "gender": body.gender,
            "blood_pressure": body.blood_pressure,
        },
    )
    return {
        "svg": svg,
        "parameters": params.model_dump(),
        "markdown": markdown,
        "model": agent.model,
        "provider": agent.provider,
    }


@router.post("/api/cases")
@limiter.limit("10/minute")
def generate_case(body: GenerateCaseRequest, request: Request) -> dict[str, Any]:
    resident_id = get_current_resident_id(request)
    clinical_case = get_or_generate_cached_case(body.diagnosis, body.difficulty)
    session_id = str(uuid4())
    SESSIONS[session_id] = WebSession(case=clinical_case, resident_id=resident_id)
    create_session(session_id, body.diagnosis, body.difficulty, clinical_case, resident_id=resident_id)
    log_event(session_id, "session.created", {"diagnosis": body.diagnosis, "difficulty": body.difficulty}, resident_id=resident_id)
    graph = get_graph()
    if graph and graph.enabled:
        graph.init_session(session_id, clinical_case)
    return {
        "session_id": session_id,
        "case_preview": case_preview(clinical_case),
    }


@router.post("/api/questions")
@limiter.limit("60/minute")
def ask_question(body: QuestionRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    history = list(zip(session.student_questions, session.patient_responses))
    answer = get_patient_service().answer_question(
        session.case, body.question,
        conversation_history=history,
        revealed_symptoms=session.revealed_symptoms,
    )
    session.student_questions.append(body.question)
    session.patient_responses.append(answer)
    for symptom in session.case.symptoms:
        if symptom not in session.revealed_symptoms and symptom.lower() in answer.lower():
            session.revealed_symptoms.append(symptom)
    graph = get_graph()
    if graph and graph.enabled:
        graph.add_turn(body.session_id, body.question, answer)
    save_session(body.session_id, session)
    log_event(body.session_id, "question.asked", {
        "question": body.question,
        "turn_index": len(session.student_questions) - 1,
    })
    return {
        "question": body.question,
        "answer": answer,
    }


@router.post("/api/investigations")
@limiter.limit("30/minute")
def order_investigation(body: InvestigationRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    results = get_investigation_service().get_results(session.case, body.investigation)
    session.investigations.extend(results)
    graph = get_graph()
    if graph and graph.enabled:
        for result in results:
            graph.add_investigation(body.session_id, result)
    save_session(body.session_id, session)
    for result in results:
        log_event(body.session_id, "investigation.ordered", {
            "investigation": result.investigation,
            "status": result.status,
        })
    displays = [
        student_facing_investigation_result(result, session.case)
        for result in results
    ]
    if len(displays) == 1:
        return displays[0]
    return {
        "investigations": displays,
    }


@router.post("/api/investigations/ecg-rerender")
@limiter.limit("60/minute")
def rerender_ecg(body: ECGRerenderRequest, request: Request) -> dict[str, Any]:
    """Re-render the most recent ECG investigation at a chosen sample rate.

    Powers the 100 Hz / 500 Hz toggle in the ECG panel. The picked PTB-XL
    record is unchanged — only the source signal's sampling rate. The session's
    stored investigation result is updated so the new SVG persists across
    reloads.
    """
    session = _get_owned_session(body.session_id, request)
    if body.sample_rate_hz not in (100, 500):
        raise HTTPException(status_code=400, detail="sample_rate_hz must be 100 or 500")

    # Find the most recent ECG investigation with a PTB-XL source_record.
    ecg_idx = next(
        (i for i in range(len(session.investigations) - 1, -1, -1)
         if "ecg" in (session.investigations[i].investigation or "").lower()
         and session.investigations[i].visual_payload is not None
         and getattr(session.investigations[i].visual_payload, "source_record", "")),
        None,
    )
    if ecg_idx is None:
        raise HTTPException(status_code=404, detail="no ECG investigation to re-render")

    inv = session.investigations[ecg_idx]
    source_record = inv.visual_payload.source_record  # type: ignore[union-attr]

    from backend.ptbxl_service import get_ptbxl_service
    svg = get_ptbxl_service().rerender_record(source_record, sample_rate_hz=body.sample_rate_hz)
    if svg is None:
        raise HTTPException(status_code=500, detail="re-render failed")

    # Mutate the visual_payload in place and persist.
    inv.visual_payload.svg = svg  # type: ignore[union-attr]
    save_session(body.session_id, session)
    log_event(body.session_id, "ecg.rerendered", {
        "sample_rate_hz": body.sample_rate_hz,
        "source_record": source_record,
    })
    return {"svg": svg, "sample_rate_hz": body.sample_rate_hz, "source_record": source_record}


@router.post("/api/interpretations")
@limiter.limit("30/minute")
def submit_interpretation(body: InterpretationRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    session.interpretations.append(body.interpretation)
    save_session(body.session_id, session)
    log_event(body.session_id, "interpretation.submitted", {"interpretation": body.interpretation})
    return {
        "interpretations": session.interpretations,
    }


@router.post("/api/diagnosis")
@limiter.limit("10/minute")
def submit_diagnosis(body: DiagnosisRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    session.diagnosis_guess = body.diagnosis
    save_session(body.session_id, session)
    log_event(body.session_id, "diagnosis.submitted", {"guess": body.diagnosis})
    return {
        "diagnosis_guess": body.diagnosis,
    }


@router.post("/api/doctor-consult")
@limiter.limit("10/minute")
def call_doctor(body: DoctorConsultRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    graph = get_graph()
    graph_context = graph.get_context(body.session_id) if graph and graph.enabled else ""
    response = get_doctor_consult_service().consult(
        session.case,
        session.student_questions,
        session.patient_responses,
        session.investigations,
        body.question,
        session_context={"clinical_picture": graph_context} if graph_context else None,
    )
    session.doctor_consults.append(response)
    save_session(body.session_id, session)
    log_event(body.session_id, "consult.called", {"question": body.question})
    return response.model_dump()


@router.post("/api/evaluate")
@limiter.limit("5/minute")
def evaluate(body: EvaluateRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    questions = list(session.student_questions)
    responses = list(session.patient_responses)
    if session.diagnosis_guess:
        questions.append(f"Diagnosis guess: {session.diagnosis_guess}")
        responses.append("I'm not sure.")

    graph = get_graph()
    graph_entity_names = graph.get_entity_names(body.session_id) if graph and graph.enabled else []

    evaluation = get_evaluator().evaluate(
        session.case,
        questions,
        responses,
        investigations_ordered=session.investigations,
        student_interpretations=session.interpretations,
        session_history=[],
        graph_entity_names=graph_entity_names or None,
        exams_performed=[e.system for e in session.physical_exams],
    )
    save_evaluation(body.session_id, evaluation, elapsed_seconds=body.elapsed_seconds)
    log_event(body.session_id, "session.evaluated", {"overall_score": getattr(evaluation, "overall_score", None), "elapsed_seconds": body.elapsed_seconds})
    return evaluation.model_dump()


@router.post("/api/physical-exam")
@limiter.limit("20/minute")
def physical_exam(body: PhysicalExamRequest, request: Request) -> dict[str, Any]:
    session = _get_owned_session(body.session_id, request)
    already_examined = [e.system for e in session.physical_exams]
    new_systems = [s for s in body.systems if s not in already_examined]
    if not new_systems:
        return {"results": [e.model_dump() for e in session.physical_exams]}

    results = get_physical_exam_agent().examine(session.case, new_systems, already_examined)
    session.physical_exams.extend(results)

    graph = get_graph()
    if graph and graph.enabled:
        for finding in results:
            try:
                graph.add_examination(body.session_id, finding)
            except Exception:
                pass

    save_session(body.session_id, session)
    log_event(body.session_id, "exam.performed", {"systems": new_systems})
    return {"results": [e.model_dump() for e in results]}
