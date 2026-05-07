from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.clinical_evaluator import ClinicalEvaluator
from backend.doctor_consult import DoctorConsultService
from backend.investigations import InvestigationService
from backend.patient_response import PatientResponseService
from backend.service import ClinicalCaseSimulator
from knowledge.graph import ClinicalKnowledgeGraph
from schemas.clinical_case import ClinicalCase
from schemas.doctor_consult import DoctorConsultResponse
from schemas.investigation import InvestigationResult

load_dotenv()

app = FastAPI(title="Clinical Case Simulator")




@dataclass
class WebSession:
    case: ClinicalCase
    student_questions: list[str] = field(default_factory=list)
    patient_responses: list[str] = field(default_factory=list)
    investigations: list[InvestigationResult] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    doctor_consults: list[DoctorConsultResponse] = field(default_factory=list)
    diagnosis_guess: str | None = None


class GenerateCaseRequest(BaseModel):
    diagnosis: str
    difficulty: str = "medium"


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class InvestigationRequest(BaseModel):
    session_id: str
    investigation: str


class InterpretationRequest(BaseModel):
    session_id: str
    interpretation: str


class DiagnosisRequest(BaseModel):
    session_id: str
    diagnosis: str


class DoctorConsultRequest(BaseModel):
    session_id: str
    question: str


class EvaluateRequest(BaseModel):
    session_id: str


SESSIONS: dict[str, WebSession] = {}
CASE_CACHE_TTL_SECONDS = 60
CASE_CACHE: dict[tuple[str, str], tuple[float, ClinicalCase]] = {}
SIMULATOR: ClinicalCaseSimulator | None = None
PATIENT_SERVICE: PatientResponseService | None = None
INVESTIGATION_SERVICE: InvestigationService | None = None
DOCTOR_CONSULT_SERVICE: DoctorConsultService | None = None
EVALUATOR: ClinicalEvaluator | None = None
GRAPH: ClinicalKnowledgeGraph | None = None


@app.get("/api/graph/{session_id}")
def get_graph_data(session_id: str) -> dict[str, Any]:
    get_session(session_id)
    try:
        import falkordb as _falkordb
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        db = _falkordb.FalkorDB(host=host, port=port)
        graph = db.select_graph(f"session:{session_id}")

        node_res = graph.query("MATCH (n) RETURN id(n), n.name, labels(n)")
        nodes = []
        seen_ids: set[str] = set()
        for row in node_res.result_set or []:
            nid, name, lbls = str(row[0]), row[1], row[2] or []
            if not name or nid in seen_ids:
                continue
            seen_ids.add(nid)
            nodes.append({"id": nid, "name": name, "labels": [l for l in lbls if l]})

        edge_res = graph.query(
            "MATCH (n)-[r]->(m) RETURN id(n), id(m), type(r)"
        )
        edges = []
        for row in edge_res.result_set or []:
            src, tgt, rel = str(row[0]), str(row[1]), row[2] or ""
            edges.append({"source": src, "target": tgt, "fact": rel.replace("_", " ").lower()})

        return {"nodes": nodes, "edges": edges}
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.post("/api/cases")
def generate_case(request: GenerateCaseRequest) -> dict[str, Any]:
    clinical_case = get_or_generate_cached_case(request.diagnosis, request.difficulty)
    session_id = str(uuid4())
    SESSIONS[session_id] = WebSession(case=clinical_case)
    graph = get_graph()
    if graph and graph.enabled:
        graph.init_session(session_id, clinical_case)
    return {
        "session_id": session_id,
        "case_preview": case_preview(clinical_case),
    }


@app.post("/api/questions")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    history = list(zip(session.student_questions, session.patient_responses))
    answer = get_patient_service().answer_question(
        session.case, request.question, conversation_history=history
    )
    session.student_questions.append(request.question)
    session.patient_responses.append(answer)
    graph = get_graph()
    if graph and graph.enabled:
        graph.add_turn(request.session_id, request.question, answer)
    return {
        "question": request.question,
        "answer": answer,
    }


@app.post("/api/investigations")
def order_investigation(request: InvestigationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    results = get_investigation_service().get_results(session.case, request.investigation)
    session.investigations.extend(results)
    graph = get_graph()
    if graph and graph.enabled:
        for result in results:
            graph.add_investigation(request.session_id, result)
    displays = [
        student_facing_investigation_result(result, session.case)
        for result in results
    ]
    if len(displays) == 1:
        return displays[0]
    return {
        "investigations": displays,
    }


@app.post("/api/interpretations")
def submit_interpretation(request: InterpretationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    session.interpretations.append(request.interpretation)
    return {
        "interpretations": session.interpretations,
    }


@app.post("/api/diagnosis")
def submit_diagnosis(request: DiagnosisRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    session.diagnosis_guess = request.diagnosis
    return {
        "diagnosis_guess": request.diagnosis,
    }


@app.post("/api/doctor-consult")
def call_doctor(request: DoctorConsultRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    graph = get_graph()
    graph_context = graph.get_context(request.session_id) if graph and graph.enabled else ""
    response = get_doctor_consult_service().consult(
        session.case,
        session.student_questions,
        session.patient_responses,
        session.investigations,
        request.question,
        session_context={"clinical_picture": graph_context} if graph_context else None,
    )
    session.doctor_consults.append(response)
    return response.model_dump()


@app.post("/api/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    questions = list(session.student_questions)
    responses = list(session.patient_responses)
    if session.diagnosis_guess:
        questions.append(f"Diagnosis guess: {session.diagnosis_guess}")
        responses.append("I'm not sure.")

    graph = get_graph()
    graph_entity_names = graph.get_entity_names(request.session_id) if graph and graph.enabled else []

    evaluation = get_evaluator().evaluate(
        session.case,
        questions,
        responses,
        investigations_ordered=session.investigations,
        student_interpretations=session.interpretations,
        session_history=[],
        graph_entity_names=graph_entity_names or None,
    )
    return evaluation.model_dump()


def get_session(session_id: str) -> WebSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def get_simulator() -> ClinicalCaseSimulator:
    global SIMULATOR
    if SIMULATOR is None:
        SIMULATOR = ClinicalCaseSimulator()
    return SIMULATOR


def get_patient_service() -> PatientResponseService:
    global PATIENT_SERVICE
    if PATIENT_SERVICE is None:
        PATIENT_SERVICE = PatientResponseService()
    return PATIENT_SERVICE


def get_investigation_service() -> InvestigationService:
    global INVESTIGATION_SERVICE
    if INVESTIGATION_SERVICE is None:
        INVESTIGATION_SERVICE = InvestigationService()
    return INVESTIGATION_SERVICE


def get_doctor_consult_service() -> DoctorConsultService:
    global DOCTOR_CONSULT_SERVICE
    if DOCTOR_CONSULT_SERVICE is None:
        DOCTOR_CONSULT_SERVICE = DoctorConsultService()
    return DOCTOR_CONSULT_SERVICE


def get_evaluator() -> ClinicalEvaluator:
    global EVALUATOR
    if EVALUATOR is None:
        EVALUATOR = ClinicalEvaluator()
    return EVALUATOR


def get_graph() -> ClinicalKnowledgeGraph | None:
    global GRAPH
    if GRAPH is None:
        GRAPH = ClinicalKnowledgeGraph()
    return GRAPH


def get_or_generate_cached_case(diagnosis: str, difficulty: str) -> ClinicalCase:
    key = cache_key(diagnosis, difficulty)
    cached = CASE_CACHE.get(key)
    now = monotonic()
    if cached and now - cached[0] <= CASE_CACHE_TTL_SECONDS:
        return cached[1].model_copy(deep=True)

    clinical_case = get_simulator().generate_case(
        diagnosis,
        difficulty=difficulty,
        session_history=[],
    )
    CASE_CACHE.clear()
    CASE_CACHE[key] = (now, clinical_case.model_copy(deep=True))
    return clinical_case


def cache_key(diagnosis: str, difficulty: str) -> tuple[str, str]:
    return (" ".join(diagnosis.split()).strip().lower(), difficulty.strip().lower())


def case_preview(case: ClinicalCase) -> dict[str, Any]:
    return {
        "age": case.patient.age,
        "gender": case.patient.gender,
        "occupation": case.patient.occupation,
        "persona_description": case.patient.persona_description,
        "chief_complaint": case.symptoms[0] if case.symptoms else "",
        "vitals": case.vitals.model_dump(),
    }


def student_facing_investigation_result(
    result: InvestigationResult,
    case: ClinicalCase,
) -> dict[str, Any]:
    if result.status == "unavailable":
        return {
            "investigation": result.investigation,
            "status": result.status,
            "summary": result.summary,
            "report": [],
            "visual_payload": None,
        }
    if is_ecg_display(result):
        return {
            "investigation": result.investigation,
            "status": result.status,
            "summary": "ECG completed",
            "report": [],
            "visual_payload": result.visual_payload.model_dump() if result.visual_payload else None,
        }
    return {
        "investigation": result.investigation,
        "status": result.status,
        "summary": f"{result.investigation} completed",
        "report": raw_investigation_report(result, case),
        "visual_payload": None,
    }


def is_ecg_display(result: InvestigationResult) -> bool:
    return result.visual_payload is not None or result.investigation.strip().lower() in {"ecg", "ekg"}


def raw_investigation_report(result: InvestigationResult, case: ClinicalCase) -> list[dict[str, Any]]:
    structured = result.structured_result
    values = structured.get("values")
    if isinstance(values, list):
        return [
            clean_lab_value(value, case)
            for value in values
            if isinstance(value, dict)
        ]

    if all(key in structured for key in ("value", "unit", "reference_range")):
        return [
            clean_lab_value(
                {
                    "name": structured.get("test", result.investigation),
                    "value": structured.get("value"),
                    "unit": structured.get("unit"),
                    "reference_range": structured.get("reference_range"),
                },
                case,
            )
        ]

    findings = structured.get("findings")
    if isinstance(findings, list):
        return [
            {"finding": redact_hidden_terms(str(finding), case)}
            for finding in findings
            if str(finding).strip()
        ]
    return []


def clean_lab_value(value: dict[str, Any], case: ClinicalCase) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in ("name", "value", "unit", "reference_range"):
        if key in value:
            item = value[key]
            cleaned[key] = redact_hidden_terms(str(item), case) if isinstance(item, str) else item
    return cleaned


def redact_hidden_terms(value: str, case: ClinicalCase) -> str:
    redacted = value
    diagnosis = case.diagnosis.strip()
    if diagnosis:
        redacted = redacted.replace(diagnosis, "the condition")
        redacted = redacted.replace(diagnosis.lower(), "the condition")
    return redacted


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>CasesBasis — Clinical Simulator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    *{font-family:'Inter',sans-serif}
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
    .fade-in{animation:fadeUp .3s ease-out}
    .spinning{animation:spin .7s linear infinite}
    .dot{animation:bounce 1s ease-in-out infinite}
    .dot:nth-child(2){animation-delay:.15s}
    .dot:nth-child(3){animation-delay:.3s}
    .bubble-you{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-radius:18px 18px 4px 18px;margin-left:auto}
    .bubble-patient{background:#fff;border:1px solid #e2e8f0;border-radius:18px 18px 18px 4px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
    .chat-area{height:420px;overflow-y:auto;scroll-behavior:smooth}
    .chat-area::-webkit-scrollbar{width:4px}
    .chat-area::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:2px}
    input,textarea,select{transition:border-color .15s,box-shadow .15s}
    input:focus,textarea:focus,select:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
    .ecg-wrap{background:#fff9f9;border-radius:8px;border:1px solid #fecaca}
    .ecg-wrap svg{display:block;width:100%;height:auto}
    .score-ring{width:110px;height:110px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.2rem;font-weight:800;flex-shrink:0}
    .chip{cursor:pointer;transition:background .15s,color .15s}
    .urgency-routine{background:#d1fae5;color:#065f46}
    .urgency-urgent{background:#fef3c7;color:#92400e}
    .urgency-critical{background:#fee2e2;color:#991b1b}
    [hidden]{display:none!important}
    #graphSvg{background:transparent}
    .graph-tooltip{position:absolute;background:rgba(15,23,42,.88);color:#e2e8f0;font-size:11px;padding:6px 10px;border-radius:8px;pointer-events:none;max-width:260px;line-height:1.4;backdrop-filter:blur(4px);z-index:100}
    .node-circle{cursor:grab;transition:opacity .2s}
    .node-circle:active{cursor:grabbing}
    .edge-label{paint-order:stroke;stroke:rgba(255,255,255,0.85);stroke-width:3px;stroke-linejoin:round;transition:opacity .2s}
    @keyframes micPulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.5)}70%{box-shadow:0 0 0 7px rgba(239,68,68,0)}}
    .mic-recording{animation:micPulse 1.1s ease-out infinite;background:#fee2e2!important;border-color:#ef4444!important;color:#dc2626!important}
    .replay-btn{opacity:0;transition:opacity .15s}
    .bubble-patient:hover .replay-btn{opacity:1}
  </style>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800">

<!-- HEADER -->
<header class="sticky top-0 z-50 bg-gradient-to-r from-slate-900 via-slate-800 to-blue-900 shadow-2xl">
  <div class="max-w-7xl mx-auto px-6 py-3.5 flex items-center gap-3">
    <div class="w-9 h-9 bg-blue-500 rounded-xl flex items-center justify-center shadow-lg">
      <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
      </svg>
    </div>
    <div>
      <p class="text-white font-bold text-base leading-tight">CasesBasis</p>
      <p class="text-blue-300 text-xs">Clinical Case Simulator</p>
    </div>
    <div id="sessionBadge" hidden class="ml-auto flex items-center gap-2 bg-white/10 rounded-full px-3 py-1.5">
      <span class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
      <span class="text-white text-xs font-medium">Session active</span>
    </div>
  </div>
</header>

<main class="max-w-7xl mx-auto px-4 sm:px-6 py-7 space-y-5">

  <!-- GENERATE CASE -->
  <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
    <div class="bg-gradient-to-r from-blue-600 to-blue-700 px-5 py-3.5 flex items-center gap-2.5">
      <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <h2 class="text-white font-semibold text-sm">Generate Clinical Case</h2>
    </div>
    <div class="p-5">
      <div class="flex flex-wrap gap-3 items-end">
        <div class="flex-1 min-w-52">
          <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Diagnosis</label>
          <select id="diagnosis" class="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm bg-slate-50 cursor-pointer">
            <optgroup label="Cardiac — Ischaemia">
              <option value="inferior STEMI">Inferior STEMI (RCA)</option>
              <option value="anterior STEMI">Anterior STEMI (LAD)</option>
              <option value="lateral STEMI">Lateral STEMI</option>
              <option value="NSTEMI">NSTEMI</option>
            </optgroup>
            <optgroup label="Cardiac — Rhythm">
              <option value="atrial fibrillation" selected>Atrial Fibrillation</option>
              <option value="atrial flutter">Atrial Flutter</option>
              <option value="supraventricular tachycardia">SVT</option>
              <option value="ventricular tachycardia">Ventricular Tachycardia</option>
              <option value="ventricular fibrillation">Ventricular Fibrillation</option>
              <option value="complete heart block">Complete Heart Block</option>
              <option value="sinus bradycardia">Sinus Bradycardia</option>
            </optgroup>
            <optgroup label="Cardiac — Conduction">
              <option value="left bundle branch block">Left Bundle Branch Block (LBBB)</option>
              <option value="right bundle branch block">Right Bundle Branch Block (RBBB)</option>
              <option value="Wolff-Parkinson-White syndrome">WPW Syndrome</option>
              <option value="left ventricular hypertrophy">LVH</option>
            </optgroup>
            <optgroup label="Cardiac — Other">
              <option value="pericarditis">Pericarditis</option>
              <option value="myocarditis">Myocarditis</option>
              <option value="heart failure">Heart Failure</option>
              <option value="hypertensive emergency">Hypertensive Emergency</option>
            </optgroup>
            <optgroup label="Respiratory">
              <option value="pulmonary embolism">Pulmonary Embolism</option>
              <option value="pneumothorax">Pneumothorax</option>
              <option value="community-acquired pneumonia">Pneumonia</option>
              <option value="acute severe asthma">Acute Severe Asthma</option>
              <option value="COPD exacerbation">COPD Exacerbation</option>
            </optgroup>
            <optgroup label="Metabolic / Electrolytes">
              <option value="diabetic ketoacidosis">Diabetic Ketoacidosis (DKA)</option>
              <option value="hyperkalaemia">Hyperkalaemia</option>
              <option value="hypokalaemia">Hypokalaemia</option>
              <option value="hypoglycaemia">Hypoglycaemia</option>
            </optgroup>
            <optgroup label="Neurological">
              <option value="ischaemic stroke">Ischaemic Stroke</option>
              <option value="subarachnoid haemorrhage">Subarachnoid Haemorrhage</option>
              <option value="meningitis">Meningitis</option>
            </optgroup>
            <optgroup label="Surgical / GI">
              <option value="appendicitis">Appendicitis</option>
              <option value="bowel obstruction">Bowel Obstruction</option>
              <option value="acute pancreatitis">Acute Pancreatitis</option>
              <option value="upper GI bleed">Upper GI Bleed</option>
            </optgroup>
            <optgroup label="Sepsis / Shock">
              <option value="septic shock">Septic Shock</option>
              <option value="anaphylaxis">Anaphylaxis</option>
            </optgroup>
          </select>
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Difficulty</label>
          <select id="difficulty" class="border border-slate-200 rounded-xl px-4 py-2.5 text-sm bg-slate-50 cursor-pointer">
            <option value="easy">Easy</option>
            <option value="medium" selected>Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
        <button id="generateBtn" onclick="generateCase()"
          class="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 active:scale-95 text-white font-semibold px-5 py-2.5 rounded-xl text-sm transition-all shadow-md">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>
          Generate
        </button>
      </div>
      <!-- Patient vitals strip -->
      <div id="vitalsStrip" hidden class="mt-5 pt-5 border-t border-slate-100 fade-in">
        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Patient Overview</p>
        <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
          <div class="bg-slate-50 rounded-xl p-3 col-span-2 sm:col-span-2 lg:col-span-2">
            <p class="text-xs text-slate-400 mb-0.5">Chief Complaint</p>
            <p id="pvCC" class="text-sm font-semibold text-slate-800 leading-snug"></p>
          </div>
          <div class="bg-slate-50 rounded-xl p-3"><p class="text-xs text-slate-400 mb-0.5">Age</p><p id="pvAge" class="text-base font-bold text-slate-800"></p></div>
          <div class="bg-slate-50 rounded-xl p-3"><p class="text-xs text-slate-400 mb-0.5">Gender</p><p id="pvGender" class="text-base font-bold text-slate-800 capitalize"></p></div>
          <div class="bg-slate-50 rounded-xl p-3"><p class="text-xs text-slate-400 mb-0.5">HR</p><p id="pvHR" class="text-base font-bold text-slate-800"></p></div>
          <div class="bg-slate-50 rounded-xl p-3"><p class="text-xs text-slate-400 mb-0.5">BP</p><p id="pvBP" class="text-base font-bold text-slate-800"></p></div>
          <div class="bg-slate-50 rounded-xl p-3"><p class="text-xs text-slate-400 mb-0.5">SpO2</p><p id="pvSPO2" class="text-base font-bold text-slate-800"></p></div>
        </div>
        <div id="personaRow" hidden class="mt-2 flex flex-wrap gap-2 items-start">
          <div id="pvOccupationWrap" hidden class="bg-indigo-50 border border-indigo-100 rounded-xl px-3 py-2 flex-shrink-0">
            <p class="text-xs text-indigo-400 mb-0.5">Occupation</p>
            <p id="pvOccupation" class="text-sm font-semibold text-indigo-800 capitalize"></p>
          </div>
          <div id="pvPersonaWrap" hidden class="bg-indigo-50 border border-indigo-100 rounded-xl px-3 py-2 flex-1 min-w-0">
            <p class="text-xs text-indigo-400 mb-0.5">Persona</p>
            <p id="pvPersona" class="text-sm text-indigo-800 leading-snug"></p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- MAIN GRID -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">

    <!-- LEFT: chat + interpretations -->
    <div class="lg:col-span-2 space-y-5">

      <!-- Patient chat -->
      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
        <div class="bg-gradient-to-r from-slate-700 to-slate-800 px-5 py-3.5 flex items-center gap-2.5">
          <div class="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center flex-shrink-0">
            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
            </svg>
          </div>
          <div><p class="text-white font-semibold text-sm">Patient Interaction</p><p class="text-slate-400 text-xs">Ask clinical questions</p></div>
          <button id="ttsBtn" onclick="toggleTts()" title="Toggle patient voice"
            class="ml-auto w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors flex-shrink-0">
            <svg id="ttsIcon" class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072M12 6v12m-3.536-9.536a5 5 0 000 7.072M6.464 9.536a8 8 0 000 4.928"/>
            </svg>
          </button>
        </div>
        <div id="chatArea" class="chat-area p-4 space-y-3 bg-slate-50/60 flex-1">
          <div class="flex justify-center pt-6">
            <p class="text-slate-400 text-sm">Generate a case to begin your assessment.</p>
          </div>
        </div>
        <div class="border-t border-slate-100 p-4 bg-white">
          <div class="flex gap-2">
            <button id="micBtn" onclick="toggleMic()" title="Speak your question"
              class="border border-slate-200 rounded-xl px-3 py-2.5 bg-slate-50 hover:bg-slate-100 transition-all flex-shrink-0 flex items-center justify-center">
              <svg class="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
              </svg>
            </button>
            <input id="question" placeholder="Ask the patient a question…"
              class="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm bg-slate-50"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQuestion()}"/>
            <button id="askBtn" onclick="askQuestion()"
              class="bg-blue-600 hover:bg-blue-700 active:scale-95 text-white px-5 py-2.5 rounded-xl text-sm font-semibold transition-all flex items-center gap-1.5">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
              </svg>Ask
            </button>
          </div>
        </div>
      </div>

      <!-- Interpretations -->
      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="bg-gradient-to-r from-violet-600 to-violet-700 px-5 py-3.5 flex items-center gap-2.5">
          <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
          </svg>
          <h2 class="text-white font-semibold text-sm">Clinical Interpretations</h2>
        </div>
        <div class="p-5 space-y-3">
          <div class="flex gap-2">
            <textarea id="interpretation" rows="2" placeholder="Enter your interpretation of ordered tests…"
              class="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm resize-none bg-slate-50"></textarea>
            <button id="interpBtn" onclick="submitInterpretation()"
              class="self-end bg-violet-600 hover:bg-violet-700 active:scale-95 text-white px-4 py-2.5 rounded-xl text-sm font-semibold transition-all">
              Submit
            </button>
          </div>
          <ul id="interpList" class="space-y-1.5 text-sm"></ul>
        </div>
      </div>

    </div>

    <!-- RIGHT sidebar -->
    <div class="space-y-5">

      <!-- Investigations -->
      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="bg-gradient-to-r from-teal-600 to-teal-700 px-5 py-3.5 flex items-center gap-2.5">
          <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
          <h2 class="text-white font-semibold text-sm">Investigations</h2>
        </div>
        <div class="p-4 space-y-3">
          <div class="flex gap-2">
            <input id="investigation" value="ECG"
              class="flex-1 min-w-0 border border-slate-200 rounded-xl px-3 py-2 text-sm bg-slate-50"
              onkeydown="if(event.key==='Enter')orderInvestigation()"/>
            <button id="invBtn" onclick="orderInvestigation()"
              class="bg-teal-600 hover:bg-teal-700 active:scale-95 text-white px-4 py-2 rounded-xl text-sm font-semibold transition-all flex-shrink-0">
              Order
            </button>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <button class="chip text-xs bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-600 px-2.5 py-1 rounded-full" onclick="quickOrder('ECG')">ECG</button>
            <button class="chip text-xs bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-600 px-2.5 py-1 rounded-full" onclick="quickOrder('Troponin')">Troponin</button>
            <button class="chip text-xs bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-600 px-2.5 py-1 rounded-full" onclick="quickOrder('CBC')">CBC</button>
            <button class="chip text-xs bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-600 px-2.5 py-1 rounded-full" onclick="quickOrder('CXR')">CXR</button>
            <button class="chip text-xs bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-600 px-2.5 py-1 rounded-full" onclick="quickOrder('ABG')">ABG</button>
          </div>
          <div id="invResults" class="space-y-3"></div>
        </div>
      </div>

      <!-- Doctor Consult -->
      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="bg-gradient-to-r from-amber-500 to-amber-600 px-5 py-3.5 flex items-center gap-2.5">
          <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/>
          </svg>
          <h2 class="text-white font-semibold text-sm">Call Doctor</h2>
        </div>
        <div class="p-4 space-y-3">
          <textarea id="doctorQuestion" rows="2" placeholder="Ask the ER physician for guidance…"
            class="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm resize-none bg-slate-50"></textarea>
          <button id="doctorBtn" onclick="callDoctor()"
            class="w-full bg-amber-500 hover:bg-amber-600 active:scale-95 text-white py-2.5 rounded-xl text-sm font-semibold transition-all">
            Consult Doctor
          </button>
          <div id="doctorConsults" class="space-y-2"></div>
        </div>
      </div>

    </div>
  </div>

  <!-- ECG PANEL (full-width, revealed when ECG is ordered) -->
  <div id="ecgPanel" hidden class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden fade-in">
    <div class="bg-gradient-to-r from-red-700 to-rose-700 px-5 py-3.5 flex items-center gap-2.5">
      <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12h2l2-6 2 12 2-8 2 6 2-4 2 4h2"/>
      </svg>
      <div>
        <p class="text-white font-semibold text-sm">Electrocardiogram (ECG)</p>
        <p class="text-rose-200 text-xs">12-lead ECG — synthesised from case parameters</p>
      </div>
    </div>
    <div class="p-4">
      <div id="ecgContent" class="ecg-wrap"></div>
    </div>
  </div>

  <!-- DIAGNOSIS + EVALUATE ROW -->
  <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">

    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div class="bg-gradient-to-r from-rose-600 to-rose-700 px-5 py-3.5 flex items-center gap-2.5">
        <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <h2 class="text-white font-semibold text-sm">Submit Diagnosis</h2>
      </div>
      <div class="p-4 space-y-3">
        <div class="flex gap-2">
          <input id="diagnosisGuess" placeholder="Your diagnosis…"
            class="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm bg-slate-50"
            onkeydown="if(event.key==='Enter')submitDiagnosis()"/>
          <button id="diagnosisBtn" onclick="submitDiagnosis()"
            class="bg-rose-600 hover:bg-rose-700 active:scale-95 text-white px-4 py-2.5 rounded-xl text-sm font-semibold transition-all flex-shrink-0">
            Submit
          </button>
        </div>
        <div id="diagnosisResult" hidden class="bg-rose-50 border border-rose-200 rounded-xl p-3 text-sm text-rose-800 fade-in"></div>
      </div>
    </div>

    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div class="bg-gradient-to-r from-emerald-600 to-emerald-700 px-5 py-3.5 flex items-center gap-2.5">
        <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
        </svg>
        <h2 class="text-white font-semibold text-sm">Final Evaluation</h2>
      </div>
      <div class="p-4">
        <p class="text-slate-500 text-sm mb-3">Submit your diagnosis and interpretations, then evaluate your clinical reasoning.</p>
        <button id="evaluateBtn" onclick="evaluateCase()"
          class="w-full bg-emerald-600 hover:bg-emerald-700 active:scale-95 text-white py-2.5 rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          Evaluate Performance
        </button>
      </div>
    </div>

  </div>

  <!-- KNOWLEDGE GRAPH -->
  <div id="graphPanel" hidden class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden fade-in">
    <div class="bg-gradient-to-r from-violet-800 to-indigo-800 px-5 py-3.5 flex items-center justify-between">
      <div class="flex items-center gap-2.5">
        <svg class="w-5 h-5 text-white flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
        </svg>
        <div>
          <p class="text-white font-semibold text-sm">Clinical Knowledge Graph</p>
          <p class="text-indigo-300 text-xs">Entities extracted from conversation — updates as you interact</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <span id="graphStatus" class="text-xs text-indigo-300 italic">Waiting for interactions…</span>
        <span id="graphBadge" hidden class="text-xs bg-white/15 text-white px-2.5 py-1 rounded-full font-medium"></span>
        <!-- zoom controls -->
        <div id="graphControls" hidden class="flex items-center gap-1 ml-2">
          <button onclick="graphZoomIn()" title="Zoom in"
            class="w-7 h-7 rounded-lg bg-white/10 hover:bg-white/25 text-white text-base font-bold flex items-center justify-center transition-colors">+</button>
          <button onclick="graphZoomOut()" title="Zoom out"
            class="w-7 h-7 rounded-lg bg-white/10 hover:bg-white/25 text-white text-base font-bold flex items-center justify-center transition-colors">−</button>
          <button onclick="graphFit()" title="Fit all nodes"
            class="h-7 px-2.5 rounded-lg bg-white/10 hover:bg-white/25 text-white flex items-center gap-1.5 transition-colors text-xs font-medium">
            <svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-5h-4m4 0v4m0-4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
            </svg>
            Fit
          </button>
        </div>
      </div>
    </div>
    <div class="relative bg-slate-50 rounded-b-2xl overflow-hidden" style="height:400px">
      <div id="graphEmpty" class="absolute inset-0 flex flex-col items-center justify-center gap-3 text-slate-400 pointer-events-none">
        <svg class="w-10 h-10 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"/>
        </svg>
        <p class="text-sm">Ask the patient questions to build the knowledge graph</p>
      </div>
      <svg id="graphSvg" class="w-full h-full" style="cursor:grab"></svg>
      <!-- Legend -->
      <div class="absolute bottom-2 left-3 flex items-center gap-3 pointer-events-none select-none">
        <span class="flex items-center gap-1 text-xs text-slate-500"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981"></span>Symptom</span>
        <span class="flex items-center gap-1 text-xs text-slate-500"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444"></span>Medication</span>
        <span class="flex items-center gap-1 text-xs text-slate-500"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#06b6d4"></span>Finding</span>
        <span class="flex items-center gap-1 text-xs text-slate-500"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f97316"></span>RiskFactor</span>
        <span class="flex items-center gap-1 text-xs text-slate-500"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f59e0b"></span>Temporal</span>
      </div>
      <p class="absolute bottom-2 right-3 text-xs text-slate-400 pointer-events-none select-none">click node to highlight · scroll to zoom · drag to pan</p>
      <div id="graphTooltip" class="graph-tooltip" style="display:none"></div>
    </div>
  </div>

  <!-- EVALUATION RESULTS -->
  <div id="evalResults" hidden class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden fade-in">
    <div class="bg-gradient-to-r from-slate-800 to-slate-900 px-5 py-3.5">
      <h2 class="text-white font-semibold">Performance Report</h2>
    </div>
    <div class="p-6 space-y-6">
      <!-- Score row -->
      <div class="flex flex-wrap items-center gap-6">
        <div id="scoreCircle" class="score-ring">—</div>
        <div class="flex-1 min-w-40">
          <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">Overall Score</p>
          <p id="scoreFeedback" class="text-slate-700 text-sm leading-relaxed"></p>
        </div>
      </div>
      <!-- Metrics -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="bg-slate-50 rounded-xl p-4">
          <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Diagnosis</p>
          <p id="evalDx" class="text-sm font-semibold"></p>
        </div>
        <div class="bg-slate-50 rounded-xl p-4">
          <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Symptoms Elicited</p>
          <div id="evalSymptoms" class="text-sm"></div>
        </div>
        <div class="bg-slate-50 rounded-xl p-4">
          <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Question Quality</p>
          <p id="evalQuality" class="text-sm text-slate-700"></p>
        </div>
      </div>
      <!-- Reasoning + Learning -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div class="bg-slate-50 rounded-xl p-4">
          <p class="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Reasoning Feedback</p>
          <div id="evalReasoning" class="text-sm text-slate-700 space-y-2"></div>
        </div>
        <div class="bg-blue-50 rounded-xl p-4">
          <p class="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-3">Learning Recommendations</p>
          <div id="evalLearning" class="text-sm text-slate-700 space-y-2"></div>
        </div>
      </div>
    </div>
  </div>

</main>

<!-- TOAST -->
<div id="toast" hidden class="fixed bottom-5 right-5 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-2xl fade-in"></div>

<script>
  let sessionId = null;

  /* ── voice I/O ── */
  let _ttsEnabled = true;
  let _patientGender = 'female';
  let _recognition = null;
  let _isRecording = false;

  function setupSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { document.getElementById('micBtn').hidden = true; return; }
    _recognition = new SR();
    _recognition.continuous = false;
    _recognition.interimResults = true;
    _recognition.lang = 'en-US';
    _recognition.onresult = ev => {
      const t = Array.from(ev.results).map(r => r[0].transcript).join('');
      document.getElementById('question').value = t;
      if (ev.results[ev.results.length - 1].isFinal) { _stopRecording(); askQuestion(); }
    };
    _recognition.onerror = ev => {
      _stopRecording();
      const msgs = {
        'not-allowed':       'Mic access denied — enable it in browser/OS settings.',
        'service-not-allowed':'Speech recognition blocked on this browser or platform.',
        'network':           'Speech service unreachable — check your connection.',
        'no-speech':         null, // user just didn't speak, no toast needed
      };
      const msg = ev.error in msgs ? msgs[ev.error] : 'Mic error: ' + ev.error;
      if (msg) toast(msg);
    };
    _recognition.onend = _stopRecording;
  }

  function toggleMic() {
    if (!sessionId) { toast('Generate a case first.'); return; }
    if (!_recognition) { toast('Voice input not supported on this browser. Use the text field.'); return; }
    _isRecording ? _stopRecording() : _startRecording();
  }
  async function _startRecording() {
    // Explicitly request mic permission first so mobile browsers show the prompt
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop()); // release immediately — SpeechRecognition manages its own
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? 'Mic access denied — enable it in your browser/OS settings.'
        : 'Microphone unavailable: ' + err.message;
      toast(msg);
      return;
    }
    window.speechSynthesis.cancel();
    _isRecording = true;
    document.getElementById('micBtn').classList.add('mic-recording');
    document.getElementById('question').value = '';
    document.getElementById('question').placeholder = 'Listening…';
    try { _recognition.start(); } catch(_) {}
  }
  function _stopRecording() {
    _isRecording = false;
    document.getElementById('micBtn').classList.remove('mic-recording');
    document.getElementById('question').placeholder = 'Ask the patient a question…';
    try { _recognition.stop(); } catch(_) {}
  }

  function speakText(text) {
    if (!_ttsEnabled || !window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const malePrefs   = ['Daniel', 'Alex', 'Google UK English Male', 'Microsoft David', 'Aaron'];
    const femalePrefs = ['Samantha', 'Karen', 'Google UK English Female', 'Microsoft Zira', 'Moira'];
    const prefs = _patientGender === 'male' ? malePrefs : femalePrefs;
    const pick = voices.find(v => prefs.some(p => v.name.includes(p)))
              || voices.find(v => v.lang.startsWith('en'))
              || null;
    if (pick) utt.voice = pick;
    utt.rate  = 0.93;
    utt.pitch = _patientGender === 'male' ? 0.88 : 1.05;
    window.speechSynthesis.speak(utt);
  }

  function toggleTts() {
    _ttsEnabled = !_ttsEnabled;
    window.speechSynthesis.cancel();
    const icon = document.getElementById('ttsIcon');
    const btn  = document.getElementById('ttsBtn');
    if (_ttsEnabled) {
      btn.title = 'Mute patient voice';
      icon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072M12 6v12m-3.536-9.536a5 5 0 000 7.072M6.464 9.536a8 8 0 000 4.928"/>';
    } else {
      btn.title = 'Unmute patient voice';
      icon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2"/>';
    }
  }

  // Preload voices (Chrome loads them async)
  if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  /* ── helpers ── */
  async function postJson(url, body) {
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const txt = await r.text();
    let data; try{data=JSON.parse(txt)}catch{data={detail:txt}};
    if(!r.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
    return data;
  }
  function req() { if(!sessionId) throw new Error('Generate a case first.'); }
  function toast(msg, type='error') {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = `fixed bottom-5 right-5 z-50 px-5 py-3 rounded-xl text-sm font-medium shadow-2xl fade-in ${type==='ok'?'bg-emerald-600 text-white':'bg-red-600 text-white'}`;
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(()=>el.hidden=true, 3500);
  }
  function spin(id, on) {
    const b = document.getElementById(id); if(!b) return;
    if(on){b._orig=b.innerHTML; b.disabled=true; b.innerHTML='<svg class="w-4 h-4 spinning inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>';}
    else{b.innerHTML=b._orig||b.innerHTML; b.disabled=false;}
  }
  function quickOrder(name){document.getElementById('investigation').value=name;orderInvestigation();}

  /* ── generateCase ── */
  async function generateCase() {
    spin('generateBtn',true);
    try {
      const data = await postJson('/api/cases', {
        diagnosis: document.getElementById('diagnosis').value,
        difficulty: document.getElementById('difficulty').value
      });
      sessionId = data.session_id;
      _patientGender = (data.case_preview.gender||'female').toLowerCase();
      window.speechSynthesis && window.speechSynthesis.cancel();
      document.getElementById('sessionBadge').hidden = false;

      const v = data.case_preview.vitals||{};
      document.getElementById('pvCC').textContent    = data.case_preview.chief_complaint||'—';
      document.getElementById('pvAge').textContent   = (data.case_preview.age||'—')+' yrs';
      document.getElementById('pvGender').textContent= data.case_preview.gender||'—';
      document.getElementById('pvHR').textContent    = (v.heart_rate||'—')+' bpm';
      document.getElementById('pvBP').textContent    = v.blood_pressure||'—';
      document.getElementById('pvSPO2').textContent  = (v.oxygen_saturation||'—')+'%';
      document.getElementById('vitalsStrip').hidden  = false;

      const occ = data.case_preview.occupation||'';
      const persona = data.case_preview.persona_description||'';
      const occWrap = document.getElementById('pvOccupationWrap');
      const personaWrap = document.getElementById('pvPersonaWrap');
      document.getElementById('pvOccupation').textContent = occ;
      document.getElementById('pvPersona').textContent = persona;
      occWrap.hidden = !occ;
      personaWrap.hidden = !persona;
      document.getElementById('personaRow').hidden = !occ && !persona;

      document.getElementById('chatArea').innerHTML  = '<div class="flex justify-center"><span class="text-xs bg-slate-200 text-slate-500 px-3 py-1 rounded-full">Case ready — begin your assessment</span></div>';
      document.getElementById('invResults').innerHTML = '';
      document.getElementById('ecgContent').innerHTML = '';
      document.getElementById('ecgPanel').hidden = true;
      document.getElementById('interpList').innerHTML = '';
      document.getElementById('doctorConsults').innerHTML = '';
      document.getElementById('diagnosisResult').hidden = true;
      document.getElementById('evalResults').hidden = true;
      startGraphPolling();
      toast('Case generated', 'ok');
    } catch(e){ toast(e.message); } finally { spin('generateBtn',false); }
  }

  /* ── askQuestion ── */
  async function askQuestion() {
    try {
      req();
      const q = document.getElementById('question').value.trim(); if(!q) return;
      addBubble('you', q);
      document.getElementById('question').value = '';
      const typId = addTyping();
      spin('askBtn',true);
      const data = await postJson('/api/questions', {session_id:sessionId, question:q});
      document.getElementById(typId)?.remove();
      addBubble('patient', data.answer);
      speakText(data.answer);
      setTimeout(refreshGraph, 12000);
    } catch(e){ toast(e.message); } finally { spin('askBtn',false); }
  }
  function addBubble(role, text) {
    const area = document.getElementById('chatArea');
    const w = document.createElement('div');
    w.className = `flex ${role==='you'?'justify-end':'justify-start'} fade-in`;
    const b = document.createElement('div');
    b.className = `${role==='you'?'bubble-you':'bubble-patient'} px-4 py-2.5 text-sm leading-relaxed max-w-[80%] relative`;
    b.textContent = text;
    if (role === 'patient') {
      const replay = document.createElement('button');
      replay.className = 'replay-btn absolute -bottom-2 -right-2 w-6 h-6 bg-slate-100 hover:bg-slate-200 border border-slate-200 rounded-full flex items-center justify-center transition-all';
      replay.title = 'Replay';
      replay.innerHTML = '<svg class="w-3 h-3 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m-4.243-1.414a1 1 0 000 0M9 12h.01M12 6v12"/></svg>';
      replay.onclick = () => speakText(text);
      b.appendChild(replay);
    }
    w.appendChild(b); area.appendChild(w); area.scrollTop=area.scrollHeight;
  }
  function addTyping() {
    const area = document.getElementById('chatArea');
    const id = 'typ'+Date.now();
    const w = document.createElement('div'); w.id=id; w.className='flex justify-start fade-in';
    w.innerHTML='<div class="bubble-patient px-4 py-3 flex gap-1 items-center"><span class="dot w-1.5 h-1.5 bg-slate-400 rounded-full inline-block"></span><span class="dot w-1.5 h-1.5 bg-slate-400 rounded-full inline-block"></span><span class="dot w-1.5 h-1.5 bg-slate-400 rounded-full inline-block"></span></div>';
    area.appendChild(w); area.scrollTop=area.scrollHeight; return id;
  }

  /* ── investigations ── */
  async function orderInvestigation() {
    try {
      req();
      const inv = document.getElementById('investigation').value.trim(); if(!inv) return;
      spin('invBtn',true);
      const data = await postJson('/api/investigations', {session_id:sessionId, investigation:inv});
      renderInv(data);
      setTimeout(refreshGraph, 12000);
    } catch(e){ toast(e.message); } finally { spin('invBtn',false); }
  }
  function renderInv(result) {
    if(result.investigations){result.investigations.forEach(renderInv);return;}

    // ECG with SVG → render in the full-width ECG panel
    if(result.visual_payload&&result.visual_payload.svg){
      const panel = document.getElementById('ecgPanel');
      const content = document.getElementById('ecgContent');
      content.innerHTML = result.visual_payload.svg;
      panel.hidden = false;
      panel.scrollIntoView({behavior:'smooth', block:'nearest'});

      // Show a small badge in the sidebar linking down
      const c = document.getElementById('invResults');
      const badge = document.createElement('div');
      badge.className='fade-in border border-red-200 rounded-xl overflow-hidden';
      badge.innerHTML=`<div class="bg-red-50 px-3 py-2.5 flex items-center justify-between cursor-pointer" onclick="document.getElementById('ecgPanel').scrollIntoView({behavior:'smooth',block:'start'})">
        <span class="text-sm font-semibold text-red-800">ECG</span>
        <span class="text-xs text-red-600 bg-red-100 px-2 py-0.5 rounded-full flex items-center gap-1">view below ↓</span>
      </div>`;
      c.appendChild(badge);
      return;
    }

    const c = document.getElementById('invResults');
    const card = document.createElement('div');
    card.className='fade-in border border-slate-200 rounded-xl overflow-hidden';
    const head = document.createElement('div');
    head.className='bg-teal-50 px-3 py-2 flex items-center justify-between';
    head.innerHTML=`<span class="text-sm font-semibold text-teal-800">${result.investigation}</span><span class="text-xs text-teal-600 bg-teal-100 px-2 py-0.5 rounded-full">${result.status}</span>`;
    card.appendChild(head);
    const body = document.createElement('div'); body.className='p-3';
    if(result.report&&result.report.length){
      const ul=document.createElement('ul'); ul.className='space-y-1 text-xs text-slate-700';
      result.report.forEach(item=>{
        const li=document.createElement('li'); li.className='flex justify-between gap-2';
        if(item.finding) li.innerHTML=`<span>${item.finding}</span>`;
        else li.innerHTML=`<span class="font-medium">${item.name||''}</span><span class="text-slate-500">${item.value}${item.unit?' '+item.unit:''}<span class="text-slate-400 ml-1">${item.reference_range?'('+item.reference_range+')':''}</span></span>`;
        ul.appendChild(li);
      });
      body.appendChild(ul);
    }
    card.appendChild(body); c.appendChild(card);
  }

  /* ── interpretations ── */
  async function submitInterpretation() {
    try {
      req();
      const txt = document.getElementById('interpretation').value.trim(); if(!txt) return;
      spin('interpBtn',true);
      const data = await postJson('/api/interpretations', {session_id:sessionId, interpretation:txt});
      const ul = document.getElementById('interpList'); ul.innerHTML='';
      data.interpretations.forEach((item,i)=>{
        const li=document.createElement('li'); li.className='fade-in flex gap-2 text-sm text-slate-700';
        li.innerHTML=`<span class="text-violet-500 font-bold flex-shrink-0">${i+1}.</span>${item}`;
        ul.appendChild(li);
      });
      document.getElementById('interpretation').value='';
    } catch(e){ toast(e.message); } finally { spin('interpBtn',false); }
  }

  /* ── diagnosis ── */
  async function submitDiagnosis() {
    try {
      req();
      const dx = document.getElementById('diagnosisGuess').value.trim(); if(!dx) return;
      spin('diagnosisBtn',true);
      const data = await postJson('/api/diagnosis', {session_id:sessionId, diagnosis:dx});
      const el = document.getElementById('diagnosisResult');
      el.innerHTML=`<span class="font-semibold">Submitted:</span> ${data.diagnosis_guess}`;
      el.hidden=false;
    } catch(e){ toast(e.message); } finally { spin('diagnosisBtn',false); }
  }

  /* ── doctor consult ── */
  async function callDoctor() {
    try {
      req();
      const q = document.getElementById('doctorQuestion').value.trim(); if(!q) return;
      spin('doctorBtn',true);
      const data = await postJson('/api/doctor-consult', {session_id:sessionId, question:q});
      const urgencyClass = {routine:'urgency-routine', urgent:'urgency-urgent', critical:'urgency-critical'}[data.urgency_level]||'urgency-routine';
      const card=document.createElement('div');
      card.className='fade-in bg-amber-50 border border-amber-200 rounded-xl p-3 space-y-2';
      let html=`<div class="flex items-center justify-between"><span class="text-xs font-semibold text-amber-700">Doctor Response</span><span class="text-xs px-2 py-0.5 rounded-full font-semibold ${urgencyClass}">${data.urgency_level||''}</span></div>`;
      if(data.concise_advice) html+=`<p class="text-sm text-slate-700">${data.concise_advice}</p>`;
      if(data.recommended_next_steps&&data.recommended_next_steps.length){
        html+=`<ul class="text-xs text-slate-600 space-y-0.5 pl-1">`;
        data.recommended_next_steps.forEach(s=>html+=`<li>→ ${s}</li>`);
        html+=`</ul>`;
      }
      if(data.safety_warnings&&data.safety_warnings.length){
        html+=`<div class="bg-red-50 rounded-lg p-2 text-xs text-red-700 space-y-0.5">`;
        data.safety_warnings.forEach(w=>html+=`<p>⚠ ${w}</p>`);
        html+=`</div>`;
      }
      card.innerHTML=html;
      document.getElementById('doctorConsults').appendChild(card);
      document.getElementById('doctorQuestion').value='';
    } catch(e){ toast(e.message); } finally { spin('doctorBtn',false); }
  }

  /* ── knowledge graph ── */
  let _graphSim = null, _graphData = {nodes:[], edges:[]}, _pollTimer = null;
  let _graphZoom = null, _graphZoomBehavior = null;
  let _liveNodes = [];  // simulation-mutated node objects (have real x,y)

  const NODE_COLORS = {
    symptom:'#10b981', condition:'#10b981', sign:'#10b981',
    medication:'#ef4444', drug:'#ef4444', treatment:'#ef4444',
    time:'#f59e0b', temporal:'#f59e0b', duration:'#f59e0b',
    finding:'#06b6d4', result:'#06b6d4', investigation:'#06b6d4',
    riskfactor:'#f97316', risk:'#f97316',
    person:'#8b5cf6', patient:'#8b5cf6',
    bodypart:'#64748b', location:'#64748b',
    action:'#0ea5e9', measurement:'#0ea5e9',
  };
  function nodeColor(d) {
    for (const lbl of (d.labels||[])) {
      const key = lbl.toLowerCase();
      for (const [k,v] of Object.entries(NODE_COLORS)) { if(key.includes(k)) return v; }
    }
    return '#6366f1';
  }

  function graphZoomIn()  { if(_graphZoomBehavior) d3.select('#graphSvg').transition().duration(250).call(_graphZoomBehavior.scaleBy, 1.4); }
  function graphZoomOut() { if(_graphZoomBehavior) d3.select('#graphSvg').transition().duration(250).call(_graphZoomBehavior.scaleBy, 0.7); }
  function graphFit() {
    if (!_graphZoomBehavior || !_liveNodes.length) return;
    const svg = document.getElementById('graphSvg');
    const W = svg.clientWidth || 900, H = svg.clientHeight || 400;
    const xs = _liveNodes.map(n => n.x || W/2);
    const ys = _liveNodes.map(n => n.y || H/2);
    const x0 = Math.min(...xs) - 40, x1 = Math.max(...xs) + 40;
    const y0 = Math.min(...ys) - 40, y1 = Math.max(...ys) + 40;
    const scale = Math.min(0.92 * W / (x1 - x0), 0.92 * H / (y1 - y0), 2);
    const tx = W / 2 - scale * (x0 + x1) / 2;
    const ty = H / 2 - scale * (y0 + y1) / 2;
    d3.select('#graphSvg').transition().duration(400)
      .call(_graphZoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  let _selectedId = null;

  function resetHighlight(linkSel, edgeLblSel, nodeSel) {
    _selectedId = null;
    linkSel.attr('stroke','#c7d2fe').attr('stroke-width',1.5).attr('stroke-opacity',1);
    edgeLblSel.attr('opacity',1);
    nodeSel.select('circle').attr('r',22).attr('fill-opacity',.18).attr('stroke-opacity',1);
    nodeSel.selectAll('text').attr('opacity',1);
  }

  function highlightNode(d, links, linkSel, edgeLblSel, nodeSel) {
    if (_selectedId === d.id) { resetHighlight(linkSel, edgeLblSel, nodeSel); return; }
    _selectedId = d.id;

    // Build set of adjacent node IDs
    const adjIds = new Set([d.id]);
    links.forEach(lk => {
      if (lk.source.id===d.id) adjIds.add(lk.target.id);
      if (lk.target.id===d.id) adjIds.add(lk.source.id);
    });

    const isAdj = lk => lk.source.id===d.id || lk.target.id===d.id;

    // Dim non-adjacent edges
    linkSel
      .attr('stroke', lk => isAdj(lk) ? '#818cf8' : '#c7d2fe')
      .attr('stroke-width', lk => isAdj(lk) ? 2.2 : 1.5)
      .attr('stroke-opacity', lk => isAdj(lk) ? 1 : 0.10);

    // Dim non-adjacent edge labels
    edgeLblSel.attr('opacity', lk => isAdj(lk) ? 1 : 0);

    // Dim non-adjacent nodes
    nodeSel.select('circle')
      .attr('r', n => n.id===d.id ? 26 : 22)
      .attr('fill-opacity', n => n.id===d.id ? 0.30 : adjIds.has(n.id) ? 0.18 : 0.04)
      .attr('stroke-opacity', n => adjIds.has(n.id) ? 1 : 0.15);
    nodeSel.selectAll('text')
      .attr('opacity', n => adjIds.has(n.id) ? 1 : 0.15);
  }

  function renderGraph(nodes, edges) {
    const svgEl = document.getElementById('graphSvg');
    const W = svgEl.clientWidth || 900, H = svgEl.clientHeight || 400;
    const tooltip = document.getElementById('graphTooltip');
    document.getElementById('graphEmpty').style.display = nodes.length ? 'none' : 'flex';
    if (!nodes.length) return;

    if (_graphSim) _graphSim.stop();
    _selectedId = null;
    _liveNodes = nodes;
    const s = d3.select('#graphSvg');
    s.selectAll('*').remove();

    // Defs: arrowhead markers per color
    const defs = s.append('defs');
    defs.append('marker')
      .attr('id','arr').attr('viewBox','-0 -5 10 10')
      .attr('refX',30).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#94a3b8');
    defs.append('marker')
      .attr('id','arr-hi').attr('viewBox','-0 -5 10 10')
      .attr('refX',30).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
      .append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#818cf8');

    const byId = Object.fromEntries(nodes.map(n=>([n.id,n])));
    const links = edges.filter(e=>byId[e.source]&&byId[e.target]);

    // Detect connected components via union-find
    const _uf = {};
    nodes.forEach(n=>{ _uf[n.id]=n.id; });
    function _find(x){ return _uf[x]===x?x:(_uf[x]=_find(_uf[x])); }
    edges.forEach(e=>{
      const s=typeof e.source==='object'?e.source.id:e.source;
      const t=typeof e.target==='object'?e.target.id:e.target;
      if(_uf[s]!==undefined&&_uf[t]!==undefined) _uf[_find(s)]=_find(t);
    });
    const _compMap={};
    nodes.forEach(n=>{ const r=_find(n.id); (_compMap[r]=_compMap[r]||[]).push(n); });
    const _comps=Object.values(_compMap);

    // Assign a grid center to each component, spread across the canvas
    const _cols=Math.ceil(Math.sqrt(_comps.length));
    const _rows=Math.ceil(_comps.length/_cols);
    _comps.forEach((comp,i)=>{
      const cx=((i%_cols)+0.5)*(W/_cols);
      const cy=(Math.floor(i/_cols)+0.5)*(H/_rows);
      comp.forEach(n=>{ n._cx=cx; n._cy=cy; });
    });

    // Scatter initial positions near each component's assigned center
    nodes.forEach(n=>{ if(n.x==null){n.x=n._cx+(Math.random()-.5)*60;n.y=n._cy+(Math.random()-.5)*60;} });

    // Zoom/pan
    const root = s.append('g');
    _graphZoomBehavior = d3.zoom()
      .scaleExtent([0.15, 4])
      .on('zoom', e => { root.attr('transform', e.transform); });
    s.call(_graphZoomBehavior).on('dblclick.zoom', null);

    // Click background → reset highlight
    s.on('click', () => resetHighlight(linkSel, edgeLblSel, nodeSel));

    // ── Edge layer ────────────────────────────────────────────────────────────
    const linkSel = root.append('g').attr('class','links').selectAll('line')
      .data(links).enter().append('line')
      .attr('stroke','#c7d2fe').attr('stroke-width',1.5)
      .attr('marker-end','url(#arr)')
      .on('mouseenter',(ev,d)=>{ if(d.fact){ tooltip.textContent=d.fact; tooltip.style.display='block'; } })
      .on('mousemove',ev=>{ const r=svgEl.getBoundingClientRect(); tooltip.style.left=(ev.clientX-r.left+14)+'px'; tooltip.style.top=(ev.clientY-r.top-10)+'px'; })
      .on('mouseleave',()=>tooltip.style.display='none');

    // ── Edge labels (with white halo via CSS paint-order) ─────────────────────
    const edgeLblSel = root.append('g').selectAll('text')
      .data(links.filter(e=>e.fact)).enter().append('text')
      .attr('class','edge-label')
      .attr('text-anchor','middle').attr('font-size','9px').attr('font-weight','500')
      .attr('fill','#6366f1').attr('pointer-events','none')
      .text(d=>{ const f=d.fact||''; return f.length>16?f.slice(0,16)+'…':f; });

    // ── Node layer ─────────────────────────────────────────────────────────────
    const nodeSel = root.append('g').attr('class','nodes').selectAll('g')
      .data(nodes).enter().append('g').attr('class','node-circle')
      .call(d3.drag()
        .on('start',(ev,d)=>{ ev.sourceEvent.stopPropagation(); if(!ev.active)_graphSim.alphaTarget(.3).restart(); d.fx=d.x;d.fy=d.y; })
        .on('drag', (ev,d)=>{ d.fx=ev.x;d.fy=ev.y; })
        .on('end',  (ev,d)=>{ if(!ev.active)_graphSim.alphaTarget(0); d.fx=null;d.fy=null; })
      )
      .on('click',(ev,d)=>{ ev.stopPropagation(); highlightNode(d,links,linkSel,edgeLblSel,nodeSel); })
      .on('mouseenter',(ev,d)=>{
        const type = d.labels&&d.labels.length ? d.labels[0] : '';
        tooltip.innerHTML = `<strong>${d.name}</strong>${type?` <span style="opacity:.7;font-size:10px">[${type}]</span>`:''}`;
        tooltip.style.display='block';
      })
      .on('mousemove',ev=>{ const r=svgEl.getBoundingClientRect(); tooltip.style.left=(ev.clientX-r.left+14)+'px'; tooltip.style.top=(ev.clientY-r.top-10)+'px'; })
      .on('mouseleave',()=>tooltip.style.display='none');

    // Node circle — solid fill with entity-type color
    nodeSel.append('circle')
      .attr('r',22)
      .attr('fill',d=>nodeColor(d))
      .attr('fill-opacity',.18)
      .attr('stroke',d=>nodeColor(d))
      .attr('stroke-width',2);

    // Node name
    nodeSel.append('text')
      .attr('class','node-name')
      .attr('text-anchor','middle').attr('dy','-.1em')
      .attr('font-size','9px').attr('font-weight','700')
      .attr('fill','#1e293b').attr('pointer-events','none')
      .text(d=>{ const n=d.name||''; return n.length>11?n.slice(0,11)+'…':n; });

    // Entity-type label in the color of the node (smaller, below name)
    nodeSel.append('text')
      .attr('class','node-type')
      .attr('text-anchor','middle').attr('dy','1em')
      .attr('font-size','7.5px').attr('font-weight','400')
      .attr('fill',d=>nodeColor(d)).attr('pointer-events','none')
      .attr('fill-opacity',0.85)
      .text(d=> d.labels&&d.labels[0] ? d.labels[0].toLowerCase() : '');

    // ── Simulation ─────────────────────────────────────────────────────────────
    // Per-component cluster force: pulls each node toward its subgraph's assigned center
    function _clusterForce(alpha) {
      nodes.forEach(n=>{
        n.vx += (n._cx - n.x) * 0.12 * alpha;
        n.vy += (n._cy - n.y) * 0.12 * alpha;
      });
    }
    _graphSim = d3.forceSimulation(nodes)
      .force('link',    d3.forceLink(links).id(d=>d.id).distance(80).strength(0.9))
      .force('charge',  d3.forceManyBody().strength(-160))
      .force('collide', d3.forceCollide(34))
      .force('cluster', _clusterForce)
      .alphaDecay(0.03)
      .on('tick',()=>{
        // Clamp nodes within SVG bounds so they can never drift off screen
        nodes.forEach(d => {
          d.x = Math.max(30, Math.min(W - 30, d.x));
          d.y = Math.max(30, Math.min(H - 30, d.y));
        });
        linkSel
          .attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
        edgeLblSel
          .attr('x',d=>(d.source.x+d.target.x)/2)
          .attr('y',d=>(d.source.y+d.target.y)/2-5);
        nodeSel.attr('transform',d=>`translate(${d.x},${d.y})`);
      })
      .on('end', graphFit);
  }

  async function refreshGraph() {
    if (!sessionId) return;
    try {
      const r = await fetch('/api/graph/'+sessionId);
      if (!r.ok) return;
      const data = await r.json();
      const nPrev = _graphData.nodes.length;
      _graphData = data;
      if (data.nodes && (data.nodes.length !== nPrev || nPrev === 0)) {
        renderGraph(data.nodes||[], data.edges||[]);
      }
      const badge = document.getElementById('graphBadge');
      const status = document.getElementById('graphStatus');
      if (data.nodes && data.nodes.length) {
        badge.textContent = data.nodes.length+' entities · '+data.edges.length+' relations';
        badge.hidden = false;
        status.textContent = 'Live';
        status.className = 'text-xs text-emerald-300 font-medium';
        document.getElementById('graphControls').hidden = false;
      }
    } catch(_) {}
  }

  function startGraphPolling() {
    document.getElementById('graphPanel').hidden = false;
    document.getElementById('graphEmpty').style.display = 'flex';
    document.getElementById('graphStatus').textContent = 'Waiting for interactions…';
    document.getElementById('graphStatus').className = 'text-xs text-indigo-300 italic';
    document.getElementById('graphBadge').hidden = true;
    _graphData = {nodes:[], edges:[]};
    _liveNodes = [];
    _graphZoom = null; _graphZoomBehavior = null;
    if (_graphSim) { _graphSim.stop(); _graphSim=null; }
    d3.select('#graphSvg').selectAll('*').remove();
    document.getElementById('graphControls').hidden = true;
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(refreshGraph, 8000);
  }

  /* ── evaluate ── */
  async function evaluateCase() {
    try {
      req();
      spin('evaluateBtn',true);
      const data = await postJson('/api/evaluate', {session_id:sessionId});
      const score = data.overall_score||0;
      const circle = document.getElementById('scoreCircle');
      circle.textContent = score;
      circle.className = `score-ring shadow-xl ${score>=75?'bg-emerald-100 text-emerald-700 ring-4 ring-emerald-200':score>=50?'bg-amber-100 text-amber-700 ring-4 ring-amber-200':'bg-red-100 text-red-700 ring-4 ring-red-200'}`;
      document.getElementById('scoreFeedback').textContent = data.brief_feedback||'';

      const dx=data.correct_diagnosis_guess||{};
      document.getElementById('evalDx').innerHTML = dx.is_correct
        ?`<span class="text-emerald-600">✓ Correct</span> <span class="text-slate-400 font-normal">— ${dx.guess||''}</span>`
        :`<span class="text-red-500">✗ Incorrect</span> <span class="text-slate-400 font-normal">— ${dx.guess||'not submitted'}</span>`;

      const found=data.key_symptoms_identified||[], missed=data.key_symptoms_missed||[];
      document.getElementById('evalSymptoms').innerHTML=`<span class="text-emerald-600 font-semibold">${found.length} found</span>${missed.length?`, <span class="text-red-500">${missed.length} missed</span>`:''}`+(missed.length?`<p class="text-slate-400 text-xs mt-1">${missed.slice(0,3).join(', ')}</p>`:'');

      const rel=data.relevance_of_questions||{};
      document.getElementById('evalQuality').innerHTML = rel.relevant_count!==undefined
        ?`<span class="text-emerald-600 font-semibold">${rel.relevant_count}</span> relevant, <span class="text-slate-500">${rel.irrelevant_count}</span> off-target`:'—';

      const rf=data.reasoning_feedback||{}, rEl=document.getElementById('evalReasoning'); rEl.innerHTML='';
      if(rf.diagnostic_path){const p=document.createElement('p');p.textContent=rf.diagnostic_path;rEl.appendChild(p);}
      (rf.missed_reasoning_steps||[]).forEach(s=>{const p=document.createElement('p');p.className='text-slate-500';p.innerHTML=`<span class="text-amber-500 mr-1">→</span>${s}`;rEl.appendChild(p);});

      const lr=data.learning_recommendations||{}, lEl=document.getElementById('evalLearning'); lEl.innerHTML='';
      if(lr.focus_next){const p=document.createElement('p');p.className='font-semibold text-blue-700';p.textContent=lr.focus_next;lEl.appendChild(p);}
      (lr.action_items||[]).forEach(item=>{const p=document.createElement('p');p.innerHTML=`<span class="text-blue-400 mr-1">•</span>${item}`;lEl.appendChild(p);});

      const res=document.getElementById('evalResults'); res.hidden=false;
      res.scrollIntoView({behavior:'smooth',block:'start'});
    } catch(e){ toast(e.message); } finally { spin('evaluateBtn',false); }
  }

  setupSpeechRecognition();
</script>
</body>
</html>
"""
