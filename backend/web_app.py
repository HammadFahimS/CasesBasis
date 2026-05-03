from __future__ import annotations

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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.post("/api/cases")
def generate_case(request: GenerateCaseRequest) -> dict[str, Any]:
    clinical_case = get_or_generate_cached_case(request.diagnosis, request.difficulty)
    session_id = str(uuid4())
    SESSIONS[session_id] = WebSession(case=clinical_case)
    return {
        "session_id": session_id,
        "case_preview": case_preview(clinical_case),
    }


@app.post("/api/questions")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    answer = get_patient_service().answer_question(session.case, request.question)
    session.student_questions.append(request.question)
    session.patient_responses.append(answer)
    return {
        "question": request.question,
        "answer": answer,
    }


@app.post("/api/investigations")
def order_investigation(request: InvestigationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    result = get_investigation_service().get_result(session.case, request.investigation)
    session.investigations.append(result)
    return student_facing_investigation_result(result, session.case)


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
    response = get_doctor_consult_service().consult(
        session.case,
        session.student_questions,
        session.patient_responses,
        session.investigations,
        request.question,
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

    evaluation = get_evaluator().evaluate(
        session.case,
        questions,
        responses,
        investigations_ordered=session.investigations,
        student_interpretations=session.interpretations,
        session_history=[],
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
<html>
  <head>
    <meta charset="utf-8" />
    <title>Clinical Case Simulator</title>
    <style>
      .ecg-tracing {
        overflow-x: auto;
        max-width: 100%;
        padding: 8px;
        border: 1px solid #ccc;
        background: #fff;
      }
      .ecg-tracing svg {
        display: block;
        min-width: 720px;
        max-width: 100%;
        height: auto;
      }
    </style>
  </head>
  <body>
    <h1>Clinical Case Simulator</h1>

    <section>
      <h2>Generate Case</h2>
      <label>Diagnosis <input id="diagnosis" value="myocardial infarction" /></label>
      <label>Difficulty
        <select id="difficulty">
          <option>easy</option>
          <option selected>medium</option>
          <option>hard</option>
        </select>
      </label>
      <button id="generateButton" onclick="generateCase()">Generate</button>
      <span id="generateStatus" aria-live="polite"></span>
      <pre id="casePreview"></pre>
    </section>

    <section>
      <h2>Patient Questions</h2>
      <input id="question" placeholder="Ask the patient..." />
      <button id="askButton" onclick="askQuestion()">Ask</button>
      <span id="askStatus" aria-live="polite"></span>
      <div id="chat"></div>
    </section>

    <section>
      <h2>Investigations</h2>
      <input id="investigation" value="ECG" />
      <button id="investigationButton" onclick="orderInvestigation()">Order Test</button>
      <span id="investigationStatus" aria-live="polite"></span>
      <div id="investigations"></div>
    </section>

    <section>
      <h2>Interpretations</h2>
      <textarea id="interpretation" rows="3" cols="80" placeholder="Enter your interpretation of ordered tests..."></textarea>
      <br />
      <button id="interpretationButton" onclick="submitInterpretation()">Submit Interpretation</button>
      <span id="interpretationStatus" aria-live="polite"></span>
      <ul id="interpretations"></ul>
    </section>

    <section>
      <h2>Doctor Consultation</h2>
      <textarea id="doctorQuestion" rows="3" cols="80" placeholder="Ask the doctor for quick guidance..."></textarea>
      <br />
      <button id="doctorButton" onclick="callDoctor()">Call Doctor</button>
      <span id="doctorStatus" aria-live="polite"></span>
      <div id="doctorConsults"></div>
    </section>

    <section>
      <h2>Diagnosis</h2>
      <input id="diagnosisGuess" placeholder="Submit diagnosis guess..." />
      <button id="diagnosisButton" onclick="submitDiagnosis()">Submit Diagnosis</button>
      <span id="diagnosisStatus" aria-live="polite"></span>
      <pre id="diagnosisResult"></pre>
    </section>

    <section>
      <h2>Final Evaluation</h2>
      <button id="evaluateButton" onclick="evaluateCase()">Evaluate</button>
      <span id="evaluateStatus" aria-live="polite"></span>
      <h3>Evaluation</h3>
      <pre id="evaluation"></pre>
      <h3>Reasoning Feedback</h3>
      <pre id="reasoning"></pre>
      <h3>Learning Recommendations</h3>
      <pre id="learning"></pre>
    </section>

    <script>
      let sessionId = null;

      async function postJson(url, payload) {
        const response = await fetch(url, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const body = await response.text();
        const data = parseJsonBody(body);
        if (!response.ok) {
          throw new Error(formatErrorMessage(data.detail || body || "Request failed"));
        }
        return data;
      }

      function parseJsonBody(body) {
        if (!body) {
          return {};
        }
        try {
          return JSON.parse(body);
        } catch {
          return {detail: body};
        }
      }

      function formatErrorMessage(detail) {
        if (typeof detail === "string") {
          return detail;
        }
        return JSON.stringify(detail);
      }

      function requireSession() {
        if (!sessionId) {
          throw new Error("Generate a case first.");
        }
      }

      function appendPre(parentId, title, payload) {
        const wrapper = document.createElement("div");
        const heading = document.createElement("strong");
        heading.textContent = title;
        const pre = document.createElement("pre");
        pre.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
        wrapper.appendChild(heading);
        wrapper.appendChild(pre);
        document.getElementById(parentId).appendChild(wrapper);
      }

      function appendInvestigationDisplay(result) {
        const wrapper = document.createElement("div");
        const heading = document.createElement("strong");
        heading.textContent = result.investigation;
        const summary = document.createElement("p");
        summary.textContent = result.summary;
        wrapper.appendChild(heading);
        wrapper.appendChild(summary);

        if (result.report && result.report.length) {
          const list = document.createElement("ul");
          result.report.forEach(item => {
            const li = document.createElement("li");
            if (Object.prototype.hasOwnProperty.call(item, "finding")) {
              li.textContent = item.finding;
            } else {
              const name = item.name || result.investigation;
              const unit = item.unit ? " " + item.unit : "";
              const range = item.reference_range ? " (reference range " + item.reference_range + ")" : "";
              li.textContent = name + ": " + item.value + unit + range;
            }
            list.appendChild(li);
          });
          wrapper.appendChild(list);
        }

        if (result.visual_payload && result.visual_payload.svg) {
          const div = document.createElement("div");
          div.className = "ecg-tracing";
          div.innerHTML = result.visual_payload.svg;
          wrapper.appendChild(div);
        }
        document.getElementById("investigations").appendChild(wrapper);
      }

      function setBusy(buttonId, statusId, message) {
        const button = document.getElementById(buttonId);
        const status = document.getElementById(statusId);
        button.disabled = true;
        status.textContent = message;
        return {button, status};
      }

      async function generateCase() {
        const busy = setBusy("generateButton", "generateStatus", "Generating case... this may take a few seconds");
        try {
          const data = await postJson("/api/cases", {
            diagnosis: document.getElementById("diagnosis").value,
            difficulty: document.getElementById("difficulty").value
          });
          sessionId = data.session_id;
          busy.status.textContent = "";
          document.getElementById("casePreview").textContent = JSON.stringify(data.case_preview, null, 2);
          document.getElementById("chat").innerHTML = "";
          document.getElementById("investigations").innerHTML = "";
          document.getElementById("interpretations").innerHTML = "";
          document.getElementById("doctorConsults").innerHTML = "";
          document.getElementById("diagnosisResult").textContent = "";
          document.getElementById("evaluation").textContent = "";
          document.getElementById("reasoning").textContent = "";
          document.getElementById("learning").textContent = "";
          document.getElementById("askStatus").textContent = "";
          document.getElementById("investigationStatus").textContent = "";
          document.getElementById("interpretationStatus").textContent = "";
          document.getElementById("doctorStatus").textContent = "";
          document.getElementById("diagnosisStatus").textContent = "";
          document.getElementById("evaluateStatus").textContent = "";
        } catch (error) {
          busy.status.textContent = "Case generation failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function askQuestion() {
        const busy = setBusy("askButton", "askStatus", "Asking patient...");
        try {
          requireSession();
          const question = document.getElementById("question").value;
          const data = await postJson("/api/questions", {session_id: sessionId, question});
          busy.status.textContent = "";
          appendPre("chat", "Student", data.question);
          appendPre("chat", "Patient", data.answer);
          document.getElementById("question").value = "";
        } catch (error) {
          busy.status.textContent = "Patient question failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function orderInvestigation() {
        const busy = setBusy("investigationButton", "investigationStatus", "Ordering investigation...");
        try {
          requireSession();
          const data = await postJson("/api/investigations", {
            session_id: sessionId,
            investigation: document.getElementById("investigation").value
          });
          busy.status.textContent = "";
          appendInvestigationDisplay(data);
        } catch (error) {
          busy.status.textContent = "Investigation order failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function submitInterpretation() {
        const busy = setBusy("interpretationButton", "interpretationStatus", "Submitting interpretation...");
        try {
          requireSession();
          const interpretation = document.getElementById("interpretation").value;
          const data = await postJson("/api/interpretations", {session_id: sessionId, interpretation});
          busy.status.textContent = "";
          const list = document.getElementById("interpretations");
          list.innerHTML = "";
          data.interpretations.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            list.appendChild(li);
          });
          document.getElementById("interpretation").value = "";
        } catch (error) {
          busy.status.textContent = "Interpretation submission failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function submitDiagnosis() {
        const busy = setBusy("diagnosisButton", "diagnosisStatus", "Submitting diagnosis...");
        try {
          requireSession();
          const data = await postJson("/api/diagnosis", {
            session_id: sessionId,
            diagnosis: document.getElementById("diagnosisGuess").value
          });
          busy.status.textContent = "";
          document.getElementById("diagnosisResult").textContent = JSON.stringify(data, null, 2);
        } catch (error) {
          busy.status.textContent = "Diagnosis submission failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function callDoctor() {
        const busy = setBusy("doctorButton", "doctorStatus", "Calling doctor...");
        try {
          requireSession();
          const question = document.getElementById("doctorQuestion").value;
          const data = await postJson("/api/doctor-consult", {session_id: sessionId, question});
          busy.status.textContent = "";
          appendPre("doctorConsults", "Doctor", data);
          document.getElementById("doctorQuestion").value = "";
        } catch (error) {
          busy.status.textContent = "Doctor consult failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }

      async function evaluateCase() {
        const busy = setBusy("evaluateButton", "evaluateStatus", "Evaluating performance...");
        try {
          requireSession();
          const data = await postJson("/api/evaluate", {session_id: sessionId});
          busy.status.textContent = "";
          document.getElementById("evaluation").textContent = JSON.stringify(data, null, 2);
          document.getElementById("reasoning").textContent = JSON.stringify(data.reasoning_feedback, null, 2);
          document.getElementById("learning").textContent = JSON.stringify(data.learning_recommendations, null, 2);
        } catch (error) {
          busy.status.textContent = "Evaluation failed.";
          alert(error.message);
        } finally {
          busy.button.disabled = false;
        }
      }
    </script>
  </body>
</html>
"""
