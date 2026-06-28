"""Schema for *simulated dynamic cases*.

A simulated case is a clinical **state machine**: the patient starts in an
initial state (vitals + status) and moves between states as time passes or as
the resident takes actions. This is what distinguishes it from the static
``ClinicalCase`` in ``clinical_case.py`` — here the patient *evolves*.

The structure is deliberately generic so any case (any specialty, any
presentation) can be authored against it and stored as JSON. CRUD tooling can
later read/write these documents; the runtime in ``backend.simulation_engine``
loads one and drives a live encounter from it.

Design notes
------------
* **Hybrid triggers.** Transitions carry a human/LLM-readable ``description``
  *and* optional deterministic fields (``after_seconds``, ``vitals_check``,
  ``keywords``). Deterministic triggers fire on their own; the description lets
  an LLM facilitator judge softer, intent-based triggers ("opts for
  intubation"). See :class:`Transition`.
* **Gated information.** History beats, exam findings and investigations are
  revealed only when the resident asks/orders them — mirroring a real
  encounter. Some become available only after a delay (e.g. family arriving).
* **Actors.** Sim actors (patient voice, EMS, family, consultants) carry a
  persona/script so an LLM can role-play them consistently.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

# Re-used from physical_exam.py so authored exams pin to the body SVG cleanly.
EXAM_SYSTEMS = [
    "General",
    "Cardiovascular",
    "Respiratory",
    "Abdominal",
    "Neurological",
    "Musculoskeletal",
    "Dermatological",
    "ENT/Head & Neck",
    "Peripheral Vascular",
]


class StrictModel(BaseModel):
    """Base model that rejects unplanned fields."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Authoring metadata                                                          #
# --------------------------------------------------------------------------- #
class CaseMeta(StrictModel):
    """Catalogue-level metadata — what shows up in a case library."""

    title: str
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""
    specialty: str | None = None
    setting: str = "Emergency Department"
    target_level: Literal["junior", "senior", "staff", "any"] = "any"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    estimated_minutes: int | None = None
    author: str | None = None
    institution: str | None = None
    source: str | None = None
    version: int = 1


class Objectives(StrictModel):
    """Educational objectives, kept as free-text bullet lists per audience."""

    overall_goal: str = ""
    medical: list[str] = Field(default_factory=list)
    nursing: list[str] = Field(default_factory=list)
    crm: list[str] = Field(default_factory=list)  # crisis resource management
    epas: list[str] = Field(default_factory=list)  # entrustable professional activities


# --------------------------------------------------------------------------- #
# Patient & vitals                                                            #
# --------------------------------------------------------------------------- #
class PatientProfile(StrictModel):
    """Demographics + background. ``persona`` instructs the LLM patient voice."""

    name: str | None = None
    age: int
    gender: str
    weight_kg: float | None = None
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    past_medical_history: list[str] = Field(default_factory=list)
    social_history: list[str] = Field(default_factory=list)
    persona: str = ""


class VitalSigns(StrictModel):
    """A vitals snapshot. Every field is optional so states can set only what
    changes; ``extra`` carries case-specific measurements (e.g. vital capacity,
    negative inspiratory force) without bloating the core model."""

    rhythm: str | None = None
    heart_rate: int | None = None
    blood_pressure: str | None = None
    respiratory_rate: int | None = None
    oxygen_saturation: int | None = None
    temperature_c: float | None = None
    glucose: float | None = None
    gcs: str | None = None  # "15" or "intubated" etc.
    fio2: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)

    # Field names that VitalsCheck.param is allowed to reference.
    NUMERIC_PARAMS: ClassVar[set[str]] = {
        "heart_rate",
        "respiratory_rate",
        "oxygen_saturation",
        "temperature_c",
        "glucose",
    }


# --------------------------------------------------------------------------- #
# Initial presentation                                                        #
# --------------------------------------------------------------------------- #
class Presentation(StrictModel):
    """What the resident is handed at t=0."""

    chief_complaint: str
    opening_vignette: str = ""  # read aloud at the start
    triage_note: list[str] = Field(default_factory=list)
    initial_vitals: VitalSigns
    on_monitor: bool = False


# --------------------------------------------------------------------------- #
# Gated information                                                           #
# --------------------------------------------------------------------------- #
class HistoryBeat(StrictModel):
    """A piece of history revealed on request (or after a delay / trigger)."""

    id: str
    label: str
    content: str
    provider: Literal["patient", "ems", "family", "nurse", "chart", "other"] = "patient"
    reveal: Literal["on_ask", "on_arrival", "on_trigger"] = "on_ask"
    available_after_seconds: int | None = None


class ExamFinding(StrictModel):
    """Exam result for one body system. ``note`` captures dynamic cues that an
    examiner would only see live (e.g. weakness that fatigues during testing)."""

    system: str
    findings: list[str]
    abnormal: bool = False
    note: str = ""


class ActorRole(StrictModel):
    """A character the LLM can role-play (patient, EMS, family, consultant)."""

    id: str
    role: str
    voice: str = ""
    persona: str = ""
    script: list[str] = Field(default_factory=list)
    # When this actor becomes present in the room, in case-clock seconds. ``None``
    # (the default) means present from the start; a value means the actor is "en
    # route" until the clock reaches it, then arrives (e.g. the husband at 300s).
    # Drives the roster's en-route → present state generically, per case.
    available_after_seconds: int | None = None


class InvestigationDef(StrictModel):
    """An orderable test and its result. ``result_by_state`` lets a result
    differ depending on which state it's ordered in (e.g. a repeat gas)."""

    id: str
    name: str
    category: Literal["lab", "imaging", "ecg", "bedside", "other"] = "lab"
    # The RAW finding the resident sees/hears — values, measurements, what's on the
    # film. It must NOT contain the clinical read or a management directive: the
    # resident interprets it. (Tests/exams hand over signal, not the answer.)
    result: str = ""
    result_by_state: dict[str, str] = Field(default_factory=dict)
    # The clinical interpretation of the result — deliberately hidden during the
    # case and surfaced only at debrief, so teaching value is kept without giving
    # the diagnosis away live (e.g. "type-2 respiratory failure").
    interpretation: str = ""
    image_ref: str | None = None
    available_after_seconds: int | None = None
    # Turn-around time *relative to when the test is ordered* (vs the absolute-clock
    # ``available_after_seconds``). When set, ordering the test schedules a deferred
    # result: the holding actor reports the number back ~this many seconds later and
    # the card lands in the diagnostics panel then — the "room talks back" loop for
    # bedside spirometry, sent labs, etc. See SimulationEngine.due_deferred.
    result_latency_seconds: int | None = None
    # Which actor delivers the result aloud when it comes back (e.g. the RT for
    # bedside VC/NIF). Defaults by category (bedside→RT, else the nurse).
    result_actor_id: str | None = None
    # Diagnosis/finding hint that drives the real ECG/CXR image generators
    # (e.g. "sinus tachycardia", "normal chest"). Falls back to the result text.
    visual_dx: str | None = None


# --------------------------------------------------------------------------- #
# State machine                                                               #
# --------------------------------------------------------------------------- #
class VitalsCheck(StrictModel):
    """Deterministic guard on a numeric vital, e.g. oxygen_saturation < 85."""

    param: str
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float

    @model_validator(mode="after")
    def validate_param(self) -> "VitalsCheck":
        if self.param not in VitalSigns.NUMERIC_PARAMS:
            raise ValueError(
                f"vitals_check.param must be one of {sorted(VitalSigns.NUMERIC_PARAMS)}"
            )
        return self


class Transition(StrictModel):
    """An edge to another state.

    ``kind`` selects how it fires:
      * ``action``  — a resident action matching ``keywords`` (deterministic) or
        ``description`` (LLM intent match).
      * ``time``    — ``after_seconds`` elapsed *in the current state*.
      * ``vitals``  — ``vitals_check`` is satisfied by the current vitals.
      * ``inaction``— ``after_seconds`` elapsed without the keyworded action.
      * ``manual``  — only a facilitator/engine call moves it.

    ``critical_error`` marks paths taken by mistake (e.g. wrong paralytic →
    arrest), which the evaluator can score against.
    """

    to_state: str
    description: str
    kind: Literal["action", "time", "vitals", "inaction", "manual"] = "action"
    keywords: list[str] = Field(default_factory=list)
    after_seconds: int | None = None
    vitals_check: VitalsCheck | None = None
    patient_effect: str = ""
    critical_error: bool = False

    @model_validator(mode="after")
    def validate_kind_payload(self) -> "Transition":
        if self.kind in {"time", "inaction"} and self.after_seconds is None:
            raise ValueError(f"{self.kind} transition requires after_seconds")
        if self.kind == "vitals" and self.vitals_check is None:
            raise ValueError("vitals transition requires vitals_check")
        return self


class StateModifier(StrictModel):
    """An in-state change to the patient (no state hop). Used for the PDF's
    'Modifiers' — e.g. deterioration if an action is delayed."""

    description: str
    when: Literal["on_action", "on_inaction", "on_time"] = "on_inaction"
    after_seconds: int | None = None
    keywords: list[str] = Field(default_factory=list)
    set_vitals: VitalSigns | None = None


class Cue(StrictModel):
    """A facilitator/actor prompt injected to nudge stuck learners."""

    text: str
    actor_id: str | None = None
    when: str = ""
    after_seconds: int | None = None
    # Presence gate (Room-of-Agents C1): the cue only fires once *every* actor id
    # listed here is in the room. A cue that references the husband ("ask him about
    # her symptoms") must not fire while he's still en route — otherwise the room
    # announces an arrival that hasn't happened. Empty = no presence requirement.
    requires_present: list[str] = Field(default_factory=list)


class ScenarioState(StrictModel):
    """One node of the simulation."""

    id: str
    name: str
    order: int = 0
    vitals: VitalSigns
    patient_status: str = ""
    expected_actions: list[str] = Field(default_factory=list)
    modifiers: list[StateModifier] = Field(default_factory=list)
    cues: list[Cue] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    facilitator_notes: str = ""
    is_terminal: bool = False


class Debrief(StrictModel):
    """Post-case teaching material (the facilitator cheat sheet)."""

    summary: str = ""
    teaching_points: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    discussion: list[dict[str, str]] = Field(default_factory=list)  # {question, answer}
    references: list[str] = Field(default_factory=list)


class MedicationExpectation(StrictModel):
    """One graded expectation about a medication the resident might give.

    Powers the debrief's medication-correctness scoring (``backend.sim_scoring``).
    The matching is keyword-based, consistent with the rest of the engine:

    * ``preferred``   — the right agent to reach for; *met* if any ``drugs`` was given.
    * ``avoid``       — should not be used; *met* if none of ``drugs`` was given.
    * ``dose_adjust`` — the dose must be tuned for this patient; only graded *if*
      the drug was given, and *met* only when the order also voices the right
      direction (``dose_keywords``, e.g. "reduced" / "increased").
    """

    id: str
    label: str
    drugs: list[str] = Field(default_factory=list)  # match if any appears in an order
    expectation: Literal["preferred", "avoid", "dose_adjust"] = "preferred"
    dose_keywords: list[str] = Field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
# Root document                                                               #
# --------------------------------------------------------------------------- #
class SimulatedCase(StrictModel):
    """Source-of-truth document for one authored simulated dynamic case."""

    id: str
    schema_version: int = SCHEMA_VERSION
    meta: CaseMeta
    objectives: Objectives = Field(default_factory=Objectives)
    patient: PatientProfile
    presentation: Presentation
    history: list[HistoryBeat] = Field(default_factory=list)
    exam: list[ExamFinding] = Field(default_factory=list)
    actors: list[ActorRole] = Field(default_factory=list)
    investigations: list[InvestigationDef] = Field(default_factory=list)
    initial_state: str
    states: list[ScenarioState]
    debrief: Debrief = Field(default_factory=Debrief)
    medication_rubric: list[MedicationExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "SimulatedCase":
        if not self.states:
            raise ValueError("a simulated case needs at least one state")

        ids = [s.id for s in self.states]
        if len(ids) != len(set(ids)):
            raise ValueError("state ids must be unique")
        state_ids = set(ids)

        if self.initial_state not in state_ids:
            raise ValueError(f"initial_state '{self.initial_state}' is not a defined state")

        for state in self.states:
            for tr in state.transitions:
                if tr.to_state not in state_ids:
                    raise ValueError(
                        f"state '{state.id}' transitions to unknown state '{tr.to_state}'"
                    )

        # Referential checks for actor-linked cues.
        actor_ids = {a.id for a in self.actors}
        for state in self.states:
            for cue in state.cues:
                if cue.actor_id and cue.actor_id not in actor_ids:
                    raise ValueError(
                        f"cue in state '{state.id}' references unknown actor '{cue.actor_id}'"
                    )
                for aid in cue.requires_present:
                    if aid not in actor_ids:
                        raise ValueError(
                            f"cue in state '{state.id}' requires unknown actor '{aid}'"
                        )

        # Referential check for investigation result holders.
        for inv in self.investigations:
            if inv.result_actor_id and inv.result_actor_id not in actor_ids:
                raise ValueError(
                    f"investigation '{inv.id}' names unknown result_actor_id "
                    f"'{inv.result_actor_id}'"
                )
        return self

    def state(self, state_id: str) -> ScenarioState:
        for s in self.states:
            if s.id == state_id:
                return s
        raise KeyError(state_id)
