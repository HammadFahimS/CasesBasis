"""Runtime for a simulated dynamic case.

:class:`SimulationEngine` drives one live encounter from a
:class:`schemas.simulated_case.SimulatedCase`. It owns the mutable run state
(current node, clock, what's been revealed) and resolves transitions as the
clock advances or the resident acts.

Trigger resolution is **hybrid**:

* deterministic triggers — ``time``/``inaction`` (clock), ``vitals`` (a guard on
  the live vitals) — fire on their own;
* ``action`` triggers match either by keyword (default, offline-friendly) or via
  an injected ``intent_matcher(action_text, transition_description) -> bool`` so
  an LLM can judge softer intent ("opts for intubation").

The engine is serialisable (:meth:`snapshot` / :meth:`restore`) so a run can be
parked in ``web_sessions.state_json`` and resumed, exactly like the manual
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Literal

from schemas.simulated_case import (
    InvestigationDef,
    ScenarioState,
    SimulatedCase,
    Transition,
    VitalSigns,
)

IntentMatcher = Callable[[str, str], bool]

ActionKind = Literal["ask", "exam", "order", "intervene", "medicate", "diagnose", "consult", "observe"]

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class LearnerAction:
    """One thing the resident does."""

    kind: ActionKind
    text: str = ""
    target: str | None = None  # e.g. investigation id or history beat id


@dataclass
class StepResult:
    """What the engine did in response to a single call."""

    state_id: str
    state_name: str
    transitioned: bool = False
    from_state: str | None = None
    message: str = ""
    revealed: str | None = None
    critical_error: bool = False


@dataclass
class RunState:
    """Mutable progress through a case. Fully serialisable."""

    current_state_id: str
    elapsed_seconds: int = 0
    state_entered_at: int = 0  # elapsed_seconds when the current state began
    revealed_history: list[str] = field(default_factory=list)
    ordered_investigations: list[str] = field(default_factory=list)
    actions_this_state: list[str] = field(default_factory=list)
    applied_modifiers: list[str] = field(default_factory=list)
    vitals_override: dict[str, object] = field(default_factory=dict)
    visited_states: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    working_diagnoses: list[str] = field(default_factory=list)
    fired_cues: list[str] = field(default_factory=list)
    log: list[dict[str, object]] = field(default_factory=list)
    ended: bool = False


class SimulationEngine:
    def __init__(self, case: SimulatedCase, intent_matcher: IntentMatcher | None = None):
        self.case = case
        self.intent_matcher = intent_matcher
        self.run = RunState(current_state_id=case.initial_state)
        self._enter_state(case.initial_state, record=True)

    # ----------------------------------------------------------------- views #
    @property
    def state(self) -> ScenarioState:
        return self.case.state(self.run.current_state_id)

    @property
    def elapsed_in_state(self) -> int:
        return self.run.elapsed_seconds - self.run.state_entered_at

    def current_vitals(self) -> VitalSigns:
        """State vitals with any in-state modifier overrides applied."""
        data = self.state.vitals.model_dump()
        data.update(self.run.vitals_override)
        return VitalSigns.model_validate(data)

    def _investigation(self, inv_id: str) -> InvestigationDef | None:
        for inv in self.case.investigations:
            if inv.id == inv_id:
                return inv
        return None

    # --------------------------------------------------------------- actions #
    def ask(self, history_id: str) -> StepResult:
        """Reveal a gated history beat if it's available."""
        beat = next((h for h in self.case.history if h.id == history_id), None)
        if beat is None:
            return self._result(message=f"No history beat '{history_id}'.")
        if beat.available_after_seconds and self.run.elapsed_seconds < beat.available_after_seconds:
            wait = beat.available_after_seconds - self.run.elapsed_seconds
            return self._result(message=f"Not available yet ({wait}s) — {beat.label}.")
        if history_id not in self.run.revealed_history:
            self.run.revealed_history.append(history_id)
        self._log("ask", history_id)
        # Asking can also satisfy an action transition (e.g. taking collateral hx).
        step = self._after_action(beat.label)
        step.revealed = beat.content
        return step

    def order_investigation(self, inv_id: str) -> StepResult:
        """Order a test and get its result (respecting availability + state)."""
        inv = self._investigation(inv_id)
        if inv is None:
            return self._result(message=f"No investigation '{inv_id}'.")
        if inv.available_after_seconds and self.run.elapsed_seconds < inv.available_after_seconds:
            wait = inv.available_after_seconds - self.run.elapsed_seconds
            return self._result(message=f"Result pending ({wait}s) — {inv.name}.")
        if inv_id not in self.run.ordered_investigations:
            self.run.ordered_investigations.append(inv_id)
        result = inv.result_by_state.get(self.run.current_state_id, inv.result)
        self._log("order", inv_id)
        step = self._after_action(inv.name)
        step.revealed = result
        return step

    def examine(self, system: str) -> StepResult:
        """Reveal the authored exam findings for a body system.

        Counts as an action (so e.g. a neuro exam can satisfy an action
        transition), and surfaces the findings as ``revealed`` text."""
        findings = [e for e in self.case.exam if e.system.lower() == system.lower()]
        self._log("exam", system)
        step = self._after_action(f"{system} exam")
        if not findings:
            step.revealed = f"No specific {system} findings documented."
            return step
        parts = []
        for finding in findings:
            line = f"{finding.system}: " + "; ".join(finding.findings)
            if finding.note:
                line += f" — {finding.note}"
            parts.append(line)
        step.revealed = "\n".join(parts)
        return step

    def diagnose(self, text: str) -> StepResult:
        """Record the resident's working diagnosis.

        Kept distinct from a generic intervention so the debrief can *score*
        recognition against the case. Still runs through ``_after_action`` so a
        case that wires a diagnosis to an action transition can use it; the engine
        itself never judges correctness (the scorer does)."""
        dx = (text or "").strip()
        if dx and dx not in self.run.working_diagnoses:
            self.run.working_diagnoses.append(dx)
        self._log("diagnose", dx)
        step = self._after_action(dx)
        step.revealed = f"Working diagnosis noted: {dx}" if dx else None
        return step

    def act(self, action: LearnerAction) -> StepResult:
        """Generic action (intervention/medication/diagnosis/consult/etc.)."""
        if action.kind == "ask" and action.target:
            return self.ask(action.target)
        if action.kind == "order" and action.target:
            return self.order_investigation(action.target)
        if action.kind == "diagnose":
            return self.diagnose(action.text or action.target or "")
        self._log(action.kind, action.text or action.target or "")
        return self._after_action(action.text or action.target or "")

    def tick(self, seconds: int) -> list[StepResult]:
        """Advance the clock and fire any time-based transitions/modifiers.

        Returns one StepResult per state change (usually zero or one)."""
        results: list[StepResult] = []
        self.run.elapsed_seconds += max(0, seconds)
        # Resolve cascading time transitions, but bound the loop.
        for _ in range(len(self.case.states) + 1):
            self._apply_time_modifiers()
            tr = self._due_time_transition()
            if tr is None:
                break
            results.append(self._fire(tr))
            if self.run.ended:
                break
        return results

    # Seconds in a state before its authored cues start to prompt the resident.
    CUE_AFTER_SECONDS = 75

    def due_cues(self) -> list[dict[str, str]]:
        """Authored cues that should prompt now — the room nudging a lingering team.

        Fires at most one cue per call (so prompts trickle rather than pile up),
        once the resident has been in the state a while. Each cue fires once; the
        record lives in ``run.fired_cues`` so it survives park/resume."""
        if self.elapsed_in_state < self.CUE_AFTER_SECONDS:
            return []
        for i, cue in enumerate(self.state.cues):
            key = f"{self.run.current_state_id}:{i}"
            if key in self.run.fired_cues:
                continue
            self.run.fired_cues.append(key)
            self._log("cue", cue.text)
            return [{"text": cue.text, "actor_id": cue.actor_id or ""}]
        return []

    def goto(self, state_id: str, reason: str = "") -> StepResult:
        """Force a transition (facilitator/manual override)."""
        from_state = self.run.current_state_id
        self._enter_state(state_id, record=True)
        self._log("goto", reason or state_id)
        return self._result(
            transitioned=True, from_state=from_state, message=reason or f"Moved to {state_id}."
        )

    # --------------------------------------------------------------- internals #
    def _after_action(self, text: str) -> StepResult:
        """Record an action's text and resolve any matching action transition."""
        norm = text.strip().lower()
        if norm:
            self.run.actions_this_state.append(norm)
            self._apply_action_modifiers(norm)
        for tr in self.state.transitions:
            if tr.kind != "action":
                continue
            if self._action_matches(norm, tr):
                return self._fire(tr)
        return self._result()

    def _action_matches(self, norm_text: str, tr: Transition) -> bool:
        if any(kw.lower() in norm_text for kw in tr.keywords):
            return True
        if self.intent_matcher is not None and norm_text:
            return bool(self.intent_matcher(norm_text, tr.description))
        return False

    def _due_time_transition(self) -> Transition | None:
        vitals = self.current_vitals()
        for tr in self.state.transitions:
            if tr.kind == "time" and self.elapsed_in_state >= (tr.after_seconds or 0):
                return tr
            if tr.kind == "inaction" and self.elapsed_in_state >= (tr.after_seconds or 0):
                if not self._any_action_taken(tr.keywords):
                    return tr
            if tr.kind == "vitals" and tr.vitals_check is not None:
                value = getattr(vitals, tr.vitals_check.param, None)
                if value is not None and _OPS[tr.vitals_check.op](float(value), tr.vitals_check.value):
                    return tr
        return None

    def _apply_time_modifiers(self) -> None:
        for i, mod in enumerate(self.state.modifiers):
            key = f"{self.run.current_state_id}:{i}"
            if key in self.run.applied_modifiers:
                continue
            due = mod.after_seconds is not None and self.elapsed_in_state >= mod.after_seconds
            if mod.when == "on_inaction" and due and not self._any_action_taken(mod.keywords):
                self._apply_modifier(mod, key)
            elif mod.when == "on_time" and due:
                self._apply_modifier(mod, key)

    def _apply_action_modifiers(self, norm_text: str) -> None:
        for i, mod in enumerate(self.state.modifiers):
            if mod.when != "on_action":
                continue
            key = f"{self.run.current_state_id}:{i}"
            if key in self.run.applied_modifiers:
                continue
            if any(kw.lower() in norm_text for kw in mod.keywords):
                self._apply_modifier(mod, key)

    def _apply_modifier(self, mod, key: str) -> None:
        self.run.applied_modifiers.append(key)
        if mod.set_vitals is not None:
            for field_name, value in mod.set_vitals.model_dump(exclude_none=True).items():
                if field_name == "extra" and not value:
                    continue
                self.run.vitals_override[field_name] = value
        self._log("modifier", mod.description)

    def _any_action_taken(self, keywords: list[str]) -> bool:
        if not keywords:
            return False
        kws = [k.lower() for k in keywords]
        return any(any(kw in act for kw in kws) for act in self.run.actions_this_state)

    def _fire(self, tr: Transition) -> StepResult:
        from_state = self.run.current_state_id
        if tr.critical_error:
            self.run.critical_errors.append(f"{from_state}->{tr.to_state}: {tr.description}")
        self._enter_state(tr.to_state, record=True)
        self._log("transition", f"{from_state} -> {tr.to_state}")
        return self._result(
            transitioned=True,
            from_state=from_state,
            message=tr.patient_effect or tr.description,
            critical_error=tr.critical_error,
        )

    def _enter_state(self, state_id: str, record: bool) -> None:
        state = self.case.state(state_id)  # raises if unknown
        self.run.current_state_id = state_id
        self.run.state_entered_at = self.run.elapsed_seconds
        self.run.actions_this_state = []
        self.run.vitals_override = {}
        if record:
            self.run.visited_states.append(state_id)
        if state.is_terminal:
            self.run.ended = True

    def _log(self, kind: str, detail: str) -> None:
        self.run.log.append(
            {"t": self.run.elapsed_seconds, "state": self.run.current_state_id, "kind": kind, "detail": detail}
        )

    def _result(self, **kwargs) -> StepResult:
        return StepResult(state_id=self.run.current_state_id, state_name=self.state.name, **kwargs)

    # --------------------------------------------------------------- persistence #
    def snapshot(self) -> dict[str, object]:
        return {"case_id": self.case.id, "run": self.run.__dict__}

    def restore(self, snapshot: dict[str, object]) -> None:
        # Tolerate snapshots written by an older/newer RunState shape: keep only
        # fields this version knows, so a renamed/dropped field can't 500 every
        # parked resume. Missing fields fall back to their dataclass defaults.
        run = snapshot.get("run") or {}
        known = {f.name for f in fields(RunState)}
        self.run = RunState(**{k: v for k, v in run.items() if k in known})  # type: ignore[arg-type]
