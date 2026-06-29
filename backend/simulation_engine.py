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
from typing import Callable, ClassVar, Literal

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
    # Turn-spanning protocol state: a confirmable action (e.g. a high-alert med
    # awaiting its dose, or an ICU admission awaiting a read-back confirmation) the
    # nurse has parked across turns. Lives on the run so it survives park/resume.
    # Shape: {op, target, dose, action_text, label, status}; status is one of
    # "awaiting_info" (a slot is missing) | "awaiting_confirm" (read back, awaiting
    # go-ahead). None when nothing is pending. Owned by nurse_brain.dispatch.
    pending_action: dict | None = None
    # Room-of-Agents presence (RA-1): actor ids the resident has explicitly summoned
    # ("call the husband", "page neurology"). Presence itself is *computed* live by
    # SimulationEngine.is_present (time-arrival + summon); only the summon events
    # need to persist. See docs/room-of-agents.md C1.
    summoned_actors: list[str] = field(default_factory=list)
    # Actors whose arrival has already been announced to the room, so the "X has
    # just arrived" line fires exactly once when presence flips true (RA-1).
    announced_arrivals: list[str] = field(default_factory=list)
    # Scheduled "room talks back" results: tests ordered before they're ready. Each
    # is {"inv_id", "fire_at" (elapsed_seconds), "actor_id"}; drained by due_deferred
    # once the clock passes fire_at, producing a spoken report + a panel card.
    deferred_events: list[dict] = field(default_factory=list)


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

    # ----------------------------------------------------------------- presence #
    # Room-of-Agents C1: the room knows who is here. The Router consults this so
    # that addressing an absent actor yields an honest "they're not here yet" line
    # instead of falling to the nurse sink. See docs/room-of-agents.md.

    def _actor(self, actor_id: str | None):
        if not actor_id:
            return None
        return next((a for a in self.case.actors if a.id == actor_id), None)

    def is_present(self, actor_id: str | None) -> bool:
        """Is this actor in the room (addressable directly) right now?

        Presence = explicitly summoned, OR a non-consultant who has arrived:
        ``available_after_seconds`` is None (here from the start) or has elapsed.
        Consultants are never "in the room" by the clock — they are reached only by
        being called (``summon`` via request_consult), since they advise by phone /
        arrive on request."""
        actor = self._actor(actor_id)
        if actor is None:
            return False
        if actor.id in self.run.summoned_actors:
            return True
        from backend.actor_voice import actor_speaker
        if actor_speaker(actor) == "consult":
            return False
        after = getattr(actor, "available_after_seconds", None)
        return after is None or self.run.elapsed_seconds >= after

    def present_actor_ids(self) -> set[str]:
        """The set of actor ids currently in the room."""
        return {a.id for a in self.case.actors if self.is_present(a.id)}

    def _beat_holder_present(self, beat) -> bool:
        """Is the actor who holds this history beat currently in the room? Maps the
        beat's provider kind to a case actor and checks presence (used so an
        early-summoned holder's timer-gated beats become answerable)."""
        from backend.actor_voice import actor_speaker, provider_speaker
        kind = provider_speaker(beat.provider)
        holder = next((a for a in self.case.actors if actor_speaker(a) == kind), None)
        return holder is not None and self.is_present(holder.id)

    def arrival_wait(self, actor_id: str | None) -> int | None:
        """Seconds until a timed-arrival actor shows up, or None if not applicable
        (already present, summoned, a consultant, or no timer)."""
        actor = self._actor(actor_id)
        if actor is None or self.is_present(actor_id):
            return None
        after = getattr(actor, "available_after_seconds", None)
        if after is None:
            return None
        return max(0, after - self.run.elapsed_seconds)

    def summon(self, actor_id: str) -> bool:
        """Bring an actor into the room now (resident actively called them). Returns
        True if this changed presence. Idempotent."""
        actor = self._actor(actor_id)
        if actor is None or actor_id in self.run.summoned_actors:
            return False
        self.run.summoned_actors.append(actor_id)
        # A summon is announced by the nurse at the moment it happens ("calling them
        # in now"), so suppress the generic arrival announce for this actor.
        if actor_id not in self.run.announced_arrivals:
            self.run.announced_arrivals.append(actor_id)
        self._log("summon", actor_id)
        return True

    def due_arrivals(self) -> list[dict[str, str]]:
        """Timed-arrival actors who just entered the room and haven't been announced.

        Fires once per actor the moment presence flips true (the clock reaches their
        ``available_after_seconds``). This is what makes "her husband's just arrived"
        land *when he actually arrives* — not on a state-linger timer that can fire
        before he's here (RA-1/C1)."""
        out: list[dict[str, str]] = []
        for actor in self.case.actors:
            after = getattr(actor, "available_after_seconds", None)
            if after is None:                              # in the room from the start
                continue
            if actor.id in self.run.announced_arrivals:
                continue
            if not self.is_present(actor.id):
                continue
            self.run.announced_arrivals.append(actor.id)
            self._log("arrival", actor.id)
            out.append({"actor_id": actor.id, "role": actor.role or actor.id})
        return out

    # --------------------------------------------------------------- actions #
    def ask(self, history_id: str) -> StepResult:
        """Reveal a gated history beat if it's available."""
        beat = next((h for h in self.case.history if h.id == history_id), None)
        if beat is None:
            return self._result(message=f"No history beat '{history_id}'.")
        if (
            beat.available_after_seconds
            and self.run.elapsed_seconds < beat.available_after_seconds
            and not self._beat_holder_present(beat)
        ):
            # The beat's timer is a proxy for "its holder isn't here yet". Once the
            # holder is in the room (e.g. the husband was summoned early), presence
            # supersedes the timer — they can answer for themselves (RA-1/C1).
            wait = beat.available_after_seconds - self.run.elapsed_seconds
            return self._result(message=f"Not available yet ({wait}s) — {beat.label}.")
        if history_id not in self.run.revealed_history:
            self.run.revealed_history.append(history_id)
        self._log("ask", history_id)
        # Asking can also satisfy an action transition (e.g. taking collateral hx).
        step = self._after_action(beat.label)
        step.revealed = beat.content
        return step

    def _result_holder(self, inv: InvestigationDef) -> str:
        """Which actor id delivers this investigation's result aloud."""
        if inv.result_actor_id:
            return inv.result_actor_id
        if inv.category == "bedside":
            from backend.actor_voice import actor_speaker
            rt = next((a for a in self.case.actors if actor_speaker(a) == "rt"), None)
            if rt is not None:
                return rt.id
        return "nurse"

    def _result_ready_at(self, inv: InvestigationDef) -> int:
        """Case-clock second at which this test's result is available.

        ``available_after_seconds`` is an absolute-clock gate; ``result_latency_seconds``
        is measured from *now* (order time). A test takes whichever is later."""
        absolute = inv.available_after_seconds or 0
        relative = self.run.elapsed_seconds + (inv.result_latency_seconds or 0)
        return max(absolute, relative)

    def order_investigation(self, inv_id: str) -> StepResult:
        """Order a test and get its result (respecting availability + state).

        If the result isn't ready yet, the order is *recorded* and a deferred event
        is scheduled so the holding actor reports the number back when it lands —
        instead of a dead "pending" that never returns (the open-loop bug)."""
        inv = self._investigation(inv_id)
        if inv is None:
            return self._result(message=f"No investigation '{inv_id}'.")
        ready_at = self._result_ready_at(inv)
        if ready_at > self.run.elapsed_seconds:
            if inv_id not in self.run.ordered_investigations:
                self.run.ordered_investigations.append(inv_id)
            self._schedule_deferred(inv, ready_at)
            wait = ready_at - self.run.elapsed_seconds
            holder = self._result_holder(inv)
            who = self._actor(holder)
            who_label = (who.role if who else "the team") or "the team"
            self._log("order", inv_id)
            return self._result(
                message=(
                    f"{who_label} is on the {inv.name} now — the result will come "
                    f"back in about {wait} seconds."
                )
            )
        if inv_id not in self.run.ordered_investigations:
            self.run.ordered_investigations.append(inv_id)
        result = inv.result_by_state.get(self.run.current_state_id, inv.result)
        self._log("order", inv_id)
        step = self._after_action(inv.name)
        step.revealed = result
        return step

    def _schedule_deferred(self, inv: InvestigationDef, fire_at: int) -> None:
        """Queue a result to report back, unless one is already queued for this test."""
        if any(ev.get("inv_id") == inv.id for ev in self.run.deferred_events):
            return
        self.run.deferred_events.append(
            {"inv_id": inv.id, "fire_at": int(fire_at), "actor_id": self._result_holder(inv)}
        )

    def due_deferred(self) -> list[dict]:
        """Drain scheduled results whose time has come (the room reporting back).

        Each returns ``{actor_id, inv, revealed}`` so the caller can voice the report
        in the holder's voice and drop a card in the diagnostics panel. Fires once —
        the event is removed from the queue here."""
        out: list[dict] = []
        remaining: list[dict] = []
        for ev in self.run.deferred_events:
            if ev.get("fire_at", 0) > self.run.elapsed_seconds:
                remaining.append(ev)
                continue
            inv = self._investigation(ev.get("inv_id", ""))
            if inv is None:
                continue
            if inv.id not in self.run.ordered_investigations:
                self.run.ordered_investigations.append(inv.id)
            revealed = inv.result_by_state.get(self.run.current_state_id, inv.result)
            self._log("result", inv.id)
            out.append({"actor_id": ev.get("actor_id", "nurse"), "inv": inv, "revealed": revealed})
        self.run.deferred_events = remaining
        return out

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
            # Presence gate (C1): a cue that references an absent actor (e.g. "ask the
            # husband") must wait until they're in the room — don't mark it fired, so
            # it becomes eligible once they arrive rather than firing a lie now.
            if any(not self.is_present(aid) for aid in cue.requires_present):
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
        self.run.pending_action = None  # a new state is fresh context — drop any parked action
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
    SNAPSHOT_VERSION = 1

    # Migration registry: keys are the *from* version (int); values are callables
    # that accept the raw run dict and return the transformed dict.
    #
    # HOW TO ADD A MIGRATION:
    #   1. Add  _MIGRATIONS[N] = lambda r: {**r, "new_field": default, ...}
    #   2. Bump SNAPSHOT_VERSION to N + 1
    #
    # Migrations run in ascending key order, so v0→v2 automatically chains
    # v0→v1→v2.  The filter after migration keeps only known RunState fields, so
    # a migration can be additive without risking unknown-field errors.
    _MIGRATIONS: ClassVar[dict[int, Callable[[dict], dict]]] = {}

    def snapshot(self) -> dict[str, object]:
        return {"snapshot_version": self.SNAPSHOT_VERSION, "case_id": self.case.id, "run": self.run.__dict__}

    @classmethod
    def _apply_migrations(cls, run: dict, from_v: int) -> dict:
        """Chain migrations from ``from_v`` up to (but not including) SNAPSHOT_VERSION."""
        for ver in sorted(k for k in cls._MIGRATIONS if from_v <= k < cls.SNAPSHOT_VERSION):
            run = cls._MIGRATIONS[ver](run)
        return run

    def restore(self, snapshot: dict[str, object]) -> None:
        # Tolerate snapshots written by an older/newer RunState shape: keep only
        # fields this version knows, so a renamed/dropped field can't 500 every
        # parked resume. Missing fields fall back to their dataclass defaults.
        import logging as _logging
        _log = _logging.getLogger(__name__)
        v = snapshot.get("snapshot_version")
        from_v: int
        if v is None:
            _log.warning("restoring unversioned snapshot; treating as v0")
            from_v = 0
        else:
            from_v = int(v)
            if from_v != self.SNAPSHOT_VERSION:
                _log.warning("snapshot version %s != engine version %s; migrating",
                             from_v, self.SNAPSHOT_VERSION)
        run = dict(snapshot.get("run") or {})
        run = self._apply_migrations(run, from_v)
        known = {f.name for f in fields(RunState)}
        self.run = RunState(**{k: val for k, val in run.items() if k in known})  # type: ignore[arg-type]
