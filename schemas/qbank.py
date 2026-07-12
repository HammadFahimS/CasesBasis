"""Schema for the *question bank* — UWorld-style practice cases.

A question-bank case is a clinical vignette (``stem``) followed by one or more
ordered ``checkpoints``. Each checkpoint is a single-best-answer question with
exactly one correct choice and a written explanation for *every* choice (the
full-explanation reveal is the core of the UWorld experience), plus a
``key_learning_point`` and teaching metadata.

This is deliberately separate from :class:`schemas.simulated_case.SimulatedCase`
(the live state-machine encounter). A qbank case is static assessment content:
it is validated on load, seeded into the ``qbank_cases`` table, and served
read-only. Only per-resident *attempts* are mutable (see
``backend.qbank_persistence``).

Integration seams left for later passes:
* ``Checkpoint.voice_suitable`` — flags checkpoints that could be delivered via
  the existing voice pipeline. Exposed through the model + API; no delivery yet.
* ``Checkpoint.triggers_remediation`` — flags checkpoints where a wrong answer
  should offer a live simulator hand-off (see the qbank front-end + router).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class Choice(BaseModel):
    """One answer option. Exactly one choice per checkpoint has ``correct=True``;
    every choice carries an explanation so the reveal can teach the distractors."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    correct: bool = False
    explanation: str = ""


class Tags(BaseModel):
    """Filterable classification used by the session builder."""

    model_config = ConfigDict(extra="forbid")

    subject: str = "General"
    subtopic: str = ""
    system: str = "General"
    exam: str = ""
    difficulty: int = Field(default=3, ge=1, le=5)
    high_yield: bool = False


class Checkpoint(BaseModel):
    """A single-best-answer question within a case, presented in ``sequence`` order."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(min_length=1)
    sequence: int = Field(default=1, ge=1)
    question: str = Field(min_length=1)
    choices: list[Choice] = Field(min_length=2)
    key_learning_point: str = ""
    skill_tested: str = ""
    voice_suitable: bool = False
    triggers_remediation: bool = False

    @model_validator(mode="after")
    def _exactly_one_correct(self) -> "Checkpoint":
        n = sum(1 for c in self.choices if c.correct)
        if n != 1:
            raise ValueError(
                f"checkpoint {self.checkpoint_id!r} must have exactly one correct "
                f"choice, found {n}"
            )
        return self


class QBankCase(BaseModel):
    """A vignette plus its ordered checkpoints."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    checkpoints: list[Checkpoint] = Field(min_length=1)
    tags: Tags = Field(default_factory=Tags)

    @model_validator(mode="after")
    def _sorted_unique_checkpoints(self) -> "QBankCase":
        ids = [c.checkpoint_id for c in self.checkpoints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"case {self.case_id!r} has duplicate checkpoint ids")
        # Present in the authored sequence order (stable on ties).
        self.checkpoints.sort(key=lambda c: c.sequence)
        return self
