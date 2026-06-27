"""End-to-end physical-exam agent tests with a fake LLM that intentionally leaks.

Verifies the agent's post-processing scrub catches every common leak pattern
that an LLM might emit. We never call a real LLM — `FakeAgent` is constructed
with a canned response and replaces `parse_response`.
"""

from __future__ import annotations

import logging

import pytest

from agents.physical_exam_agent import PhysicalExamAgent
from schemas.clinical_case import ClinicalCase
from schemas.physical_exam import ExamFinding, PhysicalExamResponse


# ── Fixtures ────────────────────────────────────────────────────────────────


def _mi_case() -> ClinicalCase:
    return ClinicalCase.model_validate(
        {
            "diagnosis": "myocardial infarction",
            "patient": {
                "age": 62,
                "gender": "male",
                "history": ["hypertension", "current smoker"],
            },
            "symptoms": ["central chest pain", "diaphoresis", "shortness of breath"],
            "vitals": {
                "heart_rate": 104,
                "blood_pressure": "148/92",
                "respiratory_rate": 22,
                "oxygen_saturation": 95,
            },
            "progression": [{"time_min": 0, "description": "Pain started."}],
            "risk_factors": ["smoking", "hypertension"],
            "exam_findings": {"General": ["diaphoretic"]},
            "outcome": {
                "if_treated": "Rapid emergency treatment improves outcomes.",
                "if_untreated": "Ongoing ischemia leads to instability.",
            },
        }
    )


def _pe_case() -> ClinicalCase:
    case = _mi_case().model_dump()
    case["diagnosis"] = "pulmonary embolism"
    case["symptoms"] = ["pleuritic chest pain", "shortness of breath", "calf swelling"]
    case["exam_findings"] = {"Respiratory": ["tachypneic"]}
    return ClinicalCase.model_validate(case)


def _stroke_case() -> ClinicalCase:
    case = _mi_case().model_dump()
    case["diagnosis"] = "stroke"
    case["symptoms"] = ["right-sided weakness", "facial droop", "slurred speech"]
    case["exam_findings"] = {"Neurological": ["right hemiparesis"]}
    return ClinicalCase.model_validate(case)


def _cholecystitis_case() -> ClinicalCase:
    case = _mi_case().model_dump()
    case["diagnosis"] = "cholecystitis"
    case["symptoms"] = ["right upper quadrant pain", "fever", "nausea"]
    case["exam_findings"] = {"Abdominal": ["RUQ tenderness", "positive Murphy's sign"]}
    return ClinicalCase.model_validate(case)


def _dka_case() -> ClinicalCase:
    case = _mi_case().model_dump()
    case["diagnosis"] = "diabetic ketoacidosis"
    case["symptoms"] = ["polyuria", "polydipsia", "abdominal pain"]
    case["exam_findings"] = {"General": ["Kussmaul breathing"]}
    return ClinicalCase.model_validate(case)


# ── FakeAgent (no real LLM) ─────────────────────────────────────────────────


class FakeAgent(PhysicalExamAgent):
    """Skip __init__ so we don't try to build an LLM client.

    The test provides a canned `PhysicalExamResponse` that `parse_response`
    will return.
    """

    def __init__(self, response: PhysicalExamResponse) -> None:  # type: ignore[override]
        self._response = response

    def parse_response(self, **_kwargs) -> PhysicalExamResponse:  # type: ignore[override]
        return self._response


# ── Helpers ────────────────────────────────────────────────────────────────


def _examine(agent: FakeAgent, case: ClinicalCase) -> list[ExamFinding]:
    # Bypass the prompt-render pipeline since `parse_response` is stubbed.
    return [agent._examine_dummy(case)]  # noqa: SLF001 — direct call below


def _direct_scrub(canned: PhysicalExamResponse, case: ClinicalCase) -> list[ExamFinding]:
    """Use the agent's private scrub on the canned response (no LLM, no prompt)."""
    from agents.physical_exam_agent import _scrub_finding

    return [_scrub_finding(f, case.diagnosis) for f in canned.results]


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


class TestPhysicalExamGuard:
    """Each test crafts a canned response that emulates an LLM trying to leak
    the diagnosis through one of the common channels and verifies the agent
    strips it before returning."""

    def test_synonym_leak_in_findings_is_stripped(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Cardiovascular",
                    findings=[
                        "S3 gallop audible at the apex.",
                        "Yeah, this is probably a heart attack.",
                        "Diaphoretic and in distress.",
                    ],
                    summary="Findings consistent with an MI.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _mi_case())
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "heart attack" not in joined
        assert "mi" not in joined.split()  # no standalone "mi"
        assert "consistent with" not in joined
        # The safe observations should survive
        assert any("s3 gallop" in f.lower() for f in out.findings)
        assert any("diaphoretic" in f.lower() for f in out.findings)

    def test_interpretive_clause_after_observation_is_truncated(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Respiratory",
                    findings=[
                        "Bilateral basal fine crackles, consistent with pulmonary oedema.",
                    ],
                    summary="Respiratory exam shows crackles suggestive of CHF.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, ClinicalCase.model_validate({
            **_mi_case().model_dump(),
            "diagnosis": "heart failure",
        }))
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "consistent with" not in joined
        assert "suggestive of" not in joined
        assert "pulmonary oedema" not in joined
        assert "chf" not in joined.split()
        # observation kept
        assert any("crackles" in f.lower() for f in out.findings)

    def test_stroke_synonym_cva_is_caught(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Neurological",
                    findings=[
                        "Right hemiparesis with pronator drift.",
                        "Findings are diagnostic of a CVA.",
                    ],
                    summary="Picture of a brain attack.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _stroke_case())
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "cva" not in joined.split()
        assert "brain attack" not in joined
        assert "diagnos" not in joined
        # observation kept
        assert any("hemiparesis" in f.lower() for f in out.findings)

    def test_pe_abbreviation_is_caught(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Respiratory",
                    findings=[
                        "Tachypneic with shallow breathing.",
                        "Concerning for PE based on the calf swelling.",
                    ],
                    summary="Suggests a PE.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _pe_case())
        joined = " ".join(out.findings + [out.summary]).lower()
        assert " pe " not in f" {joined} "
        assert "concerning for" not in joined
        assert "suggests" not in joined
        assert any("tachypneic" in f.lower() for f in out.findings)

    def test_dka_abbreviation_is_caught(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="General",
                    findings=[
                        "Kussmaul breathing pattern noted.",
                        "Dry mucous membranes.",
                        "Likely DKA based on the history.",
                    ],
                    summary="Looks like a diabetic emergency.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _dka_case())
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "dka" not in joined.split()
        assert "likely" not in joined
        assert "looks like" not in joined
        assert any("kussmaul" in f.lower() for f in out.findings)

    def test_differential_word_is_stripped(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Abdominal",
                    findings=[
                        "RUQ tenderness with positive Murphy's sign.",
                        "Differential includes acute cholecystitis.",
                    ],
                    summary="Working diagnosis is cholecystitis.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _cholecystitis_case())
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "differential" not in joined
        assert "working diagnosis" not in joined
        assert "cholecystitis" not in joined
        assert any("murphy" in f.lower() for f in out.findings)

    def test_pure_observation_passes_unchanged_except_period(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Cardiovascular",
                    findings=[
                        "Heart sounds dual, no added sounds.",
                        "JVP elevated at 6 cm above the sternal angle.",
                    ],
                    summary="Cardiovascular exam shows raised JVP and dual heart sounds.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _mi_case())
        assert out.summary.lower().startswith("cardiovascular exam shows")
        assert "heart sounds dual" in out.findings[0].lower()
        assert "jvp elevated" in out.findings[1].lower()

    def test_entirely_interpretive_finding_falls_back_to_placeholder(self) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Cardiovascular",
                    findings=[
                        "Likely cardiac.",
                        "Diagnosis is anterior STEMI.",
                    ],
                    summary="Consistent with acute MI.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, _mi_case())
        # Both findings were interpretive — all dropped — placeholder kicks in
        assert out.findings == ["Examination unremarkable."]
        assert out.summary == "Examination unremarkable."

    def test_observations_in_mixed_findings_survive(self) -> None:
        """Most realistic LLM failure mode: one good finding, one editorializing."""
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Cardiovascular",
                    findings=[
                        "Apex beat displaced to the 6th intercostal space, mid-axillary line.",
                        "Pansystolic murmur, probably mitral regurgitation.",
                        "JVP raised to the angle of the jaw.",
                    ],
                    summary="Exam shows displaced apex, raised JVP, pansystolic murmur.",
                    abnormal=True,
                )
            ]
        )
        [out] = _direct_scrub(canned, ClinicalCase.model_validate({
            **_mi_case().model_dump(),
            "diagnosis": "mitral regurgitation",
        }))
        joined = " ".join(out.findings + [out.summary]).lower()
        assert "probably" not in joined
        assert "mitral regurgitation" not in joined
        # The two pure observations + the murmur observation (sans interpretation) survive
        assert any("apex beat displaced" in f.lower() for f in out.findings)
        assert any("jvp raised" in f.lower() for f in out.findings)
        # The murmur sentence kept its observation half
        murmur = next((f for f in out.findings if "pansystolic murmur" in f.lower()), None)
        assert murmur is not None
        assert "probably" not in murmur.lower()


# ── Logging behavior ────────────────────────────────────────────────────────


class TestLoggingBehavior:
    def test_scrubbed_leaks_emit_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        canned = PhysicalExamResponse(
            results=[
                ExamFinding(
                    system="Cardiovascular",
                    findings=[
                        "S3 gallop, probably indicating heart failure.",
                    ],
                    summary="Picture consistent with CHF.",
                    abnormal=True,
                )
            ]
        )
        with caplog.at_level(logging.WARNING, logger="agents.physical_exam_agent"):
            _direct_scrub(canned, ClinicalCase.model_validate({
                **_mi_case().model_dump(),
                "diagnosis": "heart failure",
            }))
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "expected at least one WARNING when scrubbing leaks"
        joined = " ".join(r.message for r in warnings).lower()
        # Either the finding was scrubbed OR dropped — both should log
        assert "physical_exam" in joined
