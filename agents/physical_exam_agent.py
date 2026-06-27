from __future__ import annotations

import logging

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from backend.diagnosis_guard import contains_diagnosis_leak, scrub_finding_text
from schemas.clinical_case import ClinicalCase
from schemas.physical_exam import ExamFinding, PhysicalExamResponse

logger = logging.getLogger(__name__)


class PhysicalExamAgent(OpenAIJsonAgent):
    def examine(
        self,
        case: ClinicalCase,
        systems: list[str],
        already_examined: list[str] | None = None,
    ) -> list[ExamFinding]:
        prompt = render(
            load("physical_exam", "user"),
            diagnosis=case.diagnosis,
            age=str(case.patient.age),
            gender=case.patient.gender,
            symptoms=", ".join(case.symptoms),
            heart_rate=str(case.vitals.heart_rate),
            blood_pressure=case.vitals.blood_pressure,
            respiratory_rate=str(case.vitals.respiratory_rate),
            oxygen_saturation=str(case.vitals.oxygen_saturation),
            history="; ".join(case.patient.history),
            risk_factors=", ".join(case.risk_factors) if case.risk_factors else "none",
            ground_truth_block=_format_ground_truth(systems, case.exam_findings),
            systems=", ".join(systems),
        )
        result = self.parse_response(
            system_prompt=load("physical_exam", "system"),
            user_prompt=prompt,
            response_model=PhysicalExamResponse,
        )
        return [_scrub_finding(f, case.diagnosis) for f in result.results]


def _format_ground_truth(systems: list[str], exam_findings: dict[str, list[str]]) -> str:
    lines = [
        f"{system}: {', '.join(findings)}"
        for system in systems
        if (findings := exam_findings.get(system))
    ]
    return "\n".join(lines) if lines else "None"


def _scrub_finding(finding: ExamFinding, diagnosis: str) -> ExamFinding:
    """Run every observable text field through the central guard pipeline.

    Substitutes diagnosis aliases and drops interpretive clauses ("consistent
    with...", "likely cardiac", etc.). Any sentence the LLM used to
    editorialize gets removed; the safe observation portion is kept.
    """
    cleaned_findings: list[str] = []
    for raw in finding.findings:
        scrubbed = scrub_finding_text(raw, diagnosis)
        if not scrubbed:
            logger.warning(
                "physical_exam: dropped finding (entirely interpretive): %r",
                raw,
            )
            continue
        if scrubbed != raw and contains_diagnosis_leak(raw, diagnosis):
            logger.warning(
                "physical_exam: scrubbed leak in finding: %r → %r",
                raw, scrubbed,
            )
        cleaned_findings.append(scrubbed)

    summary = scrub_finding_text(finding.summary, diagnosis)
    if summary != finding.summary and contains_diagnosis_leak(finding.summary, diagnosis):
        logger.warning(
            "physical_exam: scrubbed leak in summary: %r → %r",
            finding.summary, summary,
        )

    # If everything got stripped, fall back to a neutral placeholder so the
    # schema's required string fields don't end up empty.
    if not summary:
        summary = "Examination unremarkable."
    if not cleaned_findings:
        cleaned_findings = ["Examination unremarkable."]

    return ExamFinding(
        system=finding.system,
        findings=cleaned_findings,
        summary=summary,
        abnormal=finding.abnormal,
    )
