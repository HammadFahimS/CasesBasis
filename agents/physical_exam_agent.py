from __future__ import annotations

import re

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from schemas.clinical_case import ClinicalCase
from schemas.physical_exam import ExamFinding, PhysicalExamResponse


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
        return [_redact_finding(f, case.diagnosis) for f in result.results]


def _format_ground_truth(systems: list[str], exam_findings: dict[str, list[str]]) -> str:
    lines = [
        f"{system}: {', '.join(findings)}"
        for system in systems
        if (findings := exam_findings.get(system))
    ]
    return "\n".join(lines) if lines else "None"


def _redact_finding(finding: ExamFinding, diagnosis: str) -> ExamFinding:
    """Scrub any accidental diagnosis mention from finding text."""
    if not diagnosis.strip():
        return finding
    pattern = re.compile(re.escape(diagnosis.strip()), re.IGNORECASE)
    redacted_findings = [pattern.sub("the condition", text) for text in finding.findings]
    redacted_summary = pattern.sub("the condition", finding.summary)
    return ExamFinding(
        system=finding.system,
        findings=redacted_findings,
        summary=redacted_summary,
        abnormal=finding.abnormal,
    )
