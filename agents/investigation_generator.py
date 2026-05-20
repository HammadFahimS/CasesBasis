from __future__ import annotations

import json
import re

from agents.base import OpenAIJsonAgent
from agents.prompt_loader import load, render
from schemas.clinical_case import ClinicalCase
from schemas.investigation import InvestigationResult

MAX_INVESTIGATION_CHARS = 120
MAX_INVESTIGATION_ATTEMPTS = 3


# ── Numerical consistency helpers ──────────────────────────────────────────────

def _parse_range(ref_range: str) -> tuple[float | None, float | None]:
    """Parse '0–0.04', '3.5-5.0', '<0.04', '>1.5' into (lower, upper)."""
    # Normalise unicode dashes to ASCII hyphen
    s = re.sub(r'[–—−]', '-', str(ref_range)).strip()
    lt = re.match(r'^[<≤]\s*([\d.]+)', s)
    gt = re.match(r'^[>≥]\s*([\d.]+)', s)
    if lt:
        return None, float(lt.group(1))
    if gt:
        return float(gt.group(1)), None
    # Match "number - number" where the dash is a range separator, not a negative sign.
    # The dash must be preceded by a digit/dot and followed by a digit.
    m = re.search(r'([\d.]+)\s*-\s*([\d.]+)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _parse_value(val: object) -> float | None:
    """Extract the first numeric value from a string or number."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        m = re.search(r'-?\d+\.?\d*', str(val))
        return float(m.group()) if m else None


def check_flag_consistency(values: list[dict]) -> list[str]:
    """
    Return a list of inconsistency messages for any value whose flag
    direction contradicts its numeric value vs reference range.
    """
    issues: list[str] = []
    for v in values:
        flag = str(v.get("flag", "normal")).lower()
        if flag in ("normal", "non-contributory"):
            continue
        numeric = _parse_value(v.get("value"))
        if numeric is None:
            continue
        lo, hi = _parse_range(str(v.get("reference_range", "")))
        name = v.get("name", "?")
        if flag in ("high", "mildly high") and hi is not None and numeric <= hi:
            issues.append(
                f"{name}: flagged '{flag}' but {numeric} is not above reference upper bound {hi}"
            )
        elif flag in ("low", "mildly low") and lo is not None and numeric >= lo:
            issues.append(
                f"{name}: flagged '{flag}' but {numeric} is not below reference lower bound {lo}"
            )
    return issues


class InvestigationGenerationError(RuntimeError):
    """Raised when investigation generation cannot produce valid results after retries."""
    pass


def sanitize_requested_investigation(requested_investigation: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", requested_investigation)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized[:MAX_INVESTIGATION_CHARS].strip()
    if not sanitized:
        raise ValueError("requested investigation is required")
    return sanitized


def build_reasoning_prompt(case: ClinicalCase, requested_investigation: str) -> str:
    return render(
        load("investigation", "reasoning_user"),
        diagnosis=case.diagnosis,
        age=str(case.patient.age),
        gender=case.patient.gender,
        symptoms=", ".join(case.symptoms),
        heart_rate=str(case.vitals.heart_rate),
        blood_pressure=case.vitals.blood_pressure,
        respiratory_rate=str(case.vitals.respiratory_rate),
        oxygen_saturation=str(case.vitals.oxygen_saturation),
        risk_factors=", ".join(case.risk_factors) if case.risk_factors else "none",
        progression="; ".join(e.description for e in case.progression),
        requested_investigation=requested_investigation,
    )


def build_formatter_prompt(
    case: ClinicalCase,
    requested_investigation: str,
    reasoning: str,
    previous_issues: list[str] | None = None,
) -> str:
    if previous_issues:
        previous_issues_block = (
            "PREVIOUS ATTEMPT FAILED — fix these issues:\n"
            + "\n".join(f"- {issue}" for issue in previous_issues)
            + "\n\n"
        )
    else:
        previous_issues_block = ""
    return render(
        load("investigation", "formatter_user"),
        reasoning=reasoning,
        requested_investigation=requested_investigation,
        case_json=case.model_dump_json(indent=2),
        previous_issues_block=previous_issues_block,
    )


def result_reveals_diagnosis(result: InvestigationResult, case: ClinicalCase) -> bool:
    diagnosis = case.diagnosis.strip().lower()
    if len(diagnosis) < 4:
        return False
    return diagnosis in result.model_dump_json().lower()


def validate_generic_investigation_result(
    result: InvestigationResult,
    case: ClinicalCase,
) -> tuple[InvestigationResult, list[str]]:
    """
    Validate the result. Returns (result, issues) where issues is a list
    of numerical consistency problems (empty = all good).
    Raises ValueError for hard failures (wrong status, reveals diagnosis, etc.).
    """
    if result.status != "available":
        raise ValueError("generic investigation results must be available")
    if result.visual_payload is not None:
        raise ValueError("generic investigation results must not include a visual payload")
    if result_reveals_diagnosis(result, case):
        raise ValueError("investigation result directly reveals the hidden diagnosis")
    if not result.investigation.strip():
        raise ValueError("investigation name is required")
    if not result.summary.strip():
        raise ValueError("investigation summary is required")
    if not result.structured_result:
        raise ValueError("structured_result is required")

    # Numerical flag consistency check
    values = result.structured_result.get("values", [])
    issues = check_flag_consistency(values) if isinstance(values, list) else []
    return result, issues


class GenericInvestigationAgent(OpenAIJsonAgent):
    """
    Three-stage pipeline for investigation result generation.

    Stage 1 (reasoning): Senior clinician reasons through what the test would
    show for this diagnosis — plain text, no JSON pressure.

    Stage 2 (formatter): Translates reasoning into structured JSON with specific
    numeric values, flags, and clinical hints.  Retried with explicit feedback
    if numerical flag/range consistency fails.

    Stage 3 (pattern validation): Lightweight LLM audit that checks the result's
    flag directions match what the reasoning stated.  Triggers a re-format if
    directional errors are found.
    """

    def generate(
        self,
        case: ClinicalCase,
        requested_investigation: str,
    ) -> InvestigationResult:
        reasoning = self._reason(case, requested_investigation)
        errors: list[Exception] = []
        previous_issues: list[str] = []

        for attempt in range(MAX_INVESTIGATION_ATTEMPTS):
            try:
                formatter_prompt = build_formatter_prompt(
                    case, requested_investigation, reasoning,
                    previous_issues if attempt > 0 else None,
                )
                result = self.parse_response(
                    system_prompt=load("investigation", "formatter_system"),
                    user_prompt=formatter_prompt,
                    response_model=InvestigationResult,
                )
                result, num_issues = validate_generic_investigation_result(result, case)

                if num_issues:
                    # Feed issues back to formatter on next attempt
                    previous_issues = num_issues
                    raise ValueError(
                        "Numerical flag inconsistency: " + "; ".join(num_issues)
                    )

                # Pattern validation: check result directions match reasoning
                pattern_issues = self._validate_pattern(
                    case, requested_investigation, reasoning, result
                )
                if pattern_issues:
                    previous_issues = pattern_issues
                    raise ValueError(
                        "Pattern mismatch: " + "; ".join(pattern_issues)
                    )

                return result

            except Exception as exc:
                errors.append(exc)

        raise InvestigationGenerationError(
            f"investigation generation failed after {MAX_INVESTIGATION_ATTEMPTS} attempts: "
            f"{errors[-1]}"
        ) from errors[-1]

    def _reason(self, case: ClinicalCase, requested_investigation: str) -> str:
        return self.text_response(
            system_prompt=load("investigation", "reasoning_system"),
            user_prompt=build_reasoning_prompt(case, requested_investigation),
        )

    def _validate_pattern(
        self,
        case: ClinicalCase,
        requested_investigation: str,
        reasoning: str,
        result: InvestigationResult,
    ) -> list[str]:
        """
        Ask the LLM whether the result's flag directions match the reasoning.
        Returns a list of issues (empty = valid).
        """
        values = result.structured_result.get("values", [])
        if not isinstance(values, list) or not values:
            return []  # no numeric values to validate (imaging, micro, etc.)

        # Only include flagged values to keep the prompt short
        flagged = [
            v for v in values
            if str(v.get("flag", "normal")).lower() not in ("normal", "non-contributory")
        ]
        if not flagged:
            return []

        user_prompt = render(
            load("investigation", "pattern_user"),
            diagnosis=case.diagnosis,
            investigation=requested_investigation,
            reasoning=reasoning,
            summary=result.summary,
            values_json=json.dumps(flagged, indent=2),
        )
        try:
            raw = self.text_response(
                system_prompt=load("investigation", "pattern_system"),
                user_prompt=user_prompt,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            if parsed.get("valid"):
                return []
            return parsed.get("issues", ["pattern validation failed"])
        except Exception:
            return []  # validation call failed; don't block result
