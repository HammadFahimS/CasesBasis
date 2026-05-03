from __future__ import annotations

from backend import manual_demo


class FailingSimulator:
    def generate_case(self, *_: object, **__: object) -> object:
        raise RuntimeError("case generation failed after retries")


class UnusedService:
    pass


def test_manual_demo_returns_clean_error_when_generation_fails(
    monkeypatch: object,
    capsys: object,
) -> None:
    monkeypatch.setattr(manual_demo, "ClinicalCaseSimulator", lambda: FailingSimulator())
    monkeypatch.setattr(manual_demo, "PatientResponseService", lambda: UnusedService())
    monkeypatch.setattr(manual_demo, "InvestigationService", lambda: UnusedService())
    monkeypatch.setattr(manual_demo, "ClinicalEvaluator", lambda: UnusedService())

    exit_code = manual_demo.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Manual demo failed: case generation failed after retries" in captured.err
