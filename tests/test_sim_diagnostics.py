from __future__ import annotations

from backend import sim_diagnostics
from schemas.investigation import CXRVisualPayload, ECGVisualPayload, LeadIIMorphology
from schemas.simulated_case import InvestigationDef, PatientProfile, VitalSigns


class FakeECG:
    def build_visual_payload(self, dx, *, heart_rate, age, sex):
        self.called_with = (dx, heart_rate, age, sex)
        return ECGVisualPayload(type="svg", leads=["II"], svg="<svg id='ecg'/>")


class FakeCXR:
    def build_visual_payload(self, dx, *, age, gender, race):
        return CXRVisualPayload(type="image", format="jpeg", data="QUJD", finding="No Finding")


def _patient():
    return PatientProfile(age=29, gender="female")


def _vitals():
    return VitalSigns(heart_rate=105)


def test_ecg_card_attaches_real_visual(monkeypatch):
    fake = FakeECG()
    monkeypatch.setattr(sim_diagnostics, "_ecg", lambda: fake)
    inv = InvestigationDef(id="ecg", name="ECG", category="ecg",
                           result="Sinus tach", visual_dx="sinus tachycardia")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "Sinus tach")
    assert card["kind"] == "ecg"
    assert card["visual_payload"]["svg"] == "<svg id='ecg'/>"
    assert card["summary"] == "Sinus tach"          # authored text kept as the report
    assert fake.called_with == ("sinus tachycardia", 105, 29, "female")


def test_ecg_card_carries_lead_ii_morphology_for_live_monitor(monkeypatch):
    # The Voice Room's live lead-II monitor reads visual_payload.lead_ii_morphology;
    # guard that it survives into the card so a schema change can't silently break it.
    morph = LeadIIMorphology(hr=140, pr_ms=120, qrs_ms=140, qt_ms=360,
                             p_absent=True, qrs="wide", t_inv=True,
                             rhythm_label="ATRIAL FIBRILLATION")

    class MorphECG:
        def build_visual_payload(self, dx, *, heart_rate, age, sex):
            return ECGVisualPayload(type="svg", leads=["II"], svg="<svg/>",
                                    lead_ii_morphology=morph)

    monkeypatch.setattr(sim_diagnostics, "_ecg", lambda: MorphECG())
    inv = InvestigationDef(id="ecg", name="ECG", category="ecg", result="AF")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "AF, rate 140")
    m = card["visual_payload"]["lead_ii_morphology"]
    assert m["rhythm_label"] == "ATRIAL FIBRILLATION"
    assert m["p_absent"] is True and m["qrs"] == "wide" and m["t_inv"] is True
    assert m["hr"] == 140


def test_cxr_card_attaches_image(monkeypatch):
    monkeypatch.setattr(sim_diagnostics, "_cxr", lambda: FakeCXR())
    inv = InvestigationDef(id="cxr", name="Chest X-ray", category="imaging")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "No acute findings")
    assert card["kind"] == "cxr"
    assert card["visual_payload"]["data"] == "QUJD"


def test_lab_is_plain_report():
    inv = InvestigationDef(id="cbc", name="CBC", category="lab")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "All normal")
    assert card["kind"] == "report"
    assert card["visual_payload"] is None
    assert card["summary"] == "All normal"


def test_ct_imaging_is_report_not_cxr():
    # CT is imaging but not a chest film — must not hit the CXR generator.
    inv = InvestigationDef(id="ct_chest", name="CT chest (with contrast)", category="imaging")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "Mediastinal mass")
    assert card["kind"] == "report"


def test_generator_failure_degrades_to_report(monkeypatch):
    class Boom:
        def build_visual_payload(self, *a, **k):
            raise RuntimeError("generator down")

    monkeypatch.setattr(sim_diagnostics, "_ecg", lambda: Boom())
    inv = InvestigationDef(id="ecg", name="ECG", category="ecg", result="Sinus tach")
    card = sim_diagnostics.build_investigation_card(inv, _vitals(), _patient(), "Sinus tach")
    assert card["kind"] == "report"          # never breaks the turn
    assert card["summary"] == "Sinus tach"


def test_exam_card():
    card = sim_diagnostics.build_exam_card("Neurological", "Neurological: bilateral ptosis")
    assert card["kind"] == "exam"
    assert "ptosis" in card["summary"]
