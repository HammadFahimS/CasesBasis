from __future__ import annotations

from backend.medication import Med, needs_dose_confirm, parse_med


# ── parse_med: drug / dose / high-alert ──────────────────────────────────────
def test_parse_med_splits_drug_and_dose_from_action_text():
    med = parse_med("rocuronium", "0.6 mg/kg IV")
    assert med.drug == "rocuronium"
    assert med.dose == "0.6 mg/kg"
    assert med.high_alert is True


def test_parse_med_reads_spelled_out_per_kilo_dose():
    med = parse_med("succinylcholine", "give 1.5 milligrams per kilo")
    assert med.dose == "1.5 milligrams per kilo"
    assert med.high_alert is True


def test_parse_med_without_target_pulls_drug_from_paraphrase():
    med = parse_med(None, "push 100 mcg of fentanyl")
    # the dose token is stripped out of the drug name
    assert "fentanyl" in med.drug.lower()
    assert med.dose == "100 mcg"


def test_parse_med_no_dose_leaves_dose_none():
    med = parse_med("rocuronium", "")
    assert med.dose is None
    assert med.high_alert is True


def test_parse_med_non_high_alert_drug():
    med = parse_med("ondansetron", "4 mg IV")
    assert med.high_alert is False
    assert med.dose == "4 mg"


def test_med_slug_is_stable_and_card_safe():
    assert parse_med("Succinylcholine", "").slug == "med-succinylcholine"
    # two orders of the same drug share a slug, so the card upserts
    assert parse_med("rocuronium", "0.6 mg/kg").slug == parse_med("rocuronium", "").slug


# ── needs_dose_confirm ───────────────────────────────────────────────────────
def test_confirm_required_for_paralytic_only():
    assert needs_dose_confirm(parse_med("rocuronium", "")) is True
    assert needs_dose_confirm(parse_med("normal saline", "1 L")) is False
