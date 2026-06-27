from __future__ import annotations

from backend.actors import kind_for_word, speaker_voice


def test_kind_for_word_resolves_synonyms():
    assert kind_for_word("paramedic").id == "ems"
    assert kind_for_word("husband").id == "family"
    assert kind_for_word("ma'am").id == "patient"
    assert kind_for_word("respiratory").id == "rt"
    assert kind_for_word("consultant").id == "consult"
    assert kind_for_word("nurse").id == "nurse"
    assert kind_for_word("lab") is None
    assert kind_for_word(None) is None


def test_holds_history_flags_only_bystanders_who_carry_a_beat():
    # the people the resident can take history from
    assert kind_for_word("ems").holds_history
    assert kind_for_word("husband").holds_history
    assert kind_for_word("patient").holds_history
    # staff / consultants don't hold history beats
    assert not kind_for_word("nurse").holds_history
    assert not kind_for_word("respiratory").holds_history
    assert not kind_for_word("consultant").holds_history


def test_speaker_voice_reads_together_env_with_default(monkeypatch):
    """Together/Cartesia path: env var overrides the registry default."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_EMS_TTS_VOICE", raising=False)
    assert speaker_voice("ems") == "newsman"            # registry default
    monkeypatch.setenv("TOGETHER_EMS_TTS_VOICE", "storyteller")
    assert speaker_voice("ems") == "storyteller"        # env override wins
    assert speaker_voice("nurse") is None               # nurse → default voice in voice_service
    assert speaker_voice("lab") is None


def test_speaker_voice_returns_el_id_when_key_set(monkeypatch):
    """ElevenLabs path: voice() returns a voice_id instead of a Cartesia name."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.delenv("ELEVENLABS_EMS_VOICE_ID", raising=False)
    assert speaker_voice("ems") == "TxGEqnHWrfWFTfGW9XjX"   # EL default for ems (Josh)
    monkeypatch.setenv("ELEVENLABS_EMS_VOICE_ID", "custom_el_id_abc123xyz")
    assert speaker_voice("ems") == "custom_el_id_abc123xyz"
    # nurse still returns None when EL is active — its default lives in voice_service
    assert speaker_voice("nurse") is None
    assert speaker_voice("lab") is None


def test_el_settings_for_voice_returns_per_actor_settings(monkeypatch):
    """el_settings_for_voice looks up the registered stability/similarity by voice_id."""
    from backend.actors import SPEAKERS, el_settings_for_voice
    monkeypatch.delenv("ELEVENLABS_PATIENT_VOICE_ID", raising=False)
    patient = next(s for s in SPEAKERS if s.id == "patient")
    stability, similarity = el_settings_for_voice(patient.el_default_voice)
    assert stability == patient.el_stability
    assert similarity == patient.el_similarity


def test_el_settings_for_voice_returns_defaults_for_unknown(monkeypatch):
    from backend.actors import el_settings_for_voice
    stability, similarity = el_settings_for_voice("unknown_voice_id_xyz")
    assert stability == 0.5
    assert similarity == 0.75
