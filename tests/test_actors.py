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


def test_speaker_voice_reads_env_with_default(monkeypatch):
    monkeypatch.delenv("TOGETHER_EMS_TTS_VOICE", raising=False)
    assert speaker_voice("ems") == "newsman"            # registry default
    monkeypatch.setenv("TOGETHER_EMS_TTS_VOICE", "storyteller")
    assert speaker_voice("ems") == "storyteller"        # env override wins
    assert speaker_voice("nurse") is None               # nurse → default voice in voice_service
    assert speaker_voice("lab") is None
