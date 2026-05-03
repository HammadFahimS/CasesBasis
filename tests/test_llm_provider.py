from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

import pytest

from agents.base import (
    OllamaClient,
    OpenAIJsonAgent,
    build_llm_client,
    get_llm_provider,
    safe_parse,
)
from agents.generator import GeneratorAgent
from agents.investigation_generator import GenericInvestigationAgent
from agents.patient import PatientResponseAgent
from agents.reasoning_reviewer import ReasoningReviewAgent
from agents.validator import ValidatorAgent
from backend.service import ClinicalCaseSimulator
from schemas.clinical_case import ClinicalCase
from tests.test_clinical_case_schema import sample_case


class SampleStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: int


class FakeOpenAIResponse:
    def __init__(self, output_parsed: Any) -> None:
        self.output_parsed = output_parsed


class FakeOpenAIResponses:
    def __init__(self, output_parsed: Any) -> None:
        self.output_parsed = output_parsed
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return FakeOpenAIResponse(self.output_parsed)


class FakeOpenAIClient:
    def __init__(self, output_parsed: Any) -> None:
        self.responses = FakeOpenAIResponses(output_parsed)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_json_response_uses_schema_and_validates(monkeypatch: object) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(http_request: Any, timeout: int) -> FakeHTTPResponse:
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"name": "heart rate", "value": 104})
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.base.request.urlopen", fake_urlopen)

    result = OllamaClient(base_url="http://localhost:11434/v1").parse_response(
        model="llama-med",
        system_prompt="Return JSON only.",
        user_prompt="Generate structured data.",
        response_model=SampleStructuredResponse,
    )

    assert result == SampleStructuredResponse(name="heart rate", value=104)
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["timeout"] == 120
    assert captured["payload"]["model"] == "llama-med"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert (
        captured["payload"]["response_format"]["json_schema"]["schema"]["properties"]["name"]["type"]
        == "string"
    )


def test_default_provider_is_ollama_when_env_var_not_set(monkeypatch: object) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert get_llm_provider() == "ollama"


def test_openai_provider_still_works_when_explicit(monkeypatch: object) -> None:
    fake_client = object()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr("agents.base.build_openai_client", lambda: fake_client)

    assert get_llm_provider() == "openai"
    assert build_llm_client() is fake_client


def test_default_ollama_client_does_not_require_openai_api_key(monkeypatch: object) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = build_llm_client()

    assert isinstance(client, OllamaClient)


def test_ollama_unreachable_error_is_clear(monkeypatch: object) -> None:
    from urllib.error import URLError

    def fake_urlopen(*_: object, **__: object) -> object:
        raise URLError("connection refused")

    monkeypatch.setattr("agents.base.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Ollama server is not reachable"):
        OllamaClient(base_url="http://localhost:11434/v1").text_response(
            model="qwen2.5:7b",
            system_prompt="Answer.",
            user_prompt="Hello.",
        )


def test_ollama_json_response_uses_safe_parse_for_clinical_case(monkeypatch: object) -> None:
    captured: dict[str, Any] = {}
    payload = sample_case()
    payload["diagnosis"] = "  septic shock  "
    payload["vitals"] = {
        "heart_rate": "132 bpm",
        "blood_pressure": "120 over 80",
        "respiratory_rate": "28",
        "oxygen_saturation": "91%",
    }

    def fake_urlopen(http_request: Any, timeout: int) -> FakeHTTPResponse:
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(payload)
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.base.request.urlopen", fake_urlopen)

    result = OllamaClient(base_url="http://localhost:11434/v1").parse_response(
        model="llama-med",
        system_prompt="Return JSON only.",
        user_prompt="Generate a case.",
        response_model=ClinicalCase,
    )

    assert result.diagnosis == "septic shock"
    assert result.vitals.heart_rate == 132
    assert result.vitals.blood_pressure == "120/80"
    assert result.vitals.oxygen_saturation == 91
    assert captured["payload"]["response_format"]["type"] == "json_schema"


def test_ollama_text_response_returns_message_content(monkeypatch: object) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(http_request: Any, timeout: int) -> FakeHTTPResponse:
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "I think the pain started about half an hour ago."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.base.request.urlopen", fake_urlopen)

    result = OllamaClient(base_url="http://localhost:11434/v1").text_response(
        model="llama-med",
        system_prompt="Answer as patient.",
        user_prompt="When did it start?",
    )

    assert result == "I think the pain started about half an hour ago."
    assert captured["payload"]["model"] == "llama-med"
    assert "response_format" not in captured["payload"]


def test_ollama_provider_uses_ollama_model_env(monkeypatch: object) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama-med")

    agent = OpenAIJsonAgent(client=object())

    assert get_llm_provider() == "ollama"
    assert agent.provider == "ollama"
    assert agent.model == "llama-med"


def test_openai_parsed_dictionary_uses_safe_parse_for_clinical_case() -> None:
    payload = sample_case()
    payload["vitals"]["heart_rate"] = "132 bpm"
    payload["vitals"]["blood_pressure"] = "BP 120/80"
    client = FakeOpenAIClient(payload)
    agent = OpenAIJsonAgent(client=client, provider="openai")

    result = agent.parse_response(
        system_prompt="Return JSON only.",
        user_prompt="Generate a case.",
        response_model=ClinicalCase,
    )

    assert result.vitals.heart_rate == 132
    assert result.vitals.blood_pressure == "120/80"


def test_safe_parse_returns_clean_error_after_failed_normalization() -> None:
    payload = sample_case()
    payload["vitals"]["blood_pressure"] = "one twenty over eighty"

    with pytest.raises(ValueError, match="failed to parse ClinicalCase after normalization"):
        safe_parse(ClinicalCase, payload)


def test_clinical_case_simulator_shares_one_ollama_model(monkeypatch: object) -> None:
    fake_client = object()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama-med")
    monkeypatch.setattr("backend.service.build_llm_client", lambda provider: fake_client)

    simulator = ClinicalCaseSimulator()

    assert simulator.generator.provider == "ollama"
    assert simulator.validator.provider == "ollama"
    assert simulator.generator.client is fake_client
    assert simulator.validator.client is fake_client
    assert simulator.generator.model == "llama-med"
    assert simulator.validator.model == "llama-med"


def test_all_llm_backed_agents_use_same_ollama_model(monkeypatch: object) -> None:
    fake_client = object()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama-med")

    agents = [
        GeneratorAgent(client=fake_client),
        PatientResponseAgent(client=fake_client),
        GenericInvestigationAgent(client=fake_client),
        ReasoningReviewAgent(client=fake_client),
        ValidatorAgent(client=fake_client),
    ]

    assert {agent.provider for agent in agents} == {"ollama"}
    assert {agent.model for agent in agents} == {"llama-med"}
    assert all(agent.client is fake_client for agent in agents)
