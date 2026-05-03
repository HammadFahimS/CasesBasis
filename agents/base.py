from __future__ import annotations

import json
import os
from typing import Any, TypeVar
from urllib import error, request

from pydantic import BaseModel
from schemas.clinical_case import ClinicalCase, normalize_case_data

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
SUPPORTED_LLM_PROVIDERS = {"openai", "ollama"}


def build_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Install dependencies with "
            "`pip install -e .` or `pip install -e .[dev]`."
        ) from exc
    return OpenAI()


def get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError("LLM_PROVIDER must be one of: openai, ollama")
    return provider


def resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def build_llm_client(provider: str | None = None) -> Any:
    resolved_provider = provider or get_llm_provider()
    if resolved_provider == "ollama":
        return OllamaClient()
    return build_openai_client()


class OllamaClient:
    """Minimal client for Ollama's OpenAI-compatible chat completions API."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")

    def parse_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        payload = self._chat_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            },
        )
        response = self._post_json("/chat/completions", payload)
        content = self._message_content(response)
        return safe_parse(response_model, content)

    def text_response(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        payload = self._chat_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=None,
        )
        response = self._post_json("/chat/completions", payload)
        content = self._message_content(response).strip()
        if not content:
            raise ValueError("Ollama response did not include text output")
        return content

    def _chat_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Ollama server is not reachable. Start Ollama and ensure "
                f"the OpenAI-compatible endpoint is available at {self.base_url}."
            ) from exc

    def _message_content(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Ollama response did not include choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Ollama response included an invalid choice")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response did not include a message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not include message content")
        return content


def safe_parse(response_model: type[ResponseModel], raw_value: Any) -> ResponseModel:
    try:
        return parse_model_value(response_model, raw_value)
    except Exception as direct_error:
        try:
            normalized_value = normalize_llm_output(response_model, raw_value)
            return parse_model_value(response_model, normalized_value)
        except Exception as normalized_error:
            raise ValueError(
                f"failed to parse {response_model.__name__} after normalization"
            ) from normalized_error or direct_error


def parse_model_value(response_model: type[ResponseModel], raw_value: Any) -> ResponseModel:
    if isinstance(raw_value, response_model):
        return raw_value
    if isinstance(raw_value, str):
        return response_model.model_validate_json(raw_value)
    return response_model.model_validate(raw_value)


def normalize_llm_output(response_model: type[ResponseModel], raw_value: Any) -> Any:
    value = decode_json_if_needed(raw_value)
    if response_model is ClinicalCase:
        return normalize_case_data(value)
    return value


def decode_json_if_needed(raw_value: Any) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    return json.loads(raw_value)


class OpenAIJsonAgent:
    """Provider-aware base class for structured JSON and text LLM calls."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.provider = provider or get_llm_provider()
        self.client = client or build_llm_client(self.provider)
        self.model = resolve_model(self.provider, model)

    def parse_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        if self.provider == "ollama":
            return self.client.parse_response(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("OpenAI response did not include parsed structured output")
        return safe_parse(response_model, parsed)

    def text_response(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "ollama":
            return self.client.text_response(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", None)
        if not text:
            raise ValueError("OpenAI response did not include text output")
        return text.strip()
