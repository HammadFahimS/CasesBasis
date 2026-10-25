from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar
from urllib import error, request

logger = logging.getLogger(__name__)

from langsmith import traceable
from pydantic import BaseModel
from schemas.clinical_case import ClinicalCase, normalize_case_data

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
# qwen-3.8-27b is the general-purpose reasoning model for the live voice sim
# (nurse brain, routing, actors, director — everything that flows through
# agent_runtime.complete) and the default for anything that doesn't pin a model.
# It is the default because it responds fast; zai-glm-4.7 is capable but too slow
# for an interactive room. gpt-oss-120b is reserved for two things where a stronger
# model is required and the latency is acceptable (off the conversational path):
#   - ECG parameter generation (see CEREBRAS_ECG_MODEL + agents/ecg.py)
#   - the manual case simulator, all of it (see CEREBRAS_MANUAL_MODEL +
#     manual_sim_model() + backend/sim_core.py), because the fast live model
#     can't reliably satisfy the strict ClinicalCase/MedicalFacts schemas.
# (gemma-4-31b held this slot until Cerebras archived it in October 2026.)
DEFAULT_CEREBRAS_MODEL = "qwen-3.8-27b"
CEREBRAS_ECG_MODEL = "gpt-oss-120b"
CEREBRAS_MANUAL_MODEL = "gpt-oss-120b"
CEREBRAS_FALLBACK_MODELS = [
    "qwen-3.8-27b",
    "gpt-oss-120b",
]
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
SUPPORTED_LLM_PROVIDERS = {"openai", "ollama", "groq", "together", "openrouter", "cerebras"}


def build_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Install dependencies with "
            "`pip install -e .` or `pip install -e .[dev]`."
        ) from exc
    return OpenAI()


def build_groq_client() -> Any:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The groq package is not installed. Run: pip install groq"
        ) from exc
    return GroqClient(Groq(api_key=os.environ["GROQ_API_KEY"]))


def build_together_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai"
        ) from exc
    client = OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url=TOGETHER_BASE_URL,
    )
    return GroqClient(client)  # same json_object interface as Groq


def build_cerebras_client() -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Run: pip install openai") from exc
    # A dedicated http client with a 5-minute keepalive so the TLS connection to
    # Cerebras stays warm between voice turns. The OpenAI SDK's default pool uses
    # httpx's 5 s keepalive_expiry, which drops the connection during the gap
    # while the resident talks/listens — making the *first* LLM call of the next
    # turn pay a full cold handshake (~1.8 s measured, vs ~150 ms warm).
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=4, max_connections=8, keepalive_expiry=300.0
        ),
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    client = OpenAI(
        api_key=os.environ["CEREBRAS_API_KEY"],
        base_url=CEREBRAS_BASE_URL,
        http_client=http_client,
    )
    return CerebrasClient(client)


def build_openrouter_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai"
        ) from exc
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/CasesBasis",
            "X-Title": "CasesBasis",
        },
    )
    return GroqClient(client)  # same json_object interface


def get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError("LLM_PROVIDER must be one of: openai, ollama, groq, together, openrouter, cerebras")
    return provider


def resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if provider == "groq":
        return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if provider == "together":
        return os.getenv("TOGETHER_MODEL", DEFAULT_TOGETHER_MODEL)
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    if provider == "cerebras":
        return os.getenv("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def manual_sim_model(provider: str | None = None) -> str | None:
    """The model the manual case simulator must use, or ``None`` to accept the
    provider default.

    On Cerebras the default (``zai-glm-4.7``) can't reliably satisfy the manual
    sim's strict ClinicalCase/MedicalFacts/investigation schemas, so we pin
    ``gpt-oss-120b`` for every manual-sim agent regardless of the ``CEREBRAS_MODEL``
    env default. Other providers keep their own default (return ``None``)."""
    provider = provider or get_llm_provider()
    return CEREBRAS_MANUAL_MODEL if provider == "cerebras" else None


def build_llm_client(provider: str | None = None) -> Any:
    resolved_provider = provider or get_llm_provider()
    if resolved_provider == "ollama":
        return OllamaClient()
    if resolved_provider == "groq":
        return build_groq_client()
    if resolved_provider == "together":
        return build_together_client()
    if resolved_provider == "openrouter":
        return build_openrouter_client()
    if resolved_provider == "cerebras":
        return build_cerebras_client()
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


class GroqClient:
    """Wrapper around the Groq SDK using chat completions with JSON schema output."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def parse_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
        )
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return safe_parse(response_model, content)

    def text_response(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq response did not include text output")
        return content.strip()


class CerebrasClient:
    """Cerebras client with automatic model fallback on 503 / server errors."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _model_chain(self, preferred: str) -> list[str]:
        chain = [preferred] + [m for m in CEREBRAS_FALLBACK_MODELS if m != preferred]
        return chain

    def _is_retryable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "503" in msg or "502" in msg or "404" in msg
            or "no server" in msg or "unavailable" in msg or "overloaded" in msg
            or "not_found" in msg or "does not exist" in msg or "model not found" in msg
        )

    def parse_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
        )
        last_exc: Exception | None = None
        for m in self._model_chain(model):
            try:
                response = self._client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": augmented_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                return safe_parse(response_model, response.choices[0].message.content)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                logger.warning("Cerebras model %s unavailable (%s), trying next fallback", m, exc)
        raise last_exc  # type: ignore[misc]

    def text_response(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        last_exc: Exception | None = None
        for m in self._model_chain(model):
            try:
                response = self._client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Cerebras response did not include text output")
                return content.strip()
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                logger.warning("Cerebras model %s unavailable (%s), trying next fallback", m, exc)
        raise last_exc  # type: ignore[misc]


def safe_parse(response_model: type[ResponseModel], raw_value: Any) -> ResponseModel:
    try:
        return parse_model_value(response_model, raw_value)
    except Exception as direct_error:
        try:
            normalized_value = normalize_llm_output(response_model, raw_value)
            return parse_model_value(response_model, normalized_value)
        except Exception:
            raise ValueError(
                f"failed to parse {response_model.__name__} after normalization"
            ) from direct_error


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

    @traceable(run_type="llm")
    def parse_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        if self.provider in ("ollama", "groq", "together", "openrouter", "cerebras"):
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

    @traceable(run_type="llm")
    def text_response(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.provider in ("ollama", "groq", "together", "openrouter", "cerebras"):
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
