from __future__ import annotations

import asyncio
from collections.abc import Iterable

from graphiti_core.embedder.client import EmbedderClient

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class LocalEmbedder(EmbedderClient):
    """Lazy-loaded local embedder via HuggingFace transformers (no API key required)."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModel.from_pretrained(MODEL_NAME)
        self._model.eval()

    def _embed_sync(self, text: str) -> list[float]:
        import torch
        self._ensure_loaded()
        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            output = self._model(**inputs)
        return output.last_hidden_state.mean(dim=1).squeeze(0).tolist()

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        text = input_data if isinstance(input_data, str) else next(iter(input_data))  # type: ignore[arg-type]
        if not isinstance(text, str):
            raise ValueError("LocalEmbedder only supports string input")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [await self.create(text) for text in input_data_list]
