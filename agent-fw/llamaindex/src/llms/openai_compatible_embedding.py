from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from openai import AsyncOpenAI, OpenAI


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Embedding client for OpenAI-compatible servers with arbitrary model names."""

    model_name: str = Field(description="Embedding model name.")
    api_key: Optional[str] = Field(default=None, description="API key.")
    api_base: Optional[str] = Field(default=None, description="Base URL.")
    timeout: float = Field(default=60.0, description="Request timeout (s).")
    max_retries: int = Field(default=3, description="Max retry attempts.")
    context_window: Optional[int] = Field(
        default=None, description="Context window (kept for config parity)."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Extra kwargs for embedding API."
    )

    _client: Optional[OpenAI] = PrivateAttr(default=None)
    _aclient: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._aclient

    def _get_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            input=[query], model=self.model_name, **self.additional_kwargs
        )
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        client = self._get_aclient()
        response = await client.embeddings.create(
            input=[query], model=self.model_name, **self.additional_kwargs
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aget_query_embedding(text)

