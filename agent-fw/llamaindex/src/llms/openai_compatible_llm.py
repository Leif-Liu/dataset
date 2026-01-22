from __future__ import annotations

from typing import Optional

from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI


class OpenAICompatibleLLM(OpenAI):
    """LLM client for OpenAI-compatible servers with custom model names."""

    def __init__(
        self,
        *args,
        context_window: Optional[int] = None,
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        system_role: MessageRole = MessageRole.SYSTEM,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._compat_context_window = context_window or 8192
        self._compat_is_chat_model = is_chat_model
        self._compat_is_function_calling_model = is_function_calling_model
        self._compat_system_role = system_role

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._compat_context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=self._compat_is_chat_model,
            is_function_calling_model=self._compat_is_function_calling_model,
            model_name=self.model,
            system_role=self._compat_system_role,
        )

