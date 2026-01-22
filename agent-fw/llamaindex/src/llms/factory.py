from __future__ import annotations

from typing import Any, Dict, Optional

from llama_index.core import Settings

from llms.openai_compatible_embedding import OpenAICompatibleEmbedding
from llms.openai_compatible_llm import OpenAICompatibleLLM

def _build_mock_llm():
    try:
        from llama_index.core.llms import MockLLM

        return MockLLM()
    except Exception as exc:  # pragma: no cover - best effort fallback
        raise RuntimeError(
            "MockLLM is not available in this LlamaIndex version."
        ) from exc


def _build_openai_llm(
    model: str,
    base_url: str | None,
    api_key: str | None,
    context_window: int | None,
):
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai.utils import openai_modelname_to_contextsize

    kwargs = {"model": model}
    if base_url:
        kwargs["api_base"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    try:
        openai_modelname_to_contextsize(model)
        return OpenAI(**kwargs)
    except ValueError:
        return OpenAICompatibleLLM(
            **kwargs,
            context_window=context_window,
        )


def _build_mock_embedding():
    try:
        from llama_index.core.embeddings import MockEmbedding

        return MockEmbedding()
    except Exception as exc:  # pragma: no cover - best effort fallback
        raise RuntimeError(
            "MockEmbedding is not available in this LlamaIndex version."
        ) from exc


def _build_openai_embedding(
    model: str,
    base_url: str | None,
    api_key: str | None,
    context_window: int | None,
):
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.openai.base import OpenAIEmbeddingModelType

    kwargs = {"model": model}
    if base_url:
        kwargs["api_base"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if context_window:
        kwargs["context_window"] = context_window
    try:
        # Validate against OpenAI official model enum
        OpenAIEmbeddingModelType(model)
        return OpenAIEmbedding(**kwargs)
    except ValueError:
        # Fallback for OpenAI-compatible servers with custom model names
        return OpenAICompatibleEmbedding(
            model_name=model,
            api_key=api_key,
            api_base=base_url,
            context_window=context_window,
        )


def _build_hf_embedding(model: str, _context_window: int | None):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=model)


def init_llamaindex_settings(config: Dict[str, Any]) -> None:
    llm_cfg = config.get("llm", {})
    emb_cfg = config.get("embedding", {})

    llm_provider = llm_cfg.get("provider", "mock")
    llm_model = llm_cfg.get("model", "gpt-4o-mini")
    llm_base_url = llm_cfg.get("base_url") or None
    llm_api_key = llm_cfg.get("api_key") or None
    llm_context_window = int(llm_cfg.get("context_window", 8192))

    emb_provider = emb_cfg.get("provider", "mock")
    emb_model = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    emb_base_url = emb_cfg.get("base_url") or None
    emb_api_key = emb_cfg.get("api_key") or None
    emb_context_window = int(emb_cfg.get("context_window", llm_context_window))
    if llm_provider in {"openai", "openai_compatible"}:
        llm = _build_openai_llm(
            llm_model,
            llm_base_url,
            llm_api_key,
            llm_context_window,
        )
    elif llm_provider == "mock":
        llm = _build_mock_llm()
    else:
        raise ValueError(f"Unsupported llm.provider: {llm_provider}")

    if emb_provider in {"openai", "openai_compatible"}:
        embed_model = _build_openai_embedding(
            emb_model, emb_base_url, emb_api_key, emb_context_window
        )
    elif emb_provider == "huggingface":
        embed_model = _build_hf_embedding(emb_model, emb_context_window)
    elif emb_provider == "mock":
        embed_model = _build_mock_embedding()
    else:
        raise ValueError(f"Unsupported embedding.provider: {emb_provider}")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = llm_context_window


def get_llm_from_settings() -> Optional[object]:
    return Settings.llm

