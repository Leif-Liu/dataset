from __future__ import annotations

from typing import Any, Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings


def init_models(config: Dict[str, Any]) -> Tuple[BaseChatModel, Embeddings]:
    llm_cfg = config.get("llm", {})
    emb_cfg = config.get("embedding", {})

    llm_provider = llm_cfg.get("provider", "mock")
    llm_model = llm_cfg.get("model", "gpt-4o-mini")
    llm_base_url = llm_cfg.get("base_url") or None
    llm_api_key = llm_cfg.get("api_key") or None

    emb_provider = emb_cfg.get("provider", "mock")
    emb_model = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    emb_base_url = emb_cfg.get("base_url") or None
    emb_api_key = emb_cfg.get("api_key") or None

    if llm_provider in {"openai", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
    elif llm_provider == "mock":
        from langchain_core.language_models.fake import FakeListLLM

        llm = FakeListLLM(responses=["Mock response"])
    else:
        raise ValueError(f"Unsupported llm.provider: {llm_provider}")

    if emb_provider in {"openai", "openai_compatible"}:
        from langchain_openai import OpenAIEmbeddings

        embed_model = OpenAIEmbeddings(
            model=emb_model,
            base_url=emb_base_url,
            api_key=emb_api_key,
        )
    elif emb_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embed_model = HuggingFaceEmbeddings(model_name=emb_model)
    elif emb_provider == "mock":
        from langchain_core.embeddings import FakeEmbeddings

        embed_model = FakeEmbeddings(size=384)
    else:
        raise ValueError(f"Unsupported embedding.provider: {emb_provider}")

    return llm, embed_model

