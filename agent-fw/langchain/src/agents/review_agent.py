from __future__ import annotations

from langchain_classic.agents import AgentExecutor
from langchain_core.language_models.chat_models import BaseChatModel

from agents.base_agent import build_react_agent


def build_review_agent(llm: BaseChatModel, tools) -> AgentExecutor:
    return build_react_agent(llm, tools)

