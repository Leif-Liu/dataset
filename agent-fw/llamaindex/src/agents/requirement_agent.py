from __future__ import annotations

from llama_index.core.agent import AgentWorkflow

from agents.tools import build_query_tools


def build_requirement_agent(vector_index, kg_index):
    tools = build_query_tools(vector_index, kg_index)
    return AgentWorkflow.from_tools_or_functions(tools, verbose=True)

