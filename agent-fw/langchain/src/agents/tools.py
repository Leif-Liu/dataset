from __future__ import annotations

from typing import Callable

from langchain_core.tools import Tool


def build_tools(rag_search: Callable[[str], str], kg_search: Callable[[str], str]):
    return [
        Tool.from_function(
            name="rag_search",
            description="Search requirement documents using vector index.",
            func=rag_search,
        ),
        Tool.from_function(
            name="kg_search",
            description="Query knowledge graph for entities and relations.",
            func=kg_search,
        ),
    ]

