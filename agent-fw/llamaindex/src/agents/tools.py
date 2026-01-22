from __future__ import annotations

from llama_index.core.tools import QueryEngineTool, ToolMetadata


def build_query_tools(vector_index, kg_index):
    rag_engine = vector_index.as_query_engine(similarity_top_k=5)
    kg_engine = kg_index.as_query_engine(similarity_top_k=5)

    tools = [
        QueryEngineTool(
            query_engine=rag_engine,
            metadata=ToolMetadata(
                name="rag_search",
                description="Search requirement documents using vector index.",
            ),
        ),
        QueryEngineTool(
            query_engine=kg_engine,
            metadata=ToolMetadata(
                name="kg_search",
                description="Query knowledge graph for entities and relations.",
            ),
        ),
    ]
    return tools

