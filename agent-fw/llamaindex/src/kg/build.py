from __future__ import annotations

from typing import List

from llama_index.core import KnowledgeGraphIndex
from llama_index.core.schema import Document


def build_kg_index(documents: List[Document], max_triplets_per_chunk: int):
    return KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
        show_progress=True,
    )

