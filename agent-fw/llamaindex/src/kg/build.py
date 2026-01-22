from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from llama_index.core import KnowledgeGraphIndex, StorageContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.schema import Document


def build_kg_index(
    documents: List[Document],
    max_triplets_per_chunk: int,
    persist_dir: Optional[str] = None,
):
    if persist_dir:
        persist_path = Path(persist_dir)
        if persist_path.exists() and any(persist_path.iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            return load_index_from_storage(storage_context)

    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        include_embeddings=True,
        show_progress=True,
    )
    if persist_dir:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_path))
    return index

