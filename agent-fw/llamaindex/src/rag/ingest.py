from __future__ import annotations

from typing import List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


def load_documents(requirements_dir: str) -> List[Document]:
    reader = SimpleDirectoryReader(input_dir=requirements_dir, recursive=True)
    return reader.load_data()


def build_rag_index(documents: List[Document], chunk_size: int, chunk_overlap: int):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )
    return index

