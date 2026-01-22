from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import Document


def load_documents(requirements_dir: str, parser_cfg: Dict[str, Any]) -> List[Document]:
    parser_type = (parser_cfg or {}).get("type", "simple")

    if parser_type == "llamaparse":
        try:
            from llama_index.readers.llama_parse import LlamaParse
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LlamaParse not available. Install llama-index-readers-llama-parse."
            ) from exc

        lp_cfg = dict(parser_cfg.get("llamaparse", {}))
        api_key = lp_cfg.pop("api_key", None)
        reader = SimpleDirectoryReader(
            input_dir=requirements_dir,
            recursive=True,
            file_extractor={
                ".pdf": LlamaParse(api_key=api_key, **lp_cfg),
            },
        )
    elif parser_type == "unstructured":
        try:
            from llama_index.readers.unstructured import UnstructuredReader
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "UnstructuredReader not available. Install llama-index-readers-unstructured."
            ) from exc

        un_cfg = dict(parser_cfg.get("unstructured", {}))
        reader = SimpleDirectoryReader(
            input_dir=requirements_dir,
            recursive=True,
            file_extractor={
                ".pdf": UnstructuredReader(**un_cfg),
                ".docx": UnstructuredReader(**un_cfg),
            },
        )
    else:
        reader = SimpleDirectoryReader(input_dir=requirements_dir, recursive=True)
    return reader.load_data()


def build_rag_index(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str = "sentence",
    persist_dir: Optional[str] = None,
):
    if persist_dir:
        persist_path = Path(persist_dir)
        if persist_path.exists() and any(persist_path.iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            return load_index_from_storage(storage_context)

    if splitter_type == "semantic":
        splitter = SemanticSplitterNodeParser(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_type == "hierarchical":
        splitter = HierarchicalNodeParser.from_defaults(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_type == "markdown":
        splitter = MarkdownNodeParser()
    else:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )
    if persist_dir:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_path))
    return index

