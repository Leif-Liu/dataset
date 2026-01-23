from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


def _load_file(path: Path, parser_type: str, parser_cfg: Dict[str, Any]) -> List[Document]:
    if parser_type == "llamaparse":
        try:
            from llama_parse import LlamaParse
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("llama-parse not installed.") from exc

        lp_cfg = dict(parser_cfg.get("llamaparse", {}))
        api_key = lp_cfg.pop("api_key", None) or os.getenv("LLAMAPARSE_API_KEY")
        parser = LlamaParse(api_key=api_key, **lp_cfg)
        text = parser.load_data(str(path))
        if isinstance(text, list):
            return [Document(page_content=str(item)) for item in text]
        return [Document(page_content=str(text))]

    if parser_type == "unstructured":
        try:
            from unstructured.partition.auto import partition
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("unstructured not installed.") from exc

        elements = partition(filename=str(path))
        content = "\n".join([str(el) for el in elements])
        return [Document(page_content=content, metadata={"source": str(path)})]

    # simple loaders
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    if path.suffix.lower() == ".docx":
        return Docx2txtLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()


def load_documents(requirements_dir: str, parser_cfg: Dict[str, Any]) -> List[Document]:
    parser_type = (parser_cfg or {}).get("type", "simple")
    base = Path(requirements_dir)
    documents: List[Document] = []
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        documents.extend(_load_file(path, parser_type, parser_cfg))
    return documents


def build_rag_index(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str = "sentence",
    persist_dir: Optional[str] = None,
    embeddings=None,
):
    if persist_dir:
        persist_path = Path(persist_dir)
        if persist_path.exists() and any(persist_path.iterdir()):
            try:
                index = FAISS.load_local(
                    str(persist_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Validate embedding dimension compatibility.
                index.similarity_search("dimension check", k=1)
                return index
            except Exception:
                # Rebuild if index is incompatible with current embedding model.
                pass

    if splitter_type == "markdown":
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "hierarchical":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
    elif splitter_type == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(embeddings)
        except Exception:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    chunks = splitter.split_documents(documents)
    index = FAISS.from_documents(chunks, embeddings)
    if persist_dir:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        index.save_local(str(persist_path))
    return index

