from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel


TRIPLE_PROMPT = """Extract knowledge triplets from the text.
Return JSON list of objects with keys: subject, relation, object.
Text:
{text}
"""


def _extract_triplets(text: str, llm: BaseChatModel) -> List[Tuple[str, str, str]]:
    response = llm.invoke(TRIPLE_PROMPT.format(text=text))
    content = getattr(response, "content", "") or str(response)
    try:
        data = json.loads(content)
    except Exception:
        return []
    triplets = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject", "")).strip()
            r = str(item.get("relation", "")).strip()
            o = str(item.get("object", "")).strip()
            if s and r and o:
                triplets.append((s, r, o))
    return triplets


def build_kg_index(
    documents: List[Document],
    max_triplets_per_chunk: int,
    llm: BaseChatModel,
    persist_dir: Optional[str] = None,
) -> nx.DiGraph:
    if persist_dir:
        persist_path = Path(persist_dir)
        if persist_path.exists() and any(persist_path.iterdir()):
            with open(persist_path / "kg.pkl", "rb") as f:
                return pickle.load(f)

    graph = nx.DiGraph()
    for doc in documents:
        triplets = _extract_triplets(doc.page_content, llm)[:max_triplets_per_chunk]
        for s, r, o in triplets:
            graph.add_edge(s, o, relation=r)

    if persist_dir:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        with open(persist_path / "kg.pkl", "wb") as f:
            pickle.dump(graph, f)
    return graph


def query_kg(graph: nx.DiGraph, query: str, limit: int = 5) -> str:
    hits = []
    q = query.lower()
    for u, v, data in graph.edges(data=True):
        if q in str(u).lower() or q in str(v).lower() or q in str(data).lower():
            rel = data.get("relation", "")
            hits.append(f"{u} -[{rel}]-> {v}")
        if len(hits) >= limit:
            break
    return "\n".join(hits) if hits else "No matching relations."

