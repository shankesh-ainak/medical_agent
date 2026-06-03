"""Unified index over all pages (typed + vision-OCR'd).

Two retrieval modes, deliberately:
  * semantic_search(...)  — LlamaIndex vector retrieval + metadata filtering.
        Efficient, context-aware EXPLORATION ("where might X be?").
  * pages_by_doc_type(...) / get_page(...) — exact, EXHAUSTIVE fetch from the
        in-memory page registry. Used by the safety-critical tools
        (reconciliation, conflict detection) because top-k retrieval can
        silently drop a page and would defeat the no-fabrication guarantee.

Every node and page keeps `page_no`, so provenance survives regardless of which
mode found the text.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..config import CONFIG
from .extractor import PageDoc


@dataclass
class Hit:
    page_no: int
    doc_type: str
    text: str
    score: float | None = None


class IndexStore:
    def __init__(self, pages: list[PageDoc], index: VectorStoreIndex):
        self._pages = {p.page_no: p for p in pages}
        self._index = index
        self._embed = index._embed_model  # reused for query embedding

    # ---- exact / exhaustive (safety-critical path) ----
    def list_pages(self) -> list[dict]:
        return [
            {
                "page_no": p.page_no,
                "doc_type": p.doc_type,
                "source": p.source,
                "confidence": p.confidence,
                "preview": p.text[:160].replace("\n", " "),
            }
            for p in sorted(self._pages.values(), key=lambda x: x.page_no)
        ]

    def get_page(self, page_no: int) -> PageDoc | None:
        return self._pages.get(page_no)

    def pages_by_doc_type(self, *doc_types: str) -> list[PageDoc]:
        wanted = set(doc_types)
        return [
            p for p in sorted(self._pages.values(), key=lambda x: x.page_no)
            if p.doc_type in wanted
        ]

    # ---- semantic (exploration path) ----
    def semantic_search(
        self, query: str, doc_type: str | None = None, top_k: int = 5
    ) -> list[Hit]:
        filters = None
        if doc_type:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="doc_type", value=doc_type,
                               operator=FilterOperator.EQ)
            ])
        retriever = self._index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes = retriever.retrieve(query)
        return [
            Hit(
                page_no=int(n.metadata.get("page_no", -1)),
                doc_type=n.metadata.get("doc_type", "unknown"),
                text=n.get_content(),
                score=n.score,
            )
            for n in nodes
        ]


def build_index_store(
    pages: list[PageDoc], collection_name: str = "dscribe"
) -> IndexStore:
    """Embed pages into a persisted Chroma-backed vector index, carrying page
    metadata onto every node."""
    embed = OpenAIEmbedding(model=CONFIG.embed_model)

    nodes: list[TextNode] = []
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    for p in pages:
        if not p.text.strip():
            continue
        meta = {
            "page_no": p.page_no,
            "doc_type": p.doc_type,
            "source": p.source,
            "confidence": p.confidence,
        }
        for chunk in splitter.split_text(p.text):
            nodes.append(TextNode(text=chunk, metadata=meta.copy()))

    CONFIG.storage_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CONFIG.storage_dir / "chroma"))
    # Fresh collection per build keeps one patient's chart isolated.
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001 — collection may not exist yet
        pass
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes, storage_context=storage_context, embed_model=embed
    )
    return IndexStore(pages, index)
