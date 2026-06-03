"""Grounded Q&A over an already-built index (the chat tab). Same no-fabrication
discipline as the agent: answer only from retrieved source excerpts, cite the
pages, and say so plainly when the records don't contain the answer."""

from __future__ import annotations

from openai import OpenAI

from .config import CONFIG
from .ingestion.index_store import IndexStore

_SYSTEM = (
    "You answer questions about a patient strictly from the provided source-note "
    "excerpts. Rules: (1) Use ONLY the excerpts; never invent or infer clinical "
    "facts. (2) Cite the page(s) you used like [p3]. (3) If the answer is not in "
    "the excerpts, say: 'I can't find that in the records.' (4) If sources "
    "disagree, present both and note the conflict."
)


def answer_question(index: IndexStore, question: str,
                    history: list[dict] | None = None) -> dict:
    hits = index.semantic_search(question, top_k=6)
    context = "\n\n".join(
        f"[p{h.page_no} · {h.doc_type}]\n{h.text}" for h in hits
    ) or "(no relevant excerpts found)"

    messages = [{"role": "system", "content": _SYSTEM}]
    for turn in (history or [])[-6:]:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": f"Source excerpts:\n{context}\n\nQuestion: {question}",
    })

    resp = OpenAI(timeout=CONFIG.llm_timeout_s).chat.completions.create(
        model=CONFIG.agent_model, temperature=0, messages=messages,
    )
    return {
        "answer": resp.choices[0].message.content or "",
        "pages": sorted({h.page_no for h in hits}),
    }
