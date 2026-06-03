"""Assemble the agent's tools as LangChain StructuredTools bound to a
ToolContext. Each tool is wrapped so that (a) exceptions become structured error
envelopes (never crash the loop) and (b) every call emits a trace step."""

from __future__ import annotations

import functools
import json
from typing import Any

from langchain_core.tools import StructuredTool

from ..robustness import with_retry
from . import conflicts, drug_interactions, reconciliation
from .context import ToolContext


def build_tools(ctx: ToolContext) -> list[StructuredTool]:
    def traced(fn):
        @functools.wraps(fn)
        def inner(**kwargs: Any) -> str:
            try:
                env = fn(**kwargs)
                if not (isinstance(env, dict) and "status" in env):
                    env = {"status": "ok", "result": env}
            except Exception as exc:  # noqa: BLE001 — boundary -> envelope
                env = {"status": "error", "error_type": type(exc).__name__,
                       "message": str(exc)}
            ctx.trace.log(
                reasoning=ctx.last_reasoning,
                action=fn.__name__,
                inputs=kwargs,
                result=env.get("summary") or json.dumps(env, default=str),
            )
            ctx.last_reasoning = ""  # consumed by this step
            return json.dumps(env, default=str)
        return inner

    # ---------- document access ----------
    def list_source_documents() -> dict:
        """List every source page in this patient's bundle with its doc_type,
        extraction source (text-layer/vision), legibility confidence, and a
        preview. Call this first to plan what to read."""
        return {"status": "ok", "pages": ctx.index.list_pages()}

    def read_document(page_no: int) -> dict:
        """Return the FULL extracted text of one source page. Use this to read a
        page end-to-end (e.g. a medication table) without retrieval truncation."""
        p = ctx.index.get_page(page_no)
        if p is None:
            return {"status": "error", "message": f"page {page_no} not found"}
        return {"status": "ok", "page_no": p.page_no, "doc_type": p.doc_type,
                "source": p.source, "confidence": p.confidence, "text": p.text}

    def search_documents(query: str, doc_type: str | None = None,
                         top_k: int = 5) -> dict:
        """Semantic search across all pages for where a fact might appear.
        Optionally filter by doc_type (e.g. 'lab_report'). Use for EXPLORATION;
        for complete comparisons prefer reading whole pages."""
        hits = ctx.index.semantic_search(query, doc_type=doc_type, top_k=top_k)
        return {"status": "ok", "hits": [
            {"page_no": h.page_no, "doc_type": h.doc_type,
             "score": h.score, "text": h.text} for h in hits]}

    # ---------- deterministic clinical logic ----------
    def reconcile_medications(admission: list[dict], discharge: list[dict]) -> dict:
        """Compare admission vs discharge medications. Each item: {name, dose?,
        reason?}. Returns added/stopped/changed/continued classification and
        flags changes lacking a documented reason. Source both lists from the
        documents first."""
        return reconciliation.reconcile(admission, discharge)

    def detect_conflicts(field: str, observations: list[dict]) -> dict:
        """Decide whether sources disagree on a field. observations: [{value,
        page_no, doc_type?, quote?}, ...]. Never picks a winner — reports the
        distinct values and their sources."""
        return conflicts.detect(field, observations)

    def check_drug_interactions(medications: list[str]) -> dict:
        """MOCKED external drug-interaction lookup. medications: list of drug
        names. May transiently fail (auto-retried). Escalate any interactions."""
        return with_retry(drug_interactions.check_interactions)(medications)

    # ---------- escalation ----------
    def flag_for_clinician_review(topic: str, reason: str,
                                  severity: str = "warning") -> dict:
        """Surface a safety concern (conflict, unreconciled med change,
        interaction, missing critical data) for the clinician. severity:
        info|warning|critical."""
        return ctx.draft.add_review_flag(topic, reason, severity)

    # ---------- guarded draft writers ----------
    def record_field(section: str, value: str, sources: list[dict]) -> dict:
        """Record a sourced value for a section. sources MUST be non-empty:
        [{page_no, doc_type?, quote?}, ...]. Rejected if you cannot cite a real
        page — use mark_missing/mark_pending/record_conflict instead."""
        return ctx.draft.record_field(section, value, sources)

    def mark_missing(section: str, note: str | None = None) -> dict:
        """Mark a section as not present anywhere in the source notes."""
        return ctx.draft.mark_missing(section, note)

    def mark_pending(section: str, note: str,
                     sources: list[dict] | None = None) -> dict:
        """Mark a section as ordered/awaited (e.g. culture pending). Do NOT
        invent a result. Cite the page that says it is pending if available."""
        return ctx.draft.mark_pending(section, note, sources)

    def record_conflict(section: str, options: list[dict],
                        note: str | None = None) -> dict:
        """Record disagreeing sources for a section without choosing. options:
        [{value, sources:[{page_no, doc_type?, quote?}]}, ...] (>=2)."""
        return ctx.draft.record_conflict(section, options, note)

    def set_discharge_medications(medications: list[dict]) -> dict:
        """Set the discharge medication list with admission->discharge changes.
        Each item: {name, admission?, discharge?, change_type
        (added|stopped|changed|continued), reason?, needs_reconciliation?,
        sources:[{page_no,...}]}."""
        return ctx.draft.set_discharge_medications(medications)

    def finalize_draft() -> dict:
        """Finish. Only call once every required section is recorded, missing,
        pending, or in conflict. Auto-marks any untouched section as missing."""
        return ctx.draft.finalize()

    funcs = [
        list_source_documents, read_document, search_documents,
        reconcile_medications, detect_conflicts, check_drug_interactions,
        flag_for_clinician_review, record_field, mark_missing, mark_pending,
        record_conflict, set_discharge_medications, finalize_draft,
    ]
    return [
        StructuredTool.from_function(
            func=traced(fn), name=fn.__name__, description=fn.__doc__ or fn.__name__,
        )
        for fn in funcs
    ]
