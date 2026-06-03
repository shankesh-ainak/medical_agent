"""End-to-end orchestration shared by the CLI and the Streamlit app:
PDF -> extract pages -> build index -> run agent -> structured draft + trace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent.loop import run_agent
from .config import CONFIG
from .ingestion.extractor import PageDoc, extract_pages
from .ingestion.index_store import IndexStore, build_index_store
from .observability.trace import TraceLogger
from .render.markdown import render_markdown
from .schema import DischargeSummary
from .tools.context import ToolContext
from .tools.draft import DraftStore


@dataclass
class PipelineResult:
    summary: DischargeSummary
    markdown: str
    trace: list[dict]
    pages: list[PageDoc]
    index: IndexStore


def ingest(pdf_path: str, use_cache: bool = True) -> IndexStore:
    pages = extract_pages(pdf_path, use_cache=use_cache)
    return build_index_store(pages)


def run(pdf_path: str, use_cache: bool = True, echo_trace: bool = True) -> PipelineResult:
    pages = extract_pages(pdf_path, use_cache=use_cache)
    index = build_index_store(pages)

    valid_pages = {p.page_no for p in pages}
    draft = DraftStore(page_exists=lambda n: n in valid_pages)
    trace = TraceLogger(
        jsonl_path=CONFIG.storage_dir / "trace" / (Path(pdf_path).stem + ".jsonl"),
        echo=echo_trace,
    )
    ctx = ToolContext(index=index, draft=draft, trace=trace)

    summary = run_agent(ctx)
    return PipelineResult(
        summary=summary,
        markdown=render_markdown(summary),
        trace=trace.as_list(),
        pages=pages,
        index=index,
    )
