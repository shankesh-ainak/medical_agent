"""Shared context handed to every tool: the index (read), the draft (write), and
the trace. `last_reasoning` carries the model's rationale from the agent node
into the tool's trace entry so each step reads reasoning -> action -> result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only; avoids pulling the heavy ingestion stack at import
    from ..ingestion.index_store import IndexStore
    from ..observability.trace import TraceLogger
    from .draft import DraftStore


@dataclass
class ToolContext:
    index: IndexStore
    draft: DraftStore
    trace: TraceLogger
    last_reasoning: str = ""
