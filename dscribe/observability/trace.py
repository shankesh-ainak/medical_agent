"""Per-step trace (requirement #10): reasoning -> tool/action -> inputs ->
result -> next decision. Emitted as in-memory records, a JSONL file, and a
human-readable console line so a reviewer can replay every decision."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _truncate(obj: Any, limit: int = 300) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


@dataclass
class TraceStep:
    step: int
    reasoning: str          # the model's rationale before acting
    action: str             # tool name, or "respond"/"finalize"
    inputs: dict[str, Any] = field(default_factory=dict)
    result: str = ""        # short summary of the observation
    next_decision: str = ""


class TraceLogger:
    def __init__(self, jsonl_path: str | Path | None = None, echo: bool = True):
        self.steps: list[TraceStep] = []
        self._counter = 0
        self._echo = echo
        self._path = Path(jsonl_path) if jsonl_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("")  # truncate previous run

    def log(self, reasoning: str, action: str, inputs: dict | None = None,
            result: str = "", next_decision: str = "") -> None:
        self._counter += 1
        step = TraceStep(
            step=self._counter,
            reasoning=_truncate(reasoning or ""),
            action=action,
            inputs={k: _truncate(v) for k, v in (inputs or {}).items()},
            result=_truncate(result),
            next_decision=_truncate(next_decision),
        )
        self.steps.append(step)
        if self._path:
            with self._path.open("a") as fh:
                fh.write(json.dumps(asdict(step), default=str) + "\n")
        if self._echo:
            self._print(step)

    def _print(self, s: TraceStep) -> None:
        out = sys.stderr
        print(f"\n── step {s.step} ─────────────────────────────", file=out)
        if s.reasoning:
            print(f"  reasoning : {s.reasoning}", file=out)
        print(f"  action    : {s.action}", file=out)
        if s.inputs:
            print(f"  inputs    : {s.inputs}", file=out)
        if s.result:
            print(f"  result    : {s.result}", file=out)
        if s.next_decision:
            print(f"  next      : {s.next_decision}", file=out)

    def as_list(self) -> list[dict]:
        return [asdict(s) for s in self.steps]
