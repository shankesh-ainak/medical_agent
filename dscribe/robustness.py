"""Shared robustness helpers (requirement #8).

Failure-prone calls (vision OCR, embeddings, external tools) are wrapped so they
retry with backoff and then surface a structured error instead of crashing. The
agent receives errors as ordinary observations and can retry, fall back to
another source, or mark a field missing/pending — but a failed call can never be
mistaken for a successful one.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import CONFIG

T = TypeVar("T")


class ToolError(Exception):
    """Raised inside a tool when work genuinely fails after retries."""


def with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator: retry a flaky callable with exponential backoff."""
    return retry(
        stop=stop_after_attempt(CONFIG.tool_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(fn)


def safe_result(fn: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
    """Decorator for tools: never let an exception escape. Returns a structured
    {"status": "ok"|"error", ...} envelope so the agent can react to failure."""

    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict) and "status" in result:
                return result
            return {"status": "ok", "result": result}
        except Exception as exc:  # noqa: BLE001 — boundary: convert to envelope
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    wrapper.__name__ = getattr(fn, "__name__", "tool")
    wrapper.__doc__ = fn.__doc__
    return wrapper
