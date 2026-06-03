"""Central configuration. Env vars (DSCRIBE_*) override the defaults below."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # --- models ---
    agent_model: str = os.getenv("DSCRIBE_AGENT_MODEL", "gpt-5.2")
    vision_model: str = os.getenv("DSCRIBE_VISION_MODEL", "gpt-5.2")
    embed_model: str = os.getenv("DSCRIBE_EMBED_MODEL", "text-embedding-3-small")

    # --- agent control (requirement #9: the agent cannot run forever) ---
    max_steps: int = int(os.getenv("DSCRIBE_MAX_STEPS", "14"))

    # --- ingestion ---
    # A page whose extractable text layer has fewer than this many non-space
    # characters is treated as scanned/handwritten and routed to vision OCR.
    text_layer_min_chars: int = int(os.getenv("DSCRIBE_TEXT_MIN_CHARS", "40"))
    page_render_dpi: int = int(os.getenv("DSCRIBE_RENDER_DPI", "180"))

    # --- robustness (requirement #8) ---
    tool_max_retries: int = int(os.getenv("DSCRIBE_TOOL_RETRIES", "3"))
    llm_timeout_s: float = float(os.getenv("DSCRIBE_LLM_TIMEOUT", "60"))

    # --- storage ---
    storage_dir: Path = PROJECT_ROOT / "storage"

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")


CONFIG = Config()
