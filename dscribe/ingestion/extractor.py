"""Hybrid extraction: route each page to the cheap digital-text path or the
vision-OCR path, producing one unified `PageDoc` per page. Results are cached on
disk by file hash so repeated runs don't re-OCR (vision calls are slow/costly)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import CONFIG
from .pdf_loader import load_pages
from .vision_ocr import transcribe_page


@dataclass
class PageDoc:
    page_no: int
    doc_type: str
    source: str         # "text-layer" | "vision"
    confidence: str     # "high" | "medium" | "low"
    text: str


# Header keywords -> doc_type, for cheap classification of typed pages.
_TYPED_HINTS = [
    ("discharge_note", ("advice on discharge", "condition at discharge",
                        "course in the hospital", "follow-up instructions")),
    ("admission_note", ("chief complaint", "history of present illness",
                        "reason for admission")),
    ("lab_report", ("investigations", "biochemistry", "haemogram", "hemogram")),
    ("medication_record", ("medication name", "dosage", "frequency")),
]


def _classify_typed(text: str) -> str:
    low = text.lower()
    for doc_type, keys in _TYPED_HINTS:
        if any(k in low for k in keys):
            return doc_type
    return "other"


def _cache_path(pdf_path: str) -> Path:
    digest = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:16]
    cache_dir = CONFIG.storage_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest}.json"


def extract_pages(pdf_path: str, use_cache: bool = True) -> list[PageDoc]:
    """Extract every page. Typed pages (good text layer) are used directly;
    pages with little/no text are sent to vision OCR. Falls back gracefully: a
    vision failure on one page yields a low-confidence PageDoc with an error
    note rather than aborting the whole ingest."""
    cache = _cache_path(pdf_path)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        return [PageDoc(**d) for d in raw]

    pages: list[PageDoc] = []
    for raw in load_pages(pdf_path, dpi=CONFIG.page_render_dpi):
        if len(raw.text_layer) >= CONFIG.text_layer_min_chars:
            pages.append(PageDoc(
                page_no=raw.page_no,
                doc_type=_classify_typed(raw.text_layer),
                source="text-layer",
                confidence="high",
                text=raw.text_layer,
            ))
            continue

        # Sparse/empty text layer -> handwritten or scanned -> vision OCR.
        try:
            v = transcribe_page(raw.image_b64)
            pages.append(PageDoc(
                page_no=raw.page_no, doc_type=v.doc_type,
                source="vision", confidence=v.confidence, text=v.text,
            ))
        except Exception as exc:  # noqa: BLE001 — page-level fallback
            pages.append(PageDoc(
                page_no=raw.page_no, doc_type="other", source="vision",
                confidence="low",
                text=f"[OCR_FAILED: {type(exc).__name__}: {exc}]",
            ))

    if use_cache:
        cache.write_text(json.dumps([asdict(p) for p in pages], indent=2))
    return pages
