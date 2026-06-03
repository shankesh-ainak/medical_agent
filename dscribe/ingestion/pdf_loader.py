"""Low-level PDF access: split into pages, pull the digital text layer, and
render a page to a PNG for vision OCR. No LLM calls happen here."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class RawPage:
    page_no: int          # 1-based
    text_layer: str       # text extracted by pdfplumber (may be empty)
    image_b64: str        # PNG render of the page, base64 (for vision OCR)


def _render_page_png_b64(doc: "fitz.Document", index: int, dpi: int) -> str:
    page = doc.load_page(index)
    pix = page.get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def load_pages(pdf_path: str, dpi: int = 180) -> list[RawPage]:
    """Return one RawPage per page: digital text + a rendered image."""
    pages: list[RawPage] = []

    # PyMuPDF for rendering, pdfplumber for the text layer. Open both once.
    with fitz.open(pdf_path) as fdoc, pdfplumber.open(pdf_path) as pdoc:
        for i, pplumber_page in enumerate(pdoc.pages):
            text = pplumber_page.extract_text() or ""
            image_b64 = _render_page_png_b64(fdoc, i, dpi)
            pages.append(
                RawPage(page_no=i + 1, text_layer=text.strip(), image_b64=image_b64)
            )
    return pages
