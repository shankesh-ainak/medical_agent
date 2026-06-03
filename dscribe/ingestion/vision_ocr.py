"""Vision OCR module — transcribes a rendered page image (typed *or*
handwritten) into plain text using a vision-capable model, and classifies the
document type. Used for pages with no usable digital text layer."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from ..config import CONFIG
from ..robustness import with_retry

DOC_TYPES = [
    "admission_note",
    "progress_note",
    "nursing_note",
    "er_chart",
    "lab_report",
    "medication_record",
    "discharge_note",
    "other",
]

_SYSTEM = (
    "You are a careful medical-record transcriptionist. You are given an image "
    "of a single page from a patient's hospital chart. The page may be typed or "
    "handwritten, and may contain tables and forms.\n"
    "Transcribe EVERYTHING you can read, verbatim, preserving table structure as "
    "readable rows. Do NOT correct, expand, infer, or invent any clinical value. "
    "If a cell or word is illegible, write [illegible] — never guess a value.\n"
    "Return strict JSON with keys: 'text' (the transcription), 'doc_type' (one "
    f"of {DOC_TYPES}), and 'confidence' (one of 'high','medium','low' reflecting "
    "how legible the page was)."
)


@dataclass
class VisionResult:
    text: str
    doc_type: str
    confidence: str


def _client() -> OpenAI:
    return OpenAI(timeout=CONFIG.llm_timeout_s)


@with_retry
def transcribe_page(image_b64: str) -> VisionResult:
    """Transcribe one page image. Retries on transient failure; raises if it
    cannot succeed (caller converts that into a structured error)."""
    resp = _client().chat.completions.create(
        model=CONFIG.vision_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this chart page."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    doc_type = data.get("doc_type", "other")
    if doc_type not in DOC_TYPES:
        doc_type = "other"
    return VisionResult(
        text=(data.get("text") or "").strip(),
        doc_type=doc_type,
        confidence=data.get("confidence", "low"),
    )
