"""Deterministic conflict detection. Given the same field observed on multiple
pages, decide whether the sources disagree. It never picks a winner — it reports
the distinct values and where each came from (requirement #6)."""

from __future__ import annotations

import re


def _norm(value: str) -> str:
    """Light normalisation: lowercase, strip punctuation, and collapse trailing
    zeros (128.00 -> 128, while 1.65 is preserved) so pure formatting noise
    doesn't trigger a conflict. It deliberately does NOT normalise units or fix
    typos — when values still differ after this, the tool flags a conflict and
    lets the clinician decide. For a safety tool, a false-positive flag is the
    safe failure mode; a false negative (silently merging real disagreement) is
    not."""
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9.]+", " ", v)              # drop punctuation/unit slashes
    v = re.sub(r"(\d+)\.0+(?=\D|$)", r"\1", v)       # 128.00 -> 128; 1.65 stays
    return re.sub(r"\s+", " ", v).strip()


def detect(field: str, observations: list[dict]) -> dict:
    """observations: [{value, page_no, doc_type?}, ...].
    Groups by normalised value; >1 group => conflict."""
    groups: dict[str, dict] = {}
    for obs in observations:
        key = _norm(str(obs["value"]))
        g = groups.setdefault(key, {"value": obs["value"], "sources": []})
        g["sources"].append({
            "page_no": int(obs["page_no"]),
            "doc_type": obs.get("doc_type", "unknown"),
            "quote": obs.get("quote"),
        })

    options = list(groups.values())
    is_conflict = len(options) > 1
    return {
        "status": "ok",
        "field": field,
        "is_conflict": is_conflict,
        "options": options,
        "summary": (
            f"CONFLICT on '{field}': {len(options)} disagreeing values across "
            f"sources — escalate, do not choose."
            if is_conflict else
            f"No conflict on '{field}': all sources agree."
        ),
    }
