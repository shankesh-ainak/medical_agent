"""The guarded draft writers — where "no fabrication" is enforced.

`record_field` REJECTS any value that does not cite at least one real source
page. The agent therefore cannot write a clinical fact it can't anchor; its only
honest alternatives are mark_missing / mark_pending / record_conflict.
"""

from __future__ import annotations

from typing import Callable

from ..schema import (
    ChangeType,
    ConflictOption,
    DischargeSummary,
    Field,
    FieldStatus,
    MedicationChange,
    ReviewFlag,
    ReviewSeverity,
    SourceRef,
    TEXT_SECTIONS,
)


class DraftRejected(Exception):
    """Raised when a write violates the no-fabrication contract."""


class DraftStore:
    def __init__(self, page_exists: Callable[[int], bool]):
        self.summary = DischargeSummary()
        self._page_exists = page_exists

    # ---- internal validation ----
    def _validate_sources(self, sources: list[dict]) -> list[SourceRef]:
        if not sources:
            raise DraftRejected(
                "A value requires at least one source page. If you cannot cite "
                "the documents, call mark_missing or mark_pending instead."
            )
        refs: list[SourceRef] = []
        for s in sources:
            page = int(s["page_no"])
            if not self._page_exists(page):
                raise DraftRejected(
                    f"page_no {page} does not exist in this patient's bundle. "
                    "Cite only real source pages."
                )
            refs.append(SourceRef(page_no=page, doc_type=s.get("doc_type", "unknown"),
                                  quote=s.get("quote")))
        return refs

    def _check_section(self, section: str) -> None:
        if section not in TEXT_SECTIONS:
            raise DraftRejected(
                f"Unknown section '{section}'. Valid: {TEXT_SECTIONS}"
            )

    # ---- guarded writers ----
    def record_field(self, section: str, value: str, sources: list[dict]) -> dict:
        self._check_section(section)
        refs = self._validate_sources(sources)  # raises if uncited/invalid
        self.summary.fields[section] = Field(
            status=FieldStatus.PRESENT, value=value, sources=refs
        )
        return {"status": "ok", "section": section,
                "cited_pages": [r.page_no for r in refs]}

    def mark_missing(self, section: str, note: str | None = None) -> dict:
        self._check_section(section)
        self.summary.fields[section] = Field(status=FieldStatus.MISSING, note=note)
        return {"status": "ok", "section": section, "field_status": "missing"}

    def mark_pending(self, section: str, note: str, sources: list[dict] | None = None) -> dict:
        self._check_section(section)
        refs = self._validate_sources(sources) if sources else []
        self.summary.fields[section] = Field(
            status=FieldStatus.PENDING, note=note, sources=refs
        )
        return {"status": "ok", "section": section, "field_status": "pending"}

    def record_conflict(self, section: str, options: list[dict], note: str | None = None) -> dict:
        """options: [{value, sources:[...]}, ...] — both kept, neither chosen."""
        self._check_section(section)
        if len(options) < 2:
            raise DraftRejected("A conflict needs at least two competing options.")
        parsed = [
            ConflictOption(value=o["value"], sources=self._validate_sources(o["sources"]))
            for o in options
        ]
        self.summary.fields[section] = Field(
            status=FieldStatus.CONFLICT, alternatives=parsed, note=note
        )
        return {"status": "ok", "section": section, "field_status": "conflict",
                "options": len(parsed)}

    def set_discharge_medications(self, medications: list[dict]) -> dict:
        meds: list[MedicationChange] = []
        for m in medications:
            sources = self._validate_sources(m.get("sources", []))
            meds.append(MedicationChange(
                name=m["name"],
                admission=m.get("admission"),
                discharge=m.get("discharge"),
                change_type=ChangeType(m["change_type"]),
                reason=m.get("reason"),
                needs_reconciliation=bool(m.get("needs_reconciliation", False)),
                sources=sources,
            ))
        self.summary.discharge_medications = meds
        return {"status": "ok", "count": len(meds),
                "flagged": sum(1 for m in meds if m.needs_reconciliation)}

    def add_review_flag(self, topic: str, reason: str, severity: str = "warning") -> dict:
        self.summary.review_flags.append(ReviewFlag(
            topic=topic, reason=reason, severity=ReviewSeverity(severity)
        ))
        return {"status": "ok", "total_flags": len(self.summary.review_flags)}

    def finalize(self) -> dict:
        """Close the draft. Any section the agent never addressed is auto-marked
        MISSING so the output is structurally complete and never silently blank."""
        unaddressed = [
            name for name, f in self.summary.fields.items() if not f.is_resolved()
        ]
        for name in unaddressed:
            self.summary.fields[name] = Field(
                status=FieldStatus.MISSING,
                note="Not addressed by agent before finalize; auto-flagged.",
            )
        self.summary.finalized = True
        return {"status": "ok", "auto_marked_missing": unaddressed,
                "review_flags": len(self.summary.review_flags)}
