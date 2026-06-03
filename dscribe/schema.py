"""The discharge-summary data model — and the no-fabrication contract.

Every clinical fact in the output is a `Field`. A field is only allowed to hold
a real value if it carries at least one `SourceRef` (a page citation). The
guarded draft tools (tools/draft.py) enforce this: a value without a source is
rejected, so the agent's only honest alternatives are MISSING / PENDING /
CONFLICT. Fabrication is therefore structurally impossible, not merely
discouraged by the prompt.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field as PydField


class FieldStatus(str, Enum):
    PRESENT = "present"   # a value sourced from the documents
    MISSING = "missing"   # not found anywhere in the source notes
    PENDING = "pending"   # ordered/awaited but no result yet (e.g. culture)
    CONFLICT = "conflict"  # sources disagree; both kept, neither chosen


class SourceRef(BaseModel):
    """Provenance for a single fact. `page_no` is 1-based within the bundle."""

    page_no: int
    doc_type: str = "unknown"
    quote: str | None = None  # short verbatim snippet supporting the value


class Field(BaseModel):
    """One section value plus its status and provenance."""

    status: FieldStatus = FieldStatus.MISSING
    value: str | None = None
    sources: list[SourceRef] = PydField(default_factory=list)
    # For CONFLICT: the competing values, each with their own source(s).
    alternatives: list["ConflictOption"] = PydField(default_factory=list)
    note: str | None = None

    def is_resolved(self) -> bool:
        """True once the agent has taken an explicit stance on this field."""
        return not (self.status == FieldStatus.MISSING and self.value is None
                    and not self.sources and self.note is None)


class ConflictOption(BaseModel):
    value: str
    sources: list[SourceRef] = PydField(default_factory=list)


class ChangeType(str, Enum):
    ADDED = "added"          # on discharge list, not on admission list
    STOPPED = "stopped"      # on admission list, not on discharge list
    CHANGED = "changed"      # present on both, dose/frequency differs
    CONTINUED = "continued"  # present on both, unchanged


class MedicationChange(BaseModel):
    name: str
    admission: str | None = None   # e.g. "40MG 1-0-0" or None if not on admission
    discharge: str | None = None
    change_type: ChangeType
    reason: str | None = None
    needs_reconciliation: bool = False  # change with no documented reason
    sources: list[SourceRef] = PydField(default_factory=list)


class ReviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ReviewFlag(BaseModel):
    topic: str
    reason: str
    severity: ReviewSeverity = ReviewSeverity.WARNING


# The required sections from the assignment brief, in display order.
TEXT_SECTIONS: list[str] = [
    "patient_demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "secondary_diagnoses",
    "hospital_course",
    "procedures",
    "allergies",
    "follow_up_instructions",
    "pending_results",
    "discharge_condition",
]


class DischargeSummary(BaseModel):
    """The structured draft. Discharge meds and review flags are modelled
    explicitly; everything else is a provenance-tagged text Field."""

    fields: dict[str, Field] = PydField(
        default_factory=lambda: {name: Field() for name in TEXT_SECTIONS}
    )
    discharge_medications: list[MedicationChange] = PydField(default_factory=list)
    review_flags: list[ReviewFlag] = PydField(default_factory=list)
    finalized: bool = False


Field.model_rebuild()
