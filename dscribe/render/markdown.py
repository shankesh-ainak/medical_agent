"""DischargeSummary -> Markdown. Honest by construction: MISSING / PENDING /
CONFLICT are shown explicitly, every value carries its source pages, and all
escalations are collected into a Clinician Review section."""

from __future__ import annotations

from ..schema import DischargeSummary, Field, FieldStatus

_TITLES = {
    "patient_demographics": "Patient Demographics",
    "admission_date": "Admission Date",
    "discharge_date": "Discharge Date",
    "principal_diagnosis": "Principal Diagnosis",
    "secondary_diagnoses": "Secondary Diagnoses",
    "hospital_course": "Hospital Course",
    "procedures": "Procedures",
    "allergies": "Allergies",
    "follow_up_instructions": "Follow-up Instructions",
    "pending_results": "Pending Results",
    "discharge_condition": "Discharge Condition",
}


def _src(sources) -> str:
    if not sources:
        return ""
    return " " + ", ".join(f"[p{s.page_no}]" for s in sources)


def _render_field(f: Field) -> str:
    if f.status == FieldStatus.PRESENT:
        return f"{f.value}{_src(f.sources)}"
    if f.status == FieldStatus.MISSING:
        return f"**[MISSING]**{(' — ' + f.note) if f.note else ''}"
    if f.status == FieldStatus.PENDING:
        return f"**[PENDING]** {f.note or ''}{_src(f.sources)}"
    if f.status == FieldStatus.CONFLICT:
        lines = ["**[CONFLICT — needs clinician decision]**"]
        for opt in f.alternatives:
            lines.append(f"  - {opt.value}{_src(opt.sources)}")
        if f.note:
            lines.append(f"  - _note: {f.note}_")
        return "\n".join(lines)
    return "**[MISSING]**"


def render_markdown(summary: DischargeSummary) -> str:
    out: list[str] = ["# Discharge Summary — DRAFT (for clinician review)\n"]
    out.append("> Auto-generated draft. Not a finalized clinical document. "
               "Every value is traceable to a source page; gaps are flagged, "
               "not filled.\n")

    for key, title in _TITLES.items():
        f = summary.fields.get(key, Field())
        out.append(f"## {title}\n\n{_render_field(f)}\n")

    # Medications
    out.append("## Discharge Medications (changes from admission)\n")
    if not summary.discharge_medications:
        out.append("**[MISSING]** — no discharge medication list recorded.\n")
    else:
        out.append("| Medication | Admission | Discharge | Change | Reason | Flag | Source |")
        out.append("|---|---|---|---|---|---|---|")
        for m in summary.discharge_medications:
            flag = "RECONCILE" if m.needs_reconciliation else ""
            src = ", ".join(f"p{s.page_no}" for s in m.sources)
            out.append(
                f"| {m.name} | {m.admission or '—'} | {m.discharge or '—'} | "
                f"{m.change_type.value} | {m.reason or '—'} | {flag} | {src} |"
            )
        out.append("")

    # Clinician review
    out.append("## ⚑ Clinician Review Required\n")
    if not summary.review_flags:
        out.append("_No items explicitly escalated._\n")
    else:
        for fl in summary.review_flags:
            out.append(f"- **[{fl.severity.value.upper()}] {fl.topic}** — {fl.reason}")
        out.append("")

    return "\n".join(out)
