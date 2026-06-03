"""Deterministic medication reconciliation. The agent supplies the two lists it
sourced from the documents; the diff itself is plain, auditable code — no LLM
judgement. A change with no documented reason is flagged for reconciliation
rather than silently resolved (requirement #5)."""

from __future__ import annotations

import re

_PREFIXES = ("tab.", "tab", "cap.", "cap", "inj.", "inj", "inf.", "inf",
             "syp.", "syp", "tablet")


def _norm_name(name: str) -> str:
    n = name.strip().lower()
    for p in _PREFIXES:
        if n.startswith(p + " "):
            n = n[len(p) + 1:]
    return re.sub(r"\s+", " ", n).strip()


def reconcile(admission: list[dict], discharge: list[dict]) -> dict:
    """admission/discharge: [{name, dose?, reason?}, ...].
    Returns per-drug change classification and a list needing reconciliation."""
    adm = {_norm_name(m["name"]): m for m in admission}
    dis = {_norm_name(m["name"]): m for m in discharge}

    changes: list[dict] = []
    for key in sorted(set(adm) | set(dis)):
        a, d = adm.get(key), dis.get(key)
        if a and not d:
            change_type, needs = "stopped", True
        elif d and not a:
            change_type, needs = "added", True
        else:
            a_dose = (a.get("dose") or "").strip().lower()
            d_dose = (d.get("dose") or "").strip().lower()
            if a_dose and d_dose and a_dose != d_dose:
                change_type, needs = "changed", True
            else:
                change_type, needs = "continued", False

        reason = (d or a or {}).get("reason")
        # A real change with a stated reason does not need reconciliation.
        if needs and reason:
            needs = False

        changes.append({
            "name": (d or a)["name"],
            "admission": (a or {}).get("dose"),
            "discharge": (d or {}).get("dose"),
            "change_type": change_type,
            "reason": reason,
            "needs_reconciliation": needs,
        })

    flagged = [c for c in changes if c["needs_reconciliation"]]
    return {
        "status": "ok",
        "changes": changes,
        "needs_reconciliation": flagged,
        "summary": (
            f"{len(changes)} medications compared; "
            f"{len(flagged)} change(s) need clinician reconciliation."
        ),
    }
