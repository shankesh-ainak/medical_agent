"""MOCKED external drug-interaction service (the brief explicitly allows mocking
external tools). It models a small known-interaction database and a flaky
network: the first call attempt may transiently fail, exercising the agent's
retry/fallback path (requirement #8). Deterministic given the same med set."""

from __future__ import annotations

import re

# Minimal illustrative interaction DB: normalised generic-name pairs -> note.
_INTERACTIONS: dict[frozenset[str], dict] = {
    frozenset({"ondansetron", "tramadol"}): {
        "severity": "moderate",
        "note": "Increased risk of serotonin syndrome and QT prolongation.",
    },
    frozenset({"meropenem", "valproate"}): {
        "severity": "major",
        "note": "Meropenem can sharply lower valproate levels, risking seizures.",
    },
    frozenset({"ofloxacin", "ondansetron"}): {
        "severity": "moderate",
        "note": "Additive QT-interval prolongation.",
    },
}

# Brand/abbrev -> generic, so the mock can match real chart names.
_BRAND_TO_GENERIC = {
    "emeset": "ondansetron",
    "zedott": "ondansetron",
    "oflox": "ofloxacin",
    "oflox tz": "ofloxacin",
    "raciper": "rabeprazole",
    "pan": "pantoprazole",
    "lopiramide": "loperamide",
    "meftal spas": "dicyclomine",
}

_PREFIXES = ("tab.", "tab", "inj.", "inj", "cap.", "cap", "inf.", "inf")
_seen_calls: set[frozenset[str]] = set()


def _generic(name: str) -> str:
    n = name.strip().lower()
    for p in _PREFIXES:
        if n.startswith(p + " "):
            n = n[len(p) + 1:]
    n = re.sub(r"\s+", " ", n).strip()
    return _BRAND_TO_GENERIC.get(n, n)


def check_interactions(medications: list[str], _simulate_flaky: bool = True) -> dict:
    """medications: list of drug names (brand or generic). Returns flagged pairs."""
    generics = sorted({_generic(m) for m in medications if m})
    key = frozenset(generics)

    # Simulate a transient outage on the first attempt for a given med set.
    if _simulate_flaky and key not in _seen_calls:
        _seen_calls.add(key)
        raise ConnectionError("drug-interaction service timed out (transient)")

    found: list[dict] = []
    for i in range(len(generics)):
        for j in range(i + 1, len(generics)):
            pair = frozenset({generics[i], generics[j]})
            if pair in _INTERACTIONS:
                hit = _INTERACTIONS[pair]
                found.append({"drugs": sorted(pair), **hit})

    return {
        "status": "ok",
        "checked": generics,
        "interactions": found,
        "summary": (
            f"{len(found)} interaction(s) found — escalate each."
            if found else "No interactions found in the database."
        ),
    }
