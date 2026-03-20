"""Soft-warning dataset audit for filtered conversation records.

Runs after hard filters to surface dataset-level quality concerns:
  - Under-represented categories (by competency code)
  - Difficulty distribution skew
  - Missing evidence spans
  - Numeric claim density (temperatures and pressures)

These are advisory only — no records are rejected here.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .detectors import word_count as _word_count

# Default warning thresholds (not exposed in FilterConfig — audit-only).
COMPETENCY_WARN_THRESHOLD = 50
_COMPETENCY_WARN_THRESHOLD = COMPETENCY_WARN_THRESHOLD  # backward-compat alias
_INTERMEDIATE_WARN_THRESHOLD = 0.80

_TEMP_RE = re.compile(r"\d+\s*°\s*[FC]", re.IGNORECASE)
_PRESSURE_RE = re.compile(r"\d+\s*(?:psi|psig|kpa|bar)\b", re.IGNORECASE)


def _get_all_assistant_text(rec: dict[str, Any]) -> str:
    """Extract all assistant response text from a record.

    Works for both ``"multi_turn"`` (concatenates all assistant turn contents)
    and ``"single_turn"`` (returns the ``"answer"`` field) records.
    """
    if rec.get("type") == "multi_turn":
        return " ".join(
            t["content"] for t in rec.get("conversation", []) if t.get("role") == "assistant"
        )
    return rec.get("answer", "")


def audit_dataset(
    kept: list[dict[str, Any]],
    *,
    strict_evidence: bool = False,
    competency_warn_threshold: int = _COMPETENCY_WARN_THRESHOLD,
    intermediate_warn_threshold: float = _INTERMEDIATE_WARN_THRESHOLD,
) -> dict[str, Any]:
    """Run soft audit checks on a set of kept records.

    No records are modified or removed.  All findings are advisory.

    Args:
        kept: Records that have already passed the hard filters.
        strict_evidence: When True, individual records missing evidence spans
            are surfaced in the ``"missing_evidence"`` warning list.  When
            False (default), only the aggregate count is tracked.
        competency_warn_threshold: Competency codes represented by fewer than
            this many records are flagged under ``"low_competency_coverage"``.
        intermediate_warn_threshold: If the fraction of ``"intermediate"``
            difficulty records exceeds this value a ``"difficulty_skew"``
            warning is emitted.

    Returns:
        A dict with the following keys:

        - ``"warnings"`` — dict mapping warning category to list of items
        - ``"competency_counts"`` — ordered dict of code → count
        - ``"difficulty_counts"`` — dict of difficulty level → count
        - ``"evidence_missing"`` — int, records without evidence spans
        - ``"temp_claims"`` — int, temperature expressions found
        - ``"pressure_claims"`` — int, pressure expressions found
    """
    warnings: dict[str, list[str]] = defaultdict(list)

    competency_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    evidence_missing = 0
    temp_claims = 0
    pressure_claims = 0

    for rec in kept:
        qa_id = rec.get("qa_id", "?")
        comp = rec.get("competency_code", "")
        diff = rec.get("difficulty", "")
        competency_counts[comp] += 1
        difficulty_counts[diff] += 1

        spans = rec.get("evidence_spans", [])
        status = rec.get("review_status", "")
        if status == "needs_evidence" or not spans:
            evidence_missing += 1
            if strict_evidence:
                warnings["missing_evidence"].append(qa_id)

        all_asst_text = _get_all_assistant_text(rec)
        temp_claims += len(_TEMP_RE.findall(all_asst_text))
        pressure_claims += len(_PRESSURE_RE.findall(all_asst_text))

    for comp, count in competency_counts.items():
        if count < competency_warn_threshold:
            warnings["low_competency_coverage"].append(f"{comp}={count}")

    total = sum(difficulty_counts.values())
    if total > 0:
        intermediate_frac = difficulty_counts.get("intermediate", 0) / total
        if intermediate_frac > intermediate_warn_threshold:
            warnings["difficulty_skew"].append(
                f"intermediate={intermediate_frac:.1%} (>{intermediate_warn_threshold:.0%})"
            )

    return {
        "warnings": dict(warnings),
        "competency_counts": dict(competency_counts.most_common()),
        "difficulty_counts": dict(difficulty_counts),
        "evidence_missing": evidence_missing,
        "temp_claims": temp_claims,
        "pressure_claims": pressure_claims,
    }
