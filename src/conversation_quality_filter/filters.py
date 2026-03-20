"""Hard-filter logic for multi-turn and single-turn conversation records.

Public API
----------
- ``FilterConfig`` — dataclass holding all tunable thresholds
- ``FilterResult`` — dataclass returned by ``filter_records``
- ``filter_records(records, config)`` — top-level Python API
- ``filter_multi_turn(rec, config)`` — filter a single multi-turn record
- ``filter_single_turn(rec, config)`` — filter a single single-turn record
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .detectors import (
    contains_bold_term,
    contains_numbered_procedure,
    detect_metadata_leakage,
    detect_trailing_ellipsis,
    fuzzy_ratio,
    fuzzy_token_set_ratio,
    jaccard_similarity,
    normalize_text,
    token_set,
    word_count,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default domain-specific standards references used by the substantive-expansion
# check.  These are well-known standards bodies and codes; the list is a
# starting point and can be overridden via FilterConfig.standards_patterns.
_DEFAULT_STANDARDS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"EPA\s*608"),
    re.compile(r"ASHRAE"),
    re.compile(r"\bNEC\b"),
    re.compile(r"\bIMC\b"),
    re.compile(r"\bUMC\b"),
    re.compile(r"\bACCA\b"),
    re.compile(r"\bARI\b"),
    re.compile(r"\bAHRI\b"),
]


@dataclass
class FilterConfig:
    """All tunable thresholds for the quality filter pipeline.

    Attributes:
        min_assistant_words_multi: Minimum word count for any assistant turn
            in a multi-turn record.
        min_user_words_multi: Minimum word count for any user turn in a
            multi-turn record.
        single_turn_min_words: Minimum answer word count for single-turn
            records.
        single_turn_max_words: Maximum answer word count for single-turn
            records.
        assistant_repeat_jaccard: Jaccard similarity threshold above which
            adjacent assistant turns are considered repetitive (0.0–1.0).
        assistant_repeat_fuzzy: Fuzzy token-set ratio threshold (0–100) above
            which adjacent assistant turns are considered repetitive.
        duplicate_assistant_fuzzy: Fuzzy sequence ratio threshold (0–100)
            above which any two assistant turns in the same record are
            considered near-duplicates.
        standards_patterns: Compiled regex patterns used to detect references
            to domain-specific standards in the substantive-expansion check.
            Defaults to a built-in list of common standards bodies.
        practical_content_patterns: A single compiled regex pattern used to
            detect domain-specific code or standards references in the
            single-turn practical-content check.  Defaults to a built-in
            pattern covering EPA 608, ASHRAE, NEC, IMC, UMC, ACCA, ARI, AHRI,
            UL, and NFPA.
    """

    min_assistant_words_multi: int = 80
    min_user_words_multi: int = 10
    single_turn_min_words: int = 120
    single_turn_max_words: int = 500
    assistant_repeat_jaccard: float = 0.60
    assistant_repeat_fuzzy: float = 90.0
    duplicate_assistant_fuzzy: float = 98.0
    standards_patterns: list[re.Pattern[str]] | None = None
    practical_content_patterns: re.Pattern[str] | None = None

    def _resolved_standards_patterns(self) -> list[re.Pattern[str]]:
        """Return the effective standards patterns list."""
        return self.standards_patterns if self.standards_patterns is not None else _DEFAULT_STANDARDS_PATTERNS

    def _resolved_practical_content_pattern(self) -> re.Pattern[str]:
        """Return the effective practical-content code-reference pattern."""
        return (
            self.practical_content_patterns
            if self.practical_content_patterns is not None
            else _DEFAULT_CODE_REF_RE
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Output of ``filter_records``.

    Attributes:
        kept: Records that passed all hard filters.
        rejected: Records that failed at least one filter.  Each entry is a
            dict with keys ``qa_id``, ``record_type``, ``reason``,
            ``detail``, and ``source_content_id``.
    """

    kept: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_substantive_expansion(
    text_a: str,
    text_b: str,
    tokens_a: set[str],
    tokens_b: set[str],
    standards_patterns: list[re.Pattern[str]],
) -> bool:
    """Return True when turn B is a genuine expansion of turn A.

    Expansion is accepted only when turn B is at least 35% longer AND
    introduces either a new numbered procedure or a new domain-specific
    standards reference not present in turn A.

    High word-count growth alone is insufficient — it was too easy to
    satisfy with padding.
    """
    wc_a = word_count(text_a)
    wc_b = word_count(text_b)
    if wc_b < wc_a * 1.35:
        return False

    has_new_procedure = contains_numbered_procedure(text_b) and not contains_numbered_procedure(text_a)

    has_new_reference = any(
        bool(pat.search(text_b)) and not bool(pat.search(text_a))
        for pat in standards_patterns
    )

    return has_new_procedure or has_new_reference


# ---------------------------------------------------------------------------
# Per-record filters
# ---------------------------------------------------------------------------


def filter_multi_turn(
    rec: dict[str, Any],
    config: FilterConfig,
) -> tuple[bool, str, str]:
    """Apply hard filters to a single multi-turn record.

    Args:
        rec: A dict with at least a ``"conversation"`` key containing a list
            of ``{"role": ..., "content": ...}`` turn dicts.
        config: Threshold configuration.

    Returns:
        ``(keep, reason_code, detail)`` — *keep* is True when the record
        passes all checks.  On failure, *reason_code* is a short machine-
        readable label and *detail* is a human-readable explanation.
    """
    conversation = rec.get("conversation", [])
    try:
        assistant_turns = [(i, t) for i, t in enumerate(conversation) if t["role"] == "assistant"]
        user_turns = [(i, t) for i, t in enumerate(conversation) if t["role"] == "user"]
    except KeyError:
        return False, "invalid_record_schema", "turn missing required 'role' key"

    if not assistant_turns:
        return False, "no_assistant_turns", "conversation has no assistant responses"

    # -- First-turn quality gate -----------------------------------------
    try:
        first_asst = assistant_turns[0][1]["content"]
    except KeyError:
        return False, "invalid_record_schema", "assistant turn missing required 'content' key"

    leaked, detail = detect_metadata_leakage(first_asst)
    if leaked:
        return False, "first_turn_quality_failure", f"metadata_leakage in turn 0: {detail}"

    if detect_trailing_ellipsis(first_asst):
        return False, "first_turn_quality_failure", "trailing_ellipsis in turn 0"

    first_wc = word_count(first_asst)
    if first_wc < config.min_assistant_words_multi:
        return (
            False,
            "first_turn_quality_failure",
            f"turn 0 has {first_wc} words (min {config.min_assistant_words_multi})",
        )

    # -- Duplicate assistant turns ----------------------------------------
    # Uses fuzzy_ratio (not token-set ratio) so that superset expansions do
    # not score 100% — those are handled by the adjacent repetition check.
    try:
        normalized_asst = [normalize_text(t["content"]) for _, t in assistant_turns]
    except KeyError:
        return False, "invalid_record_schema", "assistant turn missing required 'content' key"
    for i in range(len(normalized_asst)):
        for j in range(i + 1, len(normalized_asst)):
            if normalized_asst[i] == normalized_asst[j]:
                return (
                    False,
                    "duplicate_assistant_turn",
                    f"turns {assistant_turns[i][0]} and {assistant_turns[j][0]} identical",
                )
            ratio = fuzzy_ratio(normalized_asst[i], normalized_asst[j])
            if ratio >= config.duplicate_assistant_fuzzy:
                return (
                    False,
                    "duplicate_assistant_turn",
                    f"turns {assistant_turns[i][0]} and {assistant_turns[j][0]} fuzzy={ratio:.0f}%",
                )

    # -- Assistant turn word count (turns after turn 0) -------------------
    for idx, turn in assistant_turns[1:]:
        wc = word_count(turn.get("content", ""))
        if wc < config.min_assistant_words_multi:
            return (
                False,
                "assistant_turn_too_short",
                f"turn {idx} has {wc} words (min {config.min_assistant_words_multi})",
            )

    # -- User turn word count ---------------------------------------------
    for idx, turn in user_turns:
        wc = word_count(turn.get("content", ""))
        if wc < config.min_user_words_multi:
            return (
                False,
                "user_turn_too_short",
                f"turn {idx} has {wc} words (min {config.min_user_words_multi})",
            )

    # -- Metadata leakage (remaining assistant turns) ---------------------
    for idx, turn in assistant_turns[1:]:
        leaked, detail = detect_metadata_leakage(turn.get("content", ""))
        if leaked:
            return False, "metadata_leakage", f"turn {idx}: {detail}"

    # -- Trailing ellipsis truncation (remaining assistant turns) ---------
    for idx, turn in assistant_turns[1:]:
        if detect_trailing_ellipsis(turn.get("content", "")):
            return (
                False,
                "trailing_ellipsis_truncation",
                f"turn {idx} ends with truncating ellipsis",
            )

    # -- Turn-to-turn repetition (adjacent assistant turns) ---------------
    standards = config._resolved_standards_patterns()
    for k in range(len(assistant_turns) - 1):
        idx_a, turn_a = assistant_turns[k]
        idx_b, turn_b = assistant_turns[k + 1]
        text_a = turn_a.get("content", "")
        text_b = turn_b.get("content", "")

        tokens_a = token_set(text_a)
        tokens_b = token_set(text_b)
        jacc = jaccard_similarity(tokens_a, tokens_b)

        if jacc >= config.assistant_repeat_jaccard:
            if _is_substantive_expansion(text_a, text_b, tokens_a, tokens_b, standards):
                continue
            return (
                False,
                "assistant_turn_repetition",
                f"turns {idx_a}->{idx_b} Jaccard={jacc:.2f}",
            )

        fuzz = fuzzy_token_set_ratio(text_a, text_b)
        if fuzz >= config.assistant_repeat_fuzzy:
            if _is_substantive_expansion(text_a, text_b, tokens_a, tokens_b, standards):
                continue
            return (
                False,
                "assistant_turn_repetition",
                f"turns {idx_a}->{idx_b} fuzzy={fuzz:.0f}%",
            )

    return True, "", ""


# Practical-content patterns for single-turn records.
_TROUBLESHOOTING_RE = re.compile(
    r"\b(?:check|measure|verify|inspect|test|diagnose|troubleshoot)\b",
    re.IGNORECASE,
)
_DEFAULT_CODE_REF_RE = re.compile(
    r"\b(?:EPA\s*608|ASHRAE|NEC|IMC|UMC|ACCA|ARI|AHRI|UL|NFPA)\b",
    re.IGNORECASE,
)
_FIELD_EXAMPLE_RE = re.compile(
    r"\b(?:for example|in the field|in practice|on the job|real[- ]world)\b",
    re.IGNORECASE,
)


def filter_single_turn(
    rec: dict[str, Any],
    config: FilterConfig,
) -> tuple[bool, str, str]:
    """Apply hard filters to a single single-turn record.

    Args:
        rec: A dict with at least an ``"answer"`` key.
        config: Threshold configuration.

    Returns:
        ``(keep, reason_code, detail)`` — *keep* is True when the record
        passes all checks.
    """
    answer = rec.get("answer", "")
    wc = word_count(answer)

    if wc < config.single_turn_min_words or wc > config.single_turn_max_words:
        return (
            False,
            "answer_word_count_out_of_range",
            f"{wc} words (range [{config.single_turn_min_words}, {config.single_turn_max_words}])",
        )

    if not contains_bold_term(answer):
        return False, "missing_bold_formatting", "no **bold** terms found"

    code_ref_re = config._resolved_practical_content_pattern()
    has_procedure = contains_numbered_procedure(answer)
    has_troubleshooting = bool(_TROUBLESHOOTING_RE.search(answer))
    has_code_ref = bool(code_ref_re.search(answer))
    has_field_example = bool(_FIELD_EXAMPLE_RE.search(answer))

    if not any([has_procedure, has_troubleshooting, has_code_ref, has_field_example]):
        return (
            False,
            "insufficient_practical_content",
            "no procedures, troubleshooting terms, code refs, or field examples",
        )

    leaked, detail = detect_metadata_leakage(answer)
    if leaked:
        return False, "metadata_leakage", detail

    if detect_trailing_ellipsis(answer):
        return False, "trailing_ellipsis_truncation", "answer ends with truncating ellipsis"

    return True, "", ""


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def filter_records(
    records: list[dict[str, Any]],
    config: FilterConfig | None = None,
) -> FilterResult:
    """Filter a list of conversation records in-process.

    Supports both ``"multi_turn"`` and ``"single_turn"`` record types as
    identified by the ``"type"`` field.  Records with unknown types are
    passed through without filtering.

    Args:
        records: List of record dicts, each containing at minimum a
            ``"type"`` field and either a ``"conversation"`` list (multi-turn)
            or an ``"answer"`` string (single-turn).
        config: Optional threshold configuration.  Defaults to
            ``FilterConfig()`` (all default thresholds).

    Returns:
        A ``FilterResult`` with ``kept`` and ``rejected`` lists.  Each
        rejected entry carries ``qa_id``, ``source_content_id``,
        ``record_type``, ``reason``, and ``detail`` keys.

    Example::

        from conversation_quality_filter import filter_records, FilterConfig

        config = FilterConfig(min_assistant_words_multi=60)
        result = filter_records(my_records, config)
        print(f"Kept {len(result.kept)} / {len(my_records)}")
    """
    if config is None:
        config = FilterConfig()

    result = FilterResult()

    for rec in records:
        qa_id = rec.get("qa_id", "")
        src_id = str(rec.get("source_content_id", ""))
        record_type = rec.get("type", "unknown")

        if record_type == "multi_turn":
            keep, reason, detail = filter_multi_turn(rec, config)
        elif record_type == "single_turn":
            keep, reason, detail = filter_single_turn(rec, config)
        else:
            result.kept.append(rec)
            continue

        if keep:
            result.kept.append(rec)
        else:
            result.rejected.append(
                {
                    "qa_id": qa_id,
                    "source_content_id": src_id,
                    "record_type": record_type,
                    "reason": reason,
                    "detail": detail,
                }
            )

    return result
