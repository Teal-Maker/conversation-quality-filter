"""Text analysis detectors for conversation quality filtering.

Provides low-level detection primitives:
  - Text normalization and measurement helpers
  - Metadata leakage detection (rubric/field-name bleed)
  - Trailing ellipsis truncation detection
  - Fuzzy similarity helpers (rapidfuzz with difflib fallback)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as rf_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    rf_fuzz = None  # type: ignore[assignment]
    HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+(?:[-/]\w+)*\b")


def word_count(text: str) -> int:
    """Count words, handling hyphenated and slash-separated compound terms."""
    return len(_WORD_RE.findall(text))


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, and strip markdown emphasis markers."""
    t = text.lower()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)  # strip bold
    t = re.sub(r"\*([^*]+)\*", r"\1", t)  # strip italic
    t = re.sub(r"\s+", " ", t).strip()
    return t


def contains_bold_term(text: str) -> bool:
    """Return True if *text* contains at least one ``**bold**`` markdown term."""
    return bool(re.search(r"\*\*[^*]{2,}\*\*", text))


def contains_numbered_procedure(text: str) -> bool:
    """Return True if *text* contains a numbered-step procedure (1. ... 2. ...).

    Year-like numbers (> 50) are excluded to avoid false positives on dates.
    """
    steps = re.findall(r"(?:^|\n)\s*(\d+)\.", text)
    if len(steps) < 2:
        return False
    nums = [int(s) for s in steps[:5] if int(s) < 50]
    if len(nums) < 2:
        return False
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Metadata leakage patterns
# ---------------------------------------------------------------------------

# Field names that should never appear verbatim in a training answer.
# A single match is sufficient to reject the record (auto-fail).
_AUTOFAIL_RUBRIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcompetency_code\b", re.IGNORECASE),
    re.compile(r"\breview_status\b", re.IGNORECASE),
    re.compile(r"\btrust_tier\b", re.IGNORECASE),
    re.compile(r"\bevidence_quote\b", re.IGNORECASE),
    re.compile(r"\bqa_complexity\b", re.IGNORECASE),
]

# Softer rubric phrases — each match counts as one signal; >= 2 signals trigger
# rejection so that isolated coincidental matches are not penalised.
_RUBRIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdoes not require\b", re.IGNORECASE),
    re.compile(r"\bdifficulty:\s*(?:beginner|intermediate|advanced)\b", re.IGNORECASE),
    re.compile(r"\btopic:\s*\S", re.IGNORECASE),
]


def _count_sentences(text: str) -> int:
    """Approximate sentence count, ignoring numbered-list markers (``1. ``)."""
    count = 0
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        if text[pos] == "." and pos > 0 and text[pos - 1].isdigit():
            continue
        count += 1
    return count


def _is_fragmentary(text: str) -> bool:
    """Return True if *text* looks like keyword fragments rather than prose."""
    stripped = text.strip()
    if not stripped:
        return True
    starts_lower = stripped[0].islower()
    open_p = stripped.count("(")
    close_p = stripped.count(")")
    unmatched_parens = abs(open_p - close_p) > 1
    commas = stripped.count(",")
    periods = stripped.count(".")
    keyword_list = commas > 3 and periods < 2 and word_count(stripped) < 30
    return sum([starts_lower, unmatched_parens, keyword_list]) >= 2


def detect_metadata_leakage(
    text: str,
    *,
    autofail_patterns: list[re.Pattern[str]] | None = None,
    rubric_patterns: list[re.Pattern[str]] | None = None,
) -> tuple[bool, str]:
    """Detect metadata leakage in *text* using a multi-signal approach.

    Auto-fails on structural field names (e.g. ``competency_code``,
    ``review_status``) that should never appear in model training outputs.
    For softer rubric phrases, requires at least two independent signals
    (including low sentence density or fragmentary syntax) before rejecting.

    Args:
        text: The assistant turn or answer text to inspect.
        autofail_patterns: Override the default auto-fail field patterns.
            Pass an empty list to disable auto-fail checks entirely.
        rubric_patterns: Override the default softer rubric patterns.

    Returns:
        ``(leaked, reason)`` — *leaked* is True when the text should be
        rejected; *reason* is a human-readable explanation string.
    """
    af_patterns = autofail_patterns if autofail_patterns is not None else _AUTOFAIL_RUBRIC_PATTERNS
    rb_patterns = rubric_patterns if rubric_patterns is not None else _RUBRIC_PATTERNS

    for pat in af_patterns:
        m = pat.search(text)
        if m:
            return True, f"autofail_field: {m.group()}"

    signals: list[str] = []
    for pat in rb_patterns:
        if pat.search(text):
            signals.append(f"rubric:{pat.pattern}")

    wc = word_count(text)
    sc = _count_sentences(text)
    if wc > 10 and sc < 2:
        signals.append("low_sentence_density")

    if _is_fragmentary(text):
        signals.append("fragmentary_syntax")

    if len(signals) >= 2:
        return True, "; ".join(signals)
    return False, ""


# ---------------------------------------------------------------------------
# Trailing ellipsis detector
# ---------------------------------------------------------------------------


def detect_trailing_ellipsis(text: str) -> bool:
    """Return True if *text* appears truncated via a trailing ellipsis.

    An ellipsis that follows a grammatically complete sentence (terminal
    punctuation before ``...``) is not considered truncation.
    """
    stripped = text.rstrip()
    if not stripped.endswith("..."):
        return False
    before_ellipsis = stripped.rstrip(".").rstrip()
    if before_ellipsis and before_ellipsis[-1] in ".!?":
        return False
    return True


# ---------------------------------------------------------------------------
# Fuzzy / Jaccard similarity helpers
# ---------------------------------------------------------------------------


def token_set(text: str) -> set[str]:
    """Return the set of whitespace-separated tokens from normalised *text*."""
    return set(normalize_text(text).split())


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets (0.0 – 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fuzzy_token_set_ratio(a: str, b: str) -> float:
    """Token-set ratio in the range 0–100.

    Uses rapidfuzz when available, otherwise falls back to difflib.
    Token-set ratio is superset-aware (a string that contains another
    scores 100), making it appropriate for repetition detection.
    """
    if HAS_RAPIDFUZZ:
        return rf_fuzz.token_set_ratio(a, b)  # type: ignore[union-attr]
    a_tokens = " ".join(sorted(normalize_text(a).split()))
    b_tokens = " ".join(sorted(normalize_text(b).split()))
    return SequenceMatcher(None, a_tokens, b_tokens).ratio() * 100


def fuzzy_ratio(a: str, b: str) -> float:
    """Sequence ratio in the range 0–100.

    Uses rapidfuzz when available, otherwise falls back to difflib.
    Unlike token-set ratio, additive text is penalised, making this
    appropriate for near-duplicate detection.
    """
    if HAS_RAPIDFUZZ:
        return rf_fuzz.ratio(a, b)  # type: ignore[union-attr]
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio() * 100
