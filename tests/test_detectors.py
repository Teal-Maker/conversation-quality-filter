"""Tests for detectors.py — text helpers and detection primitives."""

from __future__ import annotations

import re

import pytest

from conversation_quality_filter.detectors import (
    HAS_RAPIDFUZZ,
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
# word_count
# ---------------------------------------------------------------------------


class TestWordCount:
    def test_simple_sentence(self) -> None:
        assert word_count("The cat sat on the mat") == 6

    def test_hyphenated_compound(self) -> None:
        # "R-410A" should count as one token
        assert word_count("The refrigerant is R-410A") == 4

    def test_slash_compound(self) -> None:
        assert word_count("check/verify the pressure") == 3

    def test_empty_string(self) -> None:
        assert word_count("") == 0

    def test_whitespace_only(self) -> None:
        assert word_count("   \t\n  ") == 0

    def test_numbers_count(self) -> None:
        assert word_count("12 psi at 75 degrees") == 5


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercases(self) -> None:
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_strips_bold(self) -> None:
        assert normalize_text("**Superheat** is important") == "superheat is important"

    def test_strips_italic(self) -> None:
        assert normalize_text("*italic* text") == "italic text"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("too   many   spaces") == "too many spaces"

    def test_strips_leading_trailing(self) -> None:
        assert normalize_text("  padded  ") == "padded"


# ---------------------------------------------------------------------------
# contains_bold_term
# ---------------------------------------------------------------------------


class TestContainsBoldTerm:
    def test_has_bold(self) -> None:
        assert contains_bold_term("The **TXV** controls flow.")

    def test_no_bold(self) -> None:
        assert not contains_bold_term("Plain text without formatting.")

    def test_single_star_not_bold(self) -> None:
        assert not contains_bold_term("*italic* only")

    def test_minimum_length_two(self) -> None:
        # Pattern requires at least 2 characters inside the asterisks
        assert not contains_bold_term("**x**")
        assert contains_bold_term("**OK**")


# ---------------------------------------------------------------------------
# contains_numbered_procedure
# ---------------------------------------------------------------------------


class TestContainsNumberedProcedure:
    def test_detects_consecutive_steps(self) -> None:
        text = "Do the following:\n1. Turn off power.\n2. Remove panel."
        assert contains_numbered_procedure(text)

    def test_requires_consecutive(self) -> None:
        text = "See steps:\n1. Step one.\n3. Step three."
        assert not contains_numbered_procedure(text)

    def test_ignores_year_numbers(self) -> None:
        text = "Installed in 2023. Updated in 2024."
        assert not contains_numbered_procedure(text)

    def test_requires_at_least_two_steps(self) -> None:
        text = "Step:\n1. Do this."
        assert not contains_numbered_procedure(text)

    def test_five_steps(self) -> None:
        text = "Steps:\n1. A\n2. B\n3. C\n4. D\n5. E"
        assert contains_numbered_procedure(text)


# ---------------------------------------------------------------------------
# detect_metadata_leakage
# ---------------------------------------------------------------------------


class TestDetectMetadataLeakage:
    def test_autofail_competency_code(self) -> None:
        leaked, reason = detect_metadata_leakage("The competency_code for this topic is REF-101.")
        assert leaked
        assert "autofail_field" in reason

    def test_autofail_review_status(self) -> None:
        leaked, reason = detect_metadata_leakage("Set review_status to approved.")
        assert leaked

    def test_autofail_trust_tier(self) -> None:
        leaked, reason = detect_metadata_leakage("trust_tier must be verified.")
        assert leaked

    def test_autofail_evidence_quote(self) -> None:
        leaked, reason = detect_metadata_leakage("Include the evidence_quote field.")
        assert leaked

    def test_autofail_qa_complexity(self) -> None:
        leaked, reason = detect_metadata_leakage("Set qa_complexity to high.")
        assert leaked

    def test_rubric_two_signals(self) -> None:
        # Two rubric patterns together should trigger rejection
        text = "difficulty: intermediate. topic: refrigerants"
        leaked, reason = detect_metadata_leakage(text)
        assert leaked

    def test_rubric_single_signal_with_fragment(self) -> None:
        # One rubric pattern + fragmentary syntax = 2 signals.
        # Need >3 commas AND <30 words to trigger keyword_list check,
        # plus starts lowercase for fragmentary detection (sum >= 2).
        text = "does not require advanced knowledge, basics, fundamentals, concepts, overview, intro"
        leaked, _ = detect_metadata_leakage(text)
        assert leaked

    def test_clean_text_passes(self) -> None:
        text = (
            "Superheat is measured by subtracting the saturation temperature from the actual "
            "suction line temperature at the evaporator outlet. A reading of 10 to 15 degrees "
            "is typical for most residential systems."
        )
        leaked, _ = detect_metadata_leakage(text)
        assert not leaked

    def test_custom_autofail_patterns(self) -> None:
        custom = [re.compile(r"\bcustom_field\b", re.IGNORECASE)]
        leaked, reason = detect_metadata_leakage("The custom_field is present.", autofail_patterns=custom)
        assert leaked
        # Default autofail patterns should NOT trigger when overridden
        leaked2, _ = detect_metadata_leakage("competency_code here", autofail_patterns=custom)
        assert not leaked2

    def test_empty_autofail_patterns_disables_autofail(self) -> None:
        leaked, _ = detect_metadata_leakage("competency_code present", autofail_patterns=[])
        # With no autofail patterns, only rubric signals matter
        assert not leaked

    def test_low_sentence_density_plus_rubric(self) -> None:
        # One rubric match + very long text with no sentences = low_sentence_density signal
        text = "topic: refrigerants " + "word " * 15
        leaked, _ = detect_metadata_leakage(text)
        assert leaked


# ---------------------------------------------------------------------------
# detect_trailing_ellipsis
# ---------------------------------------------------------------------------


class TestDetectTrailingEllipsis:
    def test_detects_trailing_ellipsis(self) -> None:
        assert detect_trailing_ellipsis("The technician should check the filter...")

    def test_ellipsis_after_complete_sentence_ok(self) -> None:
        # Ellipsis after terminal punctuation is NOT truncation
        assert not detect_trailing_ellipsis("The job is done. ...")

    def test_no_ellipsis(self) -> None:
        assert not detect_trailing_ellipsis("This is a complete sentence.")

    def test_two_dots_not_ellipsis(self) -> None:
        assert not detect_trailing_ellipsis("See page 3..")

    def test_empty_string(self) -> None:
        assert not detect_trailing_ellipsis("")

    def test_trailing_whitespace_ignored(self) -> None:
        assert detect_trailing_ellipsis("truncated text...   ")


# ---------------------------------------------------------------------------
# Fuzzy / Jaccard helpers
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self) -> None:
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self) -> None:
        score = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.0 < score < 1.0

    def test_both_empty(self) -> None:
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert jaccard_similarity(set(), {"a"}) == 0.0


class TestFuzzyRatio:
    def test_identical_strings(self) -> None:
        assert fuzzy_ratio("hello world", "hello world") == pytest.approx(100.0, abs=1.0)

    def test_completely_different(self) -> None:
        assert fuzzy_ratio("aaa", "zzz") < 50.0

    def test_superset_penalised(self) -> None:
        # fuzzy_ratio should penalise the longer superset string
        short = "the cat sat"
        long_text = "the cat sat on the mat with a hat and a bat"
        score = fuzzy_ratio(short, long_text)
        assert score < 80.0


class TestFuzzyTokenSetRatio:
    def test_identical(self) -> None:
        assert fuzzy_token_set_ratio("hello world", "hello world") == pytest.approx(100.0, abs=1.0)

    def test_superset_scores_high_with_rapidfuzz(self) -> None:
        # rapidfuzz token_set_ratio is superset-aware (short string contained in long → ~100).
        # The difflib fallback sorts tokens and uses SequenceMatcher so it does NOT guarantee
        # high scores for supersets — this behaviour difference is intentional and documented.
        if not HAS_RAPIDFUZZ:
            pytest.skip("Superset-aware scoring requires rapidfuzz")
        short = "cat sat"
        long_text = "the cat sat on a mat"
        score = fuzzy_token_set_ratio(short, long_text)
        assert score >= 95.0

    def test_unrelated_strings(self) -> None:
        assert fuzzy_token_set_ratio("alpha beta gamma", "delta epsilon zeta") < 50.0


class TestTokenSet:
    def test_basic(self) -> None:
        assert token_set("Hello WORLD") == {"hello", "world"}

    def test_strips_bold(self) -> None:
        assert token_set("**Superheat** is key") == {"superheat", "is", "key"}

    def test_empty(self) -> None:
        assert token_set("") == set()
