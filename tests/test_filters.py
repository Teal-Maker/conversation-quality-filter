"""Tests for filters.py — FilterConfig, FilterResult, and record-level filters."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import pytest

from conversation_quality_filter.filters import (
    FilterConfig,
    FilterResult,
    filter_multi_turn,
    filter_records,
    filter_single_turn,
)


# ---------------------------------------------------------------------------
# FilterConfig defaults
# ---------------------------------------------------------------------------


class TestFilterConfigDefaults:
    def test_default_min_assistant_words(self) -> None:
        cfg = FilterConfig()
        assert cfg.min_assistant_words_multi == 80

    def test_default_min_user_words(self) -> None:
        assert FilterConfig().min_user_words_multi == 10

    def test_default_single_min_words(self) -> None:
        assert FilterConfig().single_turn_min_words == 120

    def test_default_single_max_words(self) -> None:
        assert FilterConfig().single_turn_max_words == 500

    def test_default_standards_patterns_none(self) -> None:
        # None means "use built-in defaults"
        assert FilterConfig().standards_patterns is None

    def test_resolved_standards_patterns_not_empty(self) -> None:
        patterns = FilterConfig()._resolved_standards_patterns()
        assert len(patterns) > 0
        assert all(isinstance(p, re.Pattern) for p in patterns)

    def test_custom_standards_patterns(self) -> None:
        custom = [re.compile(r"ISO\s*9001")]
        cfg = FilterConfig(standards_patterns=custom)
        assert cfg._resolved_standards_patterns() == custom


# ---------------------------------------------------------------------------
# filter_multi_turn — passing cases
# ---------------------------------------------------------------------------


class TestFilterMultiTurnPass:
    def test_good_record_passes(self, good_multi_turn_record: dict[str, Any]) -> None:
        keep, reason, detail = filter_multi_turn(good_multi_turn_record, FilterConfig())
        assert keep is True
        assert reason == ""

    def test_single_turn_record_in_multi_filter(self, good_multi_turn_record: dict[str, Any]) -> None:
        # Even with one assistant turn, it should pass if quality is sufficient
        rec = deepcopy(good_multi_turn_record)
        rec["conversation"] = [
            {
                "role": "user",
                "content": "What is superheat and how is it correctly measured on a residential split system?",
            },
            {"role": "assistant", "content": rec["conversation"][1]["content"]},
        ]
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert keep is True


# ---------------------------------------------------------------------------
# filter_multi_turn — rejection cases
# ---------------------------------------------------------------------------


class TestFilterMultiTurnReject:
    def test_no_assistant_turns(self) -> None:
        rec = {
            "type": "multi_turn",
            "conversation": [{"role": "user", "content": "Hello?"}],
        }
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "no_assistant_turns"

    def test_first_turn_too_short(self) -> None:
        rec = {
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": "What is a capacitor used for in HVAC equipment?"},
                {"role": "assistant", "content": "It stores charge."},
            ],
        }
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "first_turn_quality_failure"

    def test_first_turn_trailing_ellipsis(self) -> None:
        short = "The capacitor stores electrical energy for motor starting purposes..."
        rec = {
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": "Explain capacitor function in detail please."},
                {"role": "assistant", "content": short + " " * 200 + "..."},
            ],
        }
        # Make it long enough for word count but end with ellipsis
        long_content = (
            "The capacitor stores electrical energy and provides the phase shift needed to start "
            "single-phase motors. When a motor fails to start or trips on overload, a weak or "
            "shorted capacitor is often the cause. Capacitors are rated in microfarads and voltage. "
            "The run capacitor maintains a phase-split after startup while the start capacitor "
            "provides additional torque during the starting cycle before being switched out..."
        )
        rec["conversation"][1]["content"] = long_content
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "first_turn_quality_failure"

    def test_first_turn_metadata_leakage(self) -> None:
        rec = {
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": "Tell me about competency tracking."},
                {"role": "assistant", "content": "The competency_code field tracks review_status and trust_tier."},
            ],
        }
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "first_turn_quality_failure"

    def test_duplicate_assistant_turns_identical(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        # Make both assistant turns identical
        first_asst = rec["conversation"][1]["content"]
        rec["conversation"][3]["content"] = first_asst
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "duplicate_assistant_turn"

    def test_duplicate_assistant_turns_fuzzy(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        first_asst = rec["conversation"][1]["content"]
        # Make second turn almost identical (98%+ similarity) with tiny change
        rec["conversation"][3]["content"] = first_asst[:-5] + "unit."
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "duplicate_assistant_turn"

    def test_subsequent_assistant_turn_too_short(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        rec["conversation"][3]["content"] = "Short answer."
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "assistant_turn_too_short"

    def test_user_turn_too_short(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        rec["conversation"][2]["content"] = "Why?"
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "user_turn_too_short"

    def test_subsequent_turn_metadata_leakage(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        # Inject an autofail field name into an otherwise long-enough turn
        leaked_content = (
            "The competency_code for this answer is REF-101 and review_status is pending. "
            "The system uses a heat pump which reverses the refrigerant flow using a four-way "
            "reversing valve. In heating mode the outdoor coil acts as the evaporator and the "
            "indoor coil acts as the condenser. The balance point is the outdoor temperature at "
            "which the heat pump capacity equals the building heating load. Below that point "
            "auxiliary electric resistance heat is staged in to maintain indoor comfort levels. "
            "Proper sizing of the auxiliary heat bank is important for energy efficiency."
        )
        rec["conversation"][3]["content"] = leaked_content
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "metadata_leakage"

    def test_subsequent_turn_trailing_ellipsis(self, good_multi_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_multi_turn_record)
        # Build a turn that is long enough to pass the word count check but ends with an ellipsis
        base = rec["conversation"][3]["content"]
        rec["conversation"][3]["content"] = base + "..."
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "trailing_ellipsis_truncation"

    def test_turn_repetition_high_jaccard(self) -> None:
        # Two adjacent turns with high Jaccard similarity (> 0.60 threshold).
        # Each turn must be >= 80 words to pass the word-count gate.
        # fuzzy_ratio must stay < 98 so the duplicate check does not fire first.
        # The turns share the same core vocabulary but differ structurally enough
        # that fuzzy_ratio < 98, while Jaccard stays above 0.60.
        similar = (
            "To measure superheat attach manifold gauges to the suction service port on the "
            "compressor and record suction pressure. Convert the measured pressure to saturation "
            "temperature using refrigerant pressure-temperature tables or a digital manifold. "
            "Measure the actual suction line temperature with a calibrated clamp thermometer "
            "near the evaporator outlet. Subtract the saturation temperature from the measured "
            "line temperature. The resulting value in degrees Fahrenheit is the superheat. "
            "Normal target values range from eight to twelve degrees for most residential systems. "
            "A high reading often indicates an undercharge or a restricted metering device."
        )
        repeat = (
            "When diagnosing a system, measuring superheat is an essential step. To do this you "
            "should attach manifold gauges to the suction service port on the compressor and record "
            "suction pressure. Convert the measured pressure to saturation temperature using "
            "refrigerant pressure-temperature tables or a digital manifold. Measure the actual "
            "suction line temperature with a calibrated clamp thermometer near the evaporator outlet. "
            "Subtract the saturation temperature from the measured line temperature. The resulting "
            "value in degrees Fahrenheit is the superheat. Normal target values range from eight to "
            "twelve degrees for most residential systems. A high reading often indicates an undercharge."
        )
        rec = {
            "type": "multi_turn",
            "conversation": [
                {
                    "role": "user",
                    "content": "How do you measure superheat correctly on a residential split system?",
                },
                {"role": "assistant", "content": similar},
                {
                    "role": "user",
                    "content": "Can you summarise that explanation again without adding anything new?",
                },
                {"role": "assistant", "content": repeat},
            ],
        }
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert not keep
        assert reason == "assistant_turn_repetition"

    def test_custom_config_threshold(self) -> None:
        # With a very low threshold, a short answer that normally fails passes
        rec = {
            "type": "multi_turn",
            "conversation": [
                {"role": "user", "content": "What is the purpose of a filter-drier in a circuit?"},
                {"role": "assistant", "content": "It removes moisture and particles from refrigerant flow."},
            ],
        }
        cfg_strict = FilterConfig()
        cfg_loose = FilterConfig(min_assistant_words_multi=5)
        keep_strict, _, _ = filter_multi_turn(rec, cfg_strict)
        keep_loose, _, _ = filter_multi_turn(rec, cfg_loose)
        assert not keep_strict
        assert keep_loose


# ---------------------------------------------------------------------------
# filter_multi_turn — substantive expansion (should NOT reject)
# ---------------------------------------------------------------------------


class TestSubstantiveExpansion:
    def test_substantive_expansion_not_rejected(self) -> None:
        # Turn A must be >= 80 words to pass the first-turn word-count gate.
        turn_a = (
            "Subcooling is the cooling of liquid refrigerant below its saturation temperature at "
            "a given pressure. It ensures that only liquid refrigerant enters the metering device, "
            "preventing flash gas that would reduce system capacity and efficiency. Normal subcooling "
            "is between 10 and 15 degrees Fahrenheit for most residential and light commercial "
            "systems. Measurement requires a manifold gauge set connected to the high side service "
            "port and a calibrated clamp thermometer placed on the liquid line downstream of the "
            "condenser. Low subcooling is a common indicator of a low refrigerant charge or a "
            "condenser that is rejecting heat inadequately due to fouling or airflow restrictions."
        )
        # Turn B introduces a numbered procedure AND an ASHRAE reference and is >35% longer.
        turn_b = (
            "Subcooling is the cooling of liquid refrigerant below its saturation temperature at "
            "a given pressure. It ensures that only liquid refrigerant enters the metering device, "
            "preventing flash gas that would reduce system capacity and efficiency. Normal subcooling "
            "is between 10 and 15 degrees Fahrenheit for most residential and light commercial "
            "systems. Measurement requires a manifold gauge set connected to the high side service "
            "port and a calibrated clamp thermometer placed on the liquid line downstream of the "
            "condenser. Low subcooling is a common indicator of a low refrigerant charge or a "
            "condenser that is rejecting heat inadequately due to fouling or airflow restrictions. "
            "To verify subcooling follow these steps precisely: "
            "1. Attach manifold gauges to the high side service port. "
            "2. Record condensing pressure and convert to saturation temperature. "
            "3. Measure liquid line temperature with a clamp probe at the service valve. "
            "4. Subtract saturation temperature from the measured line temperature. "
            "5. Compare result to the manufacturer specification on the data plate. "
            "ASHRAE Standard 15 governs safe refrigerant handling during this procedure. "
            "Always wear safety glasses and insulated gloves when connecting gauges to a "
            "live system under pressure. Allow the system to stabilise for at least fifteen "
            "minutes before taking final measurements. Document all readings on the service record."
        )
        rec = {
            "type": "multi_turn",
            "conversation": [
                {
                    "role": "user",
                    "content": "What is subcooling and how do you measure it in practice on a split system?",
                },
                {"role": "assistant", "content": turn_a},
                {
                    "role": "user",
                    "content": "Can you walk me through the exact measurement procedure step by step?",
                },
                {"role": "assistant", "content": turn_b},
            ],
        }
        keep, reason, _ = filter_multi_turn(rec, FilterConfig())
        assert keep is True, f"Expected keep=True but got reason={reason!r}"


# ---------------------------------------------------------------------------
# filter_single_turn — passing cases
# ---------------------------------------------------------------------------


class TestFilterSingleTurnPass:
    def test_good_record_passes(self, good_single_turn_record: dict[str, Any]) -> None:
        keep, reason, _ = filter_single_turn(good_single_turn_record, FilterConfig())
        assert keep is True

    def test_passes_with_troubleshooting_no_procedure(self, good_single_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_single_turn_record)
        # No numbered procedure — passes via troubleshooting terms and field example.
        # Must be >= 120 words to clear the word count gate.
        rec["answer"] = (
            "**Superheat** is measured at the suction line near the evaporator outlet. To verify "
            "it, check the suction pressure using a calibrated manifold gauge and measure the "
            "actual line temperature with a clamp thermometer. If readings are off, inspect the "
            "metering device for signs of restriction or improper setting. Diagnose any unusual "
            "readings by comparing against the manufacturer specifications and the system data "
            "plate. In the field, a reading above 20 degrees Fahrenheit typically indicates an "
            "undercharge or a restricted metering device, while low superheat near zero suggests "
            "an overcharge or a flooding evaporator. EPA 608 certification is required when adding "
            "or removing refrigerant from any system. Always verify your instruments are calibrated "
            "before taking measurements on live equipment. Consistent diagnostic practice reduces "
            "errors and improves first-call resolution rates across all system types encountered."
        )
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert keep is True


# ---------------------------------------------------------------------------
# filter_single_turn — rejection cases
# ---------------------------------------------------------------------------


class TestFilterSingleTurnReject:
    def test_too_short(self) -> None:
        rec = {"type": "single_turn", "answer": "**Short.** Check the filter."}
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason == "answer_word_count_out_of_range"

    def test_too_long(self) -> None:
        # Generate a 600-word answer
        sentence = "This is a sentence about refrigeration systems and their components. "
        long_answer = "**Refrigerants** are used in cooling. " + sentence * 50
        rec = {"type": "single_turn", "answer": long_answer}
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason == "answer_word_count_out_of_range"

    def test_missing_bold(self) -> None:
        # Answer passes word count (>= 120) and practical content but has no **bold** terms.
        answer = (
            "Superheat is measured by subtracting the saturation temperature from the actual "
            "suction line temperature at the evaporator outlet. To verify it accurately you "
            "should: 1. Attach gauges to the suction service port. 2. Record suction pressure. "
            "3. Convert pressure to saturation temperature using refrigerant tables. "
            "This is an EPA 608 requirement for any system where refrigerant is handled. "
            "In the field this procedure is standard practice for all residential split system "
            "installations and light commercial equipment. Always verify that system pressures "
            "are stable and that the system has run for at least fifteen minutes before recording "
            "final values. Unstable readings indicate transient conditions that do not represent "
            "steady-state operation and will produce inaccurate results. Document all readings "
            "on the service record for future reference and regulatory compliance."
        )
        rec = {"type": "single_turn", "answer": answer}
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason == "missing_bold_formatting"

    def test_insufficient_practical_content(self) -> None:
        # Has bold and >= 120 words but zero procedures, troubleshooting terms, code refs,
        # or field examples — all four practical-content signals are absent.
        answer = (
            "**Refrigerants** are chemical compounds used as working fluids in cooling systems. "
            "They circulate through the refrigeration cycle, absorbing heat at low pressure and "
            "releasing it at high pressure. The thermodynamic properties of a refrigerant "
            "determine system efficiency, capacity, and safety characteristics. Different "
            "refrigerants have different global warming potentials and ozone depletion potentials, "
            "which influence regulatory status and long-term availability for service technicians. "
            "Modern systems commonly use HFCs and HFOs as refrigerants due to environmental "
            "regulations that have phased out CFCs and HCFCs over recent decades. Refrigerant "
            "selection depends on operating pressures, temperature glide, and the specific "
            "application requirements of the equipment design. Proper matching of refrigerant "
            "type to system design is essential for reliable long-term operation and for "
            "maintaining the manufacturer's rated capacity and efficiency output values."
        )
        rec = {"type": "single_turn", "answer": answer}
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason == "insufficient_practical_content"

    def test_metadata_leakage_rejected(self) -> None:
        # Answer must be >= 120 words, pass practical-content check (added 'check' verb and
        # EPA 608 code reference), but contain an autofail metadata field name (competency_code).
        answer = (
            "**Competency** tracking uses the competency_code field to classify each training "
            "record within the curriculum framework. The system stores difficulty ratings and "
            "topic tags alongside each record to support analysis and reporting. difficulty: "
            "intermediate. topic: refrigerants. does not require advanced skills to perform. "
            "These fields are tracked in the review system for all technician training records "
            "and are updated by curriculum staff after each content review cycle. Always check "
            "that fields are updated correctly before publishing content to the training platform. "
            "EPA 608 certification status is stored alongside competency records. Ensuring "
            "review_status is set to approved is a mandatory step in the workflow before any "
            "learner can access the material. Incomplete records are flagged automatically and "
            "routed back to the author for correction and re-submission to the review queue. "
            "Additional validation steps are applied by the quality assurance team regularly."
        )
        rec = {"type": "single_turn", "answer": answer}
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason == "metadata_leakage"

    def test_trailing_ellipsis_rejected(self, good_single_turn_record: dict[str, Any]) -> None:
        rec = deepcopy(good_single_turn_record)
        rec["answer"] = rec["answer"][:100] + "..."
        keep, reason, _ = filter_single_turn(rec, FilterConfig())
        assert not keep
        assert reason in ("answer_word_count_out_of_range", "trailing_ellipsis_truncation")


# ---------------------------------------------------------------------------
# filter_records — top-level API
# ---------------------------------------------------------------------------


class TestFilterRecords:
    def test_returns_filter_result(self, good_multi_turn_record: dict[str, Any]) -> None:
        result = filter_records([good_multi_turn_record])
        assert isinstance(result, FilterResult)

    def test_good_record_kept(self, good_multi_turn_record: dict[str, Any]) -> None:
        result = filter_records([good_multi_turn_record])
        assert len(result.kept) == 1
        assert len(result.rejected) == 0

    def test_bad_record_rejected(self) -> None:
        bad = {
            "qa_id": "bad_001",
            "type": "multi_turn",
            "source_content_id": 1,
            "conversation": [
                {"role": "user", "content": "Explain something."},
                {"role": "assistant", "content": "Short."},
            ],
        }
        result = filter_records([bad])
        assert len(result.kept) == 0
        assert len(result.rejected) == 1

    def test_rejected_record_has_required_keys(self) -> None:
        bad = {
            "qa_id": "bad_002",
            "type": "multi_turn",
            "source_content_id": 42,
            "conversation": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "Nothing."},
            ],
        }
        result = filter_records([bad])
        row = result.rejected[0]
        for key in ("qa_id", "source_content_id", "record_type", "reason", "detail"):
            assert key in row

    def test_unknown_type_passes_through(self) -> None:
        rec = {"qa_id": "unk_001", "type": "unknown_type", "data": "something"}
        result = filter_records([rec])
        assert len(result.kept) == 1
        assert len(result.rejected) == 0

    def test_empty_list(self) -> None:
        result = filter_records([])
        assert result.kept == []
        assert result.rejected == []

    def test_none_config_uses_defaults(self, good_multi_turn_record: dict[str, Any]) -> None:
        result = filter_records([good_multi_turn_record], config=None)
        assert len(result.kept) == 1

    def test_mixed_record_types(
        self,
        good_multi_turn_record: dict[str, Any],
        good_single_turn_record: dict[str, Any],
    ) -> None:
        records = [good_multi_turn_record, good_single_turn_record]
        result = filter_records(records)
        assert len(result.kept) == 2
        assert len(result.rejected) == 0

    def test_fixture_file_multi_turn(self, multi_turn_samples: list[dict[str, Any]]) -> None:
        result = filter_records(multi_turn_samples)
        # mt_001 and mt_004 should pass; mt_002 (short), mt_003 (dup), mt_005 (leakage) fail
        kept_ids = {r["qa_id"] for r in result.kept}
        rejected_ids = {r["qa_id"] for r in result.rejected}
        assert "mt_001" in kept_ids
        assert "mt_004" in kept_ids
        assert "mt_002" in rejected_ids
        assert "mt_003" in rejected_ids
        assert "mt_005" in rejected_ids

    def test_fixture_file_single_turn(self, single_turn_samples: list[dict[str, Any]]) -> None:
        result = filter_records(single_turn_samples)
        kept_ids = {r["qa_id"] for r in result.kept}
        rejected_ids = {r["qa_id"] for r in result.rejected}
        # Good records
        assert "st_001" in kept_ids
        assert "st_003" in kept_ids
        assert "st_004" in kept_ids
        # Bad records
        assert "st_002" in rejected_ids  # too short + no bold
        assert "st_005" in rejected_ids  # metadata leakage
        assert "st_007" in rejected_ids  # trailing ellipsis

    def test_count_invariant(self, multi_turn_samples: list[dict[str, Any]]) -> None:
        result = filter_records(multi_turn_samples)
        assert len(result.kept) + len(result.rejected) == len(multi_turn_samples)

    def test_custom_standards_patterns(self, good_multi_turn_record: dict[str, Any]) -> None:
        # Use a custom standards pattern that matches nothing in the fixture
        custom = [re.compile(r"ISO\s*9001")]
        cfg = FilterConfig(standards_patterns=custom)
        result = filter_records([good_multi_turn_record], cfg)
        assert len(result.kept) == 1
