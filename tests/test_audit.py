"""Tests for audit.py — soft-warning dataset audit."""

from __future__ import annotations

from typing import Any

import pytest

from conversation_quality_filter.audit import audit_dataset


def _make_record(
    qa_id: str = "test",
    competency: str = "REF-001",
    difficulty: str = "intermediate",
    has_evidence: bool = True,
    answer: str = "The refrigerant temperature was 75 °F at 150 psig.",
    record_type: str = "single_turn",
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "qa_id": qa_id,
        "type": record_type,
        "competency_code": competency,
        "difficulty": difficulty,
        "review_status": "approved",
        "answer": answer,
    }
    if has_evidence:
        rec["evidence_spans"] = ["some evidence"]
    else:
        rec["evidence_spans"] = []
    return rec


class TestAuditDataset:
    def test_empty_dataset(self) -> None:
        result = audit_dataset([])
        assert result["evidence_missing"] == 0
        assert result["temp_claims"] == 0
        assert result["pressure_claims"] == 0
        assert result["warnings"] == {}

    def test_returns_required_keys(self) -> None:
        result = audit_dataset([_make_record()])
        for key in ("warnings", "competency_counts", "difficulty_counts", "evidence_missing", "temp_claims", "pressure_claims"):
            assert key in result

    def test_temperature_claim_detection(self) -> None:
        rec = _make_record(answer="The discharge temperature is 130 °F at the outlet port.")
        result = audit_dataset([rec])
        assert result["temp_claims"] >= 1

    def test_pressure_claim_detection(self) -> None:
        rec = _make_record(answer="System pressure is 280 psig on the high side during operation.")
        result = audit_dataset([rec])
        assert result["pressure_claims"] >= 1

    def test_celsius_temperature_detected(self) -> None:
        rec = _make_record(answer="The condensing temperature is 54 °C at peak summer conditions.")
        result = audit_dataset([rec])
        assert result["temp_claims"] >= 1

    def test_kpa_pressure_detected(self) -> None:
        rec = _make_record(answer="High-side pressure is 1930 kpa in this configuration.")
        result = audit_dataset([rec])
        assert result["pressure_claims"] >= 1

    def test_evidence_missing_counted(self) -> None:
        no_ev = _make_record(qa_id="no_ev", has_evidence=False)
        with_ev = _make_record(qa_id="with_ev", has_evidence=True)
        result = audit_dataset([no_ev, with_ev])
        assert result["evidence_missing"] == 1

    def test_strict_evidence_populates_warnings(self) -> None:
        no_ev = _make_record(qa_id="no_ev_001", has_evidence=False)
        result = audit_dataset([no_ev], strict_evidence=True)
        assert "missing_evidence" in result["warnings"]
        assert "no_ev_001" in result["warnings"]["missing_evidence"]

    def test_strict_evidence_false_no_warning_list(self) -> None:
        no_ev = _make_record(has_evidence=False)
        result = audit_dataset([no_ev], strict_evidence=False)
        assert "missing_evidence" not in result["warnings"]

    def test_low_competency_coverage_warning(self) -> None:
        # Only 1 record for "RARE-001" — below default threshold of 50
        rec = _make_record(qa_id="rare_001", competency="RARE-001")
        result = audit_dataset([rec])
        warnings = result["warnings"]
        assert "low_competency_coverage" in warnings
        items = warnings["low_competency_coverage"]
        assert any("RARE-001" in item for item in items)

    def test_no_low_coverage_warning_above_threshold(self) -> None:
        # 60 records for the same competency — above default threshold of 50
        records = [_make_record(qa_id=f"r_{i}", competency="COMMON-001") for i in range(60)]
        result = audit_dataset(records, competency_warn_threshold=50)
        low = result["warnings"].get("low_competency_coverage", [])
        assert not any("COMMON-001" in item for item in low)

    def test_difficulty_skew_warning(self) -> None:
        # 90% intermediate → should trigger skew warning (threshold 0.80)
        records = [_make_record(qa_id=f"r_{i}", difficulty="intermediate") for i in range(9)]
        records.append(_make_record(qa_id="r_9", difficulty="beginner"))
        result = audit_dataset(records)
        assert "difficulty_skew" in result["warnings"]

    def test_no_skew_below_threshold(self) -> None:
        records = [_make_record(qa_id=f"r_{i}", difficulty="intermediate") for i in range(5)]
        records += [_make_record(qa_id=f"b_{i}", difficulty="beginner") for i in range(5)]
        result = audit_dataset(records)
        assert "difficulty_skew" not in result["warnings"]

    def test_competency_counts_present(self) -> None:
        records = [
            _make_record(qa_id="a", competency="REF-001"),
            _make_record(qa_id="b", competency="REF-001"),
            _make_record(qa_id="c", competency="REF-002"),
        ]
        result = audit_dataset(records, competency_warn_threshold=1)
        assert result["competency_counts"]["REF-001"] == 2
        assert result["competency_counts"]["REF-002"] == 1

    def test_difficulty_counts_present(self) -> None:
        records = [
            _make_record(qa_id="a", difficulty="beginner"),
            _make_record(qa_id="b", difficulty="intermediate"),
            _make_record(qa_id="c", difficulty="intermediate"),
        ]
        result = audit_dataset(records)
        assert result["difficulty_counts"]["beginner"] == 1
        assert result["difficulty_counts"]["intermediate"] == 2

    def test_multi_turn_text_extracted(self) -> None:
        rec: dict[str, Any] = {
            "qa_id": "mt_audit",
            "type": "multi_turn",
            "competency_code": "REF-001",
            "difficulty": "intermediate",
            "review_status": "approved",
            "evidence_spans": ["span"],
            "conversation": [
                {"role": "user", "content": "What temperature does the discharge reach?"},
                {"role": "assistant", "content": "Discharge temperature is typically 140 °F at 300 psig."},
            ],
        }
        result = audit_dataset([rec])
        assert result["temp_claims"] >= 1
        assert result["pressure_claims"] >= 1

    def test_custom_thresholds_respected(self) -> None:
        records = [_make_record(qa_id=f"r_{i}", competency="TEST-001") for i in range(5)]
        result_strict = audit_dataset(records, competency_warn_threshold=10)
        result_loose = audit_dataset(records, competency_warn_threshold=3)
        # With threshold=10, 5 records should trigger warning
        assert "low_competency_coverage" in result_strict["warnings"]
        # With threshold=3, 5 records should NOT trigger warning
        assert "low_competency_coverage" not in result_loose["warnings"]

    def test_needs_evidence_status_counted(self) -> None:
        rec = _make_record(has_evidence=True)  # has spans but status is needs_evidence
        rec["review_status"] = "needs_evidence"
        result = audit_dataset([rec])
        assert result["evidence_missing"] == 1
