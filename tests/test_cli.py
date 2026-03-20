"""Tests for cli.py — I/O helpers and the main() entry point."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conversation_quality_filter.cli import (
    emit_numeric_claims,
    load_records,
    print_summary,
    write_jsonl,
    write_rejections,
    write_summary_json,
)


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


class TestLoadRecords:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        records, malformed = load_records(p)
        assert len(records) == 2
        assert malformed == 0

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
        records, malformed = load_records(p)
        assert len(records) == 2
        assert malformed == 0

    def test_counts_malformed_lines(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\nnot-json\n{"c": 3}\n', encoding="utf-8")
        records, malformed = load_records(p)
        assert len(records) == 2
        assert malformed == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        records, malformed = load_records(p)
        assert records == []
        assert malformed == 0

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "unicode.jsonl"
        p.write_text('{"text": "caf\u00e9 \u00b0F"}\n', encoding="utf-8")
        records, _ = load_records(p)
        assert records[0]["text"] == "caf\u00e9 \u00b0F"


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------


class TestWriteJsonl:
    def test_writes_all_records(self, tmp_path: Path) -> None:
        records = [{"id": 1}, {"id": 2, "val": "hello"}]
        p = tmp_path / "out.jsonl"
        write_jsonl(records, p)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1

    def test_empty_list_creates_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        write_jsonl([], p)
        assert p.read_text(encoding="utf-8") == ""

    def test_unicode_not_escaped(self, tmp_path: Path) -> None:
        p = tmp_path / "uni.jsonl"
        write_jsonl([{"t": "\u00b0F"}], p)
        content = p.read_text(encoding="utf-8")
        assert "\u00b0F" in content


# ---------------------------------------------------------------------------
# write_rejections
# ---------------------------------------------------------------------------


class TestWriteRejections:
    def test_writes_csv_with_header(self, tmp_path: Path) -> None:
        rejections = [
            {
                "qa_id": "r1",
                "source_content_id": "1",
                "record_type": "multi_turn",
                "reason": "first_turn_quality_failure",
                "detail": "too short",
                "stage": "quality_filter",
            }
        ]
        p = tmp_path / "rej.csv"
        write_rejections(rejections, p)
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["qa_id"] == "r1"
        assert rows[0]["reason"] == "first_turn_quality_failure"

    def test_empty_rejections(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        write_rejections([], p)
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert rows == []

    def test_stage_column_defaulted(self, tmp_path: Path) -> None:
        rejections = [
            {
                "qa_id": "r2",
                "source_content_id": "2",
                "record_type": "single_turn",
                "reason": "too_short",
                "detail": "5 words",
                # no 'stage' key
            }
        ]
        p = tmp_path / "rej2.csv"
        write_rejections(rejections, p)
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert rows[0]["stage"] == "quality_filter"


# ---------------------------------------------------------------------------
# write_summary_json
# ---------------------------------------------------------------------------


class TestWriteSummaryJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        summary = {"kept": 10, "rejected": 2, "rate": 0.167}
        p = tmp_path / "summary.json"
        write_summary_json(summary, p)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["kept"] == 10

    def test_pretty_printed(self, tmp_path: Path) -> None:
        p = tmp_path / "summary.json"
        write_summary_json({"a": 1}, p)
        content = p.read_text(encoding="utf-8")
        assert "\n" in content  # pretty-printed has newlines


# ---------------------------------------------------------------------------
# emit_numeric_claims
# ---------------------------------------------------------------------------


class TestEmitNumericClaims:
    def _single_turn(self, qa_id: str, answer: str) -> dict[str, Any]:
        return {"qa_id": qa_id, "type": "single_turn", "source_content_id": 1, "answer": answer}

    def test_temperature_extracted(self, tmp_path: Path) -> None:
        rec = self._single_turn("t1", "Discharge temp is 135 °F at the service port.")
        p = tmp_path / "claims.csv"
        count = emit_numeric_claims([rec], p)
        assert count >= 1
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert any(r["claim_type"] == "temperature" for r in rows)

    def test_pressure_extracted(self, tmp_path: Path) -> None:
        rec = self._single_turn("p1", "High side reads 285 psig during normal operation.")
        p = tmp_path / "claims.csv"
        count = emit_numeric_claims([rec], p)
        assert count >= 1
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert any(r["claim_type"] == "pressure" for r in rows)

    def test_no_claims_empty_file(self, tmp_path: Path) -> None:
        rec = self._single_turn("n1", "No numeric claims here.")
        p = tmp_path / "claims.csv"
        count = emit_numeric_claims([rec], p)
        assert count == 0

    def test_context_window_included(self, tmp_path: Path) -> None:
        answer = "x" * 100 + " the pressure is 200 psig " + "y" * 100
        rec = self._single_turn("ctx1", answer)
        p = tmp_path / "claims.csv"
        emit_numeric_claims([rec], p)
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        assert len(rows[0]["context"]) > 0

    def test_multi_turn_claims(self, tmp_path: Path) -> None:
        rec: dict[str, Any] = {
            "qa_id": "mt1",
            "type": "multi_turn",
            "source_content_id": 2,
            "conversation": [
                {"role": "user", "content": "What is the temperature?"},
                {"role": "assistant", "content": "It is 140 °F at 300 psig typically."},
            ],
        }
        p = tmp_path / "claims.csv"
        count = emit_numeric_claims([rec], p)
        assert count >= 2


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def _make_audit(self) -> dict[str, Any]:
        return {
            "warnings": {},
            "competency_counts": {},
            "difficulty_counts": {"beginner": 2, "intermediate": 5, "advanced": 1},
            "evidence_missing": 1,
            "temp_claims": 3,
            "pressure_claims": 2,
        }

    def test_returns_dict(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = print_summary(10, 8, [{"reason": "too_short", "record_type": "multi_turn"}] * 2, self._make_audit(), 1.5)
        assert isinstance(result, dict)

    def test_summary_keys_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = print_summary(10, 8, [{"reason": "r", "record_type": "t"}] * 2, self._make_audit(), 1.0)
        for key in ("input_count", "kept_count", "rejected_count", "rejection_rate", "reason_breakdown"):
            assert key in result

    def test_count_mismatch_raises(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(RuntimeError):
            print_summary(10, 5, [{"reason": "r", "record_type": "t"}], self._make_audit(), 1.0)

    def test_rejection_rate_calculated(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = print_summary(10, 8, [{"reason": "r", "record_type": "t"}] * 2, self._make_audit(), 1.0)
        assert result["rejection_rate"] == pytest.approx(0.2, abs=0.01)

    def test_zero_input_no_division_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Should not raise ZeroDivisionError
        result = print_summary(0, 0, [], self._make_audit(), 0.1)
        assert result["rejection_rate"] == 0.0

    def test_outputs_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_summary(5, 4, [{"reason": "r", "record_type": "t"}], self._make_audit(), 0.5)
        captured = capsys.readouterr()
        assert "QUALITY FILTER SUMMARY" in captured.out
        assert "Record Counts" in captured.out


# ---------------------------------------------------------------------------
# CLI main() integration test
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Smoke-test the full CLI pipeline on the fixture files."""
        fixtures = Path(__file__).parent / "fixtures"
        # Combine both fixture files
        combined = tmp_path / "input.jsonl"
        with open(combined, "w", encoding="utf-8") as out:
            for src in (fixtures / "multi_turn_samples.jsonl", fixtures / "single_turn_samples.jsonl"):
                out.write(src.read_text(encoding="utf-8"))

        output = tmp_path / "filtered.jsonl"
        rejections = tmp_path / "rejected.csv"
        summary_json = tmp_path / "summary.json"

        # Import and call main() directly rather than subprocess to keep tests fast
        from conversation_quality_filter.cli import main

        sys.argv = [
            "conversation-quality-filter",
            "--input", str(combined),
            "--output", str(output),
            "--rejections", str(rejections),
            "--summary-json", str(summary_json),
        ]
        main()

        # Verify outputs exist and are non-empty / correctly formatted
        assert output.exists()
        assert rejections.exists()
        assert summary_json.exists()

        kept_records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        rejected_rows = list(csv.DictReader(rejections.open(encoding="utf-8")))
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

        assert len(kept_records) + len(rejected_rows) == summary["input_count"]
        assert summary["kept_count"] == len(kept_records)
        assert summary["rejected_count"] == len(rejected_rows)
        # Verify stage column
        for row in rejected_rows:
            assert row["stage"] == "quality_filter"

    def test_empty_input_does_not_crash(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")

        from conversation_quality_filter.cli import main

        sys.argv = [
            "conversation-quality-filter",
            "--input", str(empty),
            "--output", str(tmp_path / "out.jsonl"),
            "--rejections", str(tmp_path / "rej.csv"),
        ]
        main()  # Should not raise

    def test_numeric_claims_csv_written(self, tmp_path: Path) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        output = tmp_path / "filtered.jsonl"
        rejections = tmp_path / "rejected.csv"
        claims_csv = tmp_path / "claims.csv"

        from conversation_quality_filter.cli import main

        sys.argv = [
            "conversation-quality-filter",
            "--input", str(fixtures / "multi_turn_samples.jsonl"),
            "--output", str(output),
            "--rejections", str(rejections),
            "--emit-numeric-claims-csv", str(claims_csv),
        ]
        main()
        assert claims_csv.exists()
