"""Command-line interface for conversation-quality-filter.

Usage::

    conversation-quality-filter \\
        --input stage4.jsonl \\
        --output filtered.jsonl \\
        --rejections rejected.csv \\
        [--summary-json summary.json] \\
        [--min-assistant-words 80] \\
        [--min-user-words 10] \\
        [--single-min-words 120] \\
        [--single-max-words 500] \\
        [--strict-evidence] \\
        [--emit-numeric-claims-csv claims.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .audit import (
    COMPETENCY_WARN_THRESHOLD as _COMPETENCY_WARN_THRESHOLD,
    _PRESSURE_RE,
    _TEMP_RE,
    _get_all_assistant_text,
    audit_dataset,
)
from .filters import FilterConfig, filter_records

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load newline-delimited JSON records from *path*.

    Args:
        path: Path to a ``.jsonl`` file.

    Returns:
        ``(records, malformed_count)`` — *malformed_count* is the number of
        lines that could not be parsed as JSON.
    """
    records: list[dict[str, Any]] = []
    malformed = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                malformed += 1
                print(f"[WARN] Malformed JSON at line {lineno}: {exc}", file=sys.stderr)
    return records, malformed


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write *records* to *path* as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_rejections(rejections: list[dict[str, Any]], path: Path) -> None:
    """Write rejection records to *path* as CSV."""
    fieldnames = ["qa_id", "source_content_id", "record_type", "reason", "detail", "stage"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Ensure the stage column is populated without mutating the caller's dicts.
        writer.writerows(
            {**row, "stage": row.get("stage", "quality_filter")} for row in rejections
        )


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    """Write *summary* dict to *path* as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def emit_numeric_claims(records: list[dict[str, Any]], path: Path) -> int:
    """Extract all temperature and pressure claims from *records* to CSV.

    Args:
        records: Kept records to scan.
        path: Output CSV path.

    Returns:
        Total number of claim rows written.
    """
    fieldnames = ["qa_id", "source_content_id", "claim_type", "value", "context"]
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            qa_id = rec.get("qa_id", "")
            src_id = rec.get("source_content_id", "")
            text = _get_all_assistant_text(rec)
            for match in _TEMP_RE.finditer(text):
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                writer.writerow(
                    {
                        "qa_id": qa_id,
                        "source_content_id": src_id,
                        "claim_type": "temperature",
                        "value": match.group(),
                        "context": text[start:end].replace("\n", " "),
                    }
                )
                count += 1
            for match in _PRESSURE_RE.finditer(text):
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                writer.writerow(
                    {
                        "qa_id": qa_id,
                        "source_content_id": src_id,
                        "claim_type": "pressure",
                        "value": match.group(),
                        "context": text[start:end].replace("\n", " "),
                    }
                )
                count += 1
    return count


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def print_summary(
    input_count: int,
    kept_count: int,
    rejections: list[dict[str, Any]],
    audit: dict[str, Any],
    elapsed: float,
    malformed_count: int = 0,
) -> dict[str, Any]:
    """Print a human-readable filter summary to stdout and return a summary dict.

    Args:
        input_count: Total records loaded (excluding malformed lines).
        kept_count: Records that passed all hard filters.
        rejections: List of rejection dicts as produced by ``filter_records``.
        audit: Dict returned by ``audit_dataset``.
        elapsed: Wall-clock seconds taken for the full run.
        malformed_count: Number of malformed JSON lines skipped at load time.

    Returns:
        Summary dict suitable for serialisation to JSON.
    """
    print("\n" + "=" * 70)
    print("QUALITY FILTER SUMMARY: Conversation Quality Filters")
    print("=" * 70)

    print("\n--- Record Counts ---")
    print(f"  Input:     {input_count:>8,}")
    if malformed_count:
        print(f"  Malformed: {malformed_count:>8,}  (skipped)")
    print(f"  Kept:      {kept_count:>8,}")
    print(f"  Rejected:  {len(rejections):>8,}  ({len(rejections) / max(input_count, 1) * 100:.1f}%)")

    if input_count != kept_count + len(rejections):
        raise RuntimeError(
            f"Count mismatch: {input_count} != {kept_count} + {len(rejections)}"
        )

    reason_counts: Counter[str] = Counter(r["reason"] for r in rejections)
    if reason_counts:
        print("\n--- Rejection Breakdown ---")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<35s} {count:>6,}")

    type_counts: Counter[str] = Counter(r["record_type"] for r in rejections)
    if type_counts:
        print("\n--- Rejections by Record Type ---")
        for rtype, count in type_counts.most_common():
            print(f"  {rtype:<20s} {count:>6,}")

    warnings = audit.get("warnings", {})
    if warnings:
        print("\n--- Soft Warnings ---")
        for category, items in warnings.items():
            print(f"  {category}: {len(items)} items")
            for item in items[:5]:
                print(f"    - {item}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")

    diff_counts = audit.get("difficulty_counts", {})
    if diff_counts:
        print("\n--- Difficulty Distribution (kept) ---")
        for diff in ("beginner", "intermediate", "advanced"):
            c = diff_counts.get(diff, 0)
            pct = c / max(kept_count, 1) * 100
            print(f"  {diff:<15s} {c:>6,}  ({pct:5.1f}%)")

    print("\n--- Evidence & Numeric Claims (kept) ---")
    print(f"  Missing evidence:   {audit.get('evidence_missing', 0):>6,}")
    print(f"  Temperature claims: {audit.get('temp_claims', 0):>6,}")
    print(f"  Pressure claims:    {audit.get('pressure_claims', 0):>6,}")

    low_comp = audit.get("warnings", {}).get("low_competency_coverage", [])
    if low_comp:
        print(f"\n--- Low Competency Coverage (<{_COMPETENCY_WARN_THRESHOLD} examples) ---")
        for item in sorted(low_comp):
            print(f"  {item}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    return {
        "input_count": input_count,
        "malformed_count": malformed_count,
        "kept_count": kept_count,
        "rejected_count": len(rejections),
        "rejection_rate": round(len(rejections) / max(input_count, 1), 4),
        "reason_breakdown": dict(reason_counts),
        "difficulty_counts": diff_counts,
        "evidence_missing": audit.get("evidence_missing", 0),
        "temp_claims": audit.get("temp_claims", 0),
        "pressure_claims": audit.get("pressure_claims", 0),
        "warnings": {k: len(v) for k, v in warnings.items()},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``conversation-quality-filter``."""
    parser = argparse.ArgumentParser(
        description="Quality filters for multi-turn and single-turn LLM conversation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output JSONL")
    parser.add_argument("--rejections", type=Path, required=True, help="Rejection CSV output")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--min-assistant-words",
        type=int,
        default=FilterConfig.min_assistant_words_multi,
        help="Min words for assistant turns in multi-turn records",
    )
    parser.add_argument(
        "--min-user-words",
        type=int,
        default=FilterConfig.min_user_words_multi,
        help="Min words for user turns in multi-turn records",
    )
    parser.add_argument(
        "--single-min-words",
        type=int,
        default=FilterConfig.single_turn_min_words,
        help="Min words for single-turn answers",
    )
    parser.add_argument(
        "--single-max-words",
        type=int,
        default=FilterConfig.single_turn_max_words,
        help="Max words for single-turn answers",
    )
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="Warn on individual records missing evidence spans",
    )
    parser.add_argument(
        "--emit-numeric-claims-csv",
        type=Path,
        default=None,
        help="Write all numeric (temperature/pressure) claims to CSV",
    )
    args = parser.parse_args()

    t_start = time.monotonic()

    # Phase 1: Load
    print(f"[quality-filter] Loading {args.input} ...")
    records, malformed_count = load_records(args.input)
    input_count = len(records)
    print(f"  Loaded {input_count:,} records")
    if malformed_count:
        print(f"  [WARN] Skipped {malformed_count:,} malformed JSON lines")

    if input_count == 0:
        print("[quality-filter] No records to process.")
        write_jsonl([], args.output)
        write_rejections([], args.rejections)
        return

    type_counts: Counter[str] = Counter(rec.get("type", "unknown") for rec in records)
    for rtype, count in type_counts.most_common():
        print(f"  {rtype}: {count:,}")

    # Phase 2: Hard filters
    print("\n[quality-filter] Applying hard filters ...")
    config = FilterConfig(
        min_assistant_words_multi=args.min_assistant_words,
        min_user_words_multi=args.min_user_words,
        single_turn_min_words=args.single_min_words,
        single_turn_max_words=args.single_max_words,
    )
    result = filter_records(records, config)

    print(f"  Kept: {len(result.kept):,}  Rejected: {len(result.rejected):,}")

    # Phase 3: Dataset audit
    print("\n[quality-filter] Running dataset audit ...")
    audit = audit_dataset(result.kept, strict_evidence=args.strict_evidence)

    # Phase 4: Write outputs
    print("\n[quality-filter] Writing outputs ...")
    write_jsonl(result.kept, args.output)
    print(f"  Filtered JSONL: {args.output} ({len(result.kept):,} records)")

    write_rejections(result.rejected, args.rejections)
    print(f"  Rejections CSV: {args.rejections} ({len(result.rejected):,} records)")

    if args.emit_numeric_claims_csv:
        nc = emit_numeric_claims(result.kept, args.emit_numeric_claims_csv)
        print(f"  Numeric claims CSV: {args.emit_numeric_claims_csv} ({nc:,} claims)")

    elapsed = time.monotonic() - t_start
    summary = print_summary(
        input_count,
        len(result.kept),
        result.rejected,
        audit,
        elapsed,
        malformed_count,
    )

    if args.summary_json:
        write_summary_json(summary, args.summary_json)
        print(f"  Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
