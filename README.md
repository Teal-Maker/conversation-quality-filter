# conversation-quality-filter

Production-grade quality filters for multi-turn LLM conversation datasets.

## Why use this?

When you generate synthetic QA pairs or multi-turn conversations with LLMs for fine-tuning, the raw output contains predictable defects: duplicated turns, truncated answers, metadata leaking into responses, and repetitive content across turns. Manually reviewing thousands of records is impractical.

This library applies deterministic hard filters that catch these defects automatically, plus soft audits that flag dataset-level issues like category imbalance or difficulty skew. It reads JSONL, writes JSONL, has zero required dependencies, and works as both a Python API and a CLI.

**Use this when you are:**
- Building fine-tuning datasets from LLM-generated conversations
- Post-processing synthetic QA pairs before training
- Auditing dataset quality (coverage gaps, difficulty distribution, numeric claim density)

## Hard Filters

Applies deterministic hard filters to catch defect patterns:

**Multi-turn records** (`"type": "multi_turn"`):
1. Duplicate or repeated assistant turns (fuzzy matching)
2. Overly short assistant or user turns
3. Metadata leakage — rubric language or field names bleeding into answers
4. Trailing ellipsis truncation
5. Turn-to-turn repetition (Jaccard + fuzzy overlap)
6. Invalid record schema (missing required fields)

**Single-turn records** (`"type": "single_turn"`):
1. Answer word count out of configured range
2. Missing bold formatting (`**key terms**`)
3. Insufficient practical content (no procedures, troubleshooting terms, standards references, or field examples)
4. Metadata leakage
5. Trailing ellipsis truncation

## Soft Audit

Also provides soft dataset audit warnings via `audit_dataset()`: category coverage
gaps, difficulty distribution skew, missing evidence spans, and numeric claim density.

## Install

```bash
pip install conversation-quality-filter

# Optional: faster fuzzy matching
pip install "conversation-quality-filter[fast]"
```

## Python API

```python
from conversation_quality_filter import filter_records, FilterConfig

records = [...]  # list of dicts, each with a "type" key

result = filter_records(records)
print(f"Kept {len(result.kept)} / {len(records)} records")

# Custom thresholds
config = FilterConfig(min_assistant_words_multi=60, single_turn_max_words=400)
result = filter_records(records, config)
```

### Dataset Audit

```python
from conversation_quality_filter.audit import audit_dataset

audit = audit_dataset(result.kept)
print(audit["warnings"])           # category coverage, difficulty skew
print(audit["difficulty_counts"])   # beginner/intermediate/advanced distribution
print(audit["temp_claims"])         # temperature claim count
```

## CLI

```bash
conversation-quality-filter \
    --input dataset.jsonl \
    --output filtered.jsonl \
    --rejections rejected.csv \
    --summary-json summary.json
```

## License

Apache-2.0
