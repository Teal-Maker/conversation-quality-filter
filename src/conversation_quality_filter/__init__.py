"""conversation-quality-filter — production-grade quality filters for LLM datasets.

Quickstart::

    from conversation_quality_filter import filter_records, FilterConfig

    records = [...]  # list of dicts, each with a "type" key

    result = filter_records(records)
    print(f"Kept {len(result.kept)} / {len(records)} records")

    # Custom thresholds
    config = FilterConfig(min_assistant_words_multi=60, single_turn_max_words=400)
    result = filter_records(records, config)

Public API
----------
``filter_records``
    Top-level function — apply all hard filters to a list of records.
``FilterConfig``
    Dataclass holding all tunable thresholds.
``FilterResult``
    Dataclass returned by ``filter_records`` with ``kept`` and ``rejected`` lists.
"""

from .filters import FilterConfig, FilterResult, filter_records

__all__ = ["filter_records", "FilterConfig", "FilterResult"]
