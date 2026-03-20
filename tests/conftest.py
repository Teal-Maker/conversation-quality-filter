"""Shared pytest fixtures for conversation-quality-filter tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture
def multi_turn_samples() -> list[dict[str, Any]]:
    """All multi-turn fixture records."""
    return _load_jsonl(FIXTURES_DIR / "multi_turn_samples.jsonl")


@pytest.fixture
def single_turn_samples() -> list[dict[str, Any]]:
    """All single-turn fixture records."""
    return _load_jsonl(FIXTURES_DIR / "single_turn_samples.jsonl")


@pytest.fixture
def good_multi_turn_record() -> dict[str, Any]:
    """A well-formed multi-turn record that should pass all filters."""
    return {
        "qa_id": "fixture_mt_good",
        "type": "multi_turn",
        "source_content_id": 999,
        "competency_code": "TEST-001",
        "difficulty": "intermediate",
        "conversation": [
            {
                "role": "user",
                "content": "Can you explain in detail how a heat pump operates in heating mode and what cycle it follows?",
            },
            {
                "role": "assistant",
                "content": (
                    "In heating mode a heat pump reverses the standard refrigeration cycle by using a "
                    "reversing valve to redirect refrigerant flow. The outdoor unit now acts as the "
                    "evaporator, absorbing heat from the ambient air even at low outdoor temperatures, "
                    "while the indoor coil acts as the condenser and rejects heat into the conditioned "
                    "space. The refrigerant picks up latent heat from the outdoor air as it evaporates "
                    "at low pressure, then the compressor raises its pressure and temperature. The hot "
                    "high-pressure refrigerant flows to the indoor coil where it condenses and gives up "
                    "heat to the indoor air stream. A metering device on the outdoor section then "
                    "reduces refrigerant pressure before it returns to the outdoor coil to repeat the "
                    "cycle. Efficiency is expressed as a coefficient of performance, which is the ratio "
                    "of heat delivered to electrical energy consumed. Modern heat pumps can achieve "
                    "COPs of 3.0 or higher at moderate outdoor temperatures."
                ),
            },
            {
                "role": "user",
                "content": "At what outdoor temperature does a heat pump typically switch to auxiliary heating and why?",
            },
            {
                "role": "assistant",
                "content": (
                    "Most heat pumps switch to auxiliary or emergency heating at an outdoor temperature "
                    "called the balance point, which is typically between 30 and 40 degrees Fahrenheit "
                    "depending on the system design and building heat loss. At that temperature the heat "
                    "pump's capacity equals the building's heating load, and below it the heat pump "
                    "alone cannot meet demand. Auxiliary heating, usually electric resistance strips "
                    "staged in the air handler, supplements the heat pump output to maintain comfort. "
                    "The balance point is calculated during system design by comparing the heat pump's "
                    "published capacity curves at various outdoor temperatures against the building's "
                    "heat loss at those same temperatures. Properly sized auxiliary heat avoids "
                    "excessive electric resistance operation, which is far less efficient than the heat "
                    "pump. ASHRAE recommends sizing auxiliary heat based on the design heating load "
                    "minus the heat pump capacity at the design outdoor temperature rather than "
                    "providing full backup capacity."
                ),
            },
        ],
    }


@pytest.fixture
def good_single_turn_record() -> dict[str, Any]:
    """A well-formed single-turn record that should pass all filters."""
    return {
        "qa_id": "fixture_st_good",
        "type": "single_turn",
        "source_content_id": 998,
        "competency_code": "TEST-002",
        "difficulty": "intermediate",
        "question": "How do you verify superheat on a residential split system?",
        "answer": (
            "**Superheat** measurement verifies that the evaporator is feeding the compressor "
            "with fully vaporized refrigerant. To measure it, follow these steps: "
            "1. Attach manifold gauges to the suction service port. "
            "2. Record suction pressure and convert to saturation temperature using refrigerant "
            "tables or a digital manifold that auto-calculates saturation. "
            "3. Measure suction line temperature at the evaporator outlet with a clamp thermometer. "
            "4. Subtract saturation temperature from the measured line temperature to obtain superheat. "
            "Target superheat for most residential systems is 10 to 15 degrees Fahrenheit. "
            "In the field, low superheat often indicates an overcharge or a flooded evaporator, "
            "while high superheat suggests an undercharge or a restricted metering device. "
            "Always verify against the manufacturer's specification placard before adjusting charge. "
            "EPA 608 certification is required when adding or removing refrigerant from any system."
        ),
    }
