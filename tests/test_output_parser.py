"""
Tests for the output parser – the most critical component.
These run without a GPU (no model required).
"""
import json

import pytest

from processing.output_parser import (
    parse_extraction_result,
    _extract_json_string,
    _parse_quantity_string,
    _repair_and_parse,
)
from schemas.extraction import ExtractionResult, UnitType


# ── _extract_json_string ──────────────────────────────────────────────

class TestExtractJsonString:
    def test_code_fence(self):
        text = 'Some preamble\n```json\n{"scopeItems": []}\n```\nmore text'
        assert _extract_json_string(text) == '{"scopeItems": []}'

    def test_bare_json(self):
        text = 'Here is the result: {"scopeItems": [{"id": "1"}]}'
        result = _extract_json_string(text)
        assert result is not None
        assert json.loads(result)["scopeItems"][0]["id"] == "1"

    def test_no_json(self):
        assert _extract_json_string("No JSON here at all") is None

    def test_nested_braces(self):
        text = '{"a": {"b": {"c": 1}}}'
        assert _extract_json_string(text) == text


# ── _parse_quantity_string ────────────────────────────────────────────

class TestParseQuantity:
    def test_feet_inches(self):
        assert _parse_quantity_string("13'-6\"") == pytest.approx(13.5)

    def test_feet_zero(self):
        assert _parse_quantity_string("4'-0\"") == pytest.approx(4.0)

    def test_plain_number(self):
        assert _parse_quantity_string("17") == 17.0

    def test_number_with_unit(self):
        assert _parse_quantity_string("22 EA") == 22.0

    def test_decimal(self):
        assert _parse_quantity_string("13.5") == 13.5

    def test_empty(self):
        assert _parse_quantity_string("") == 0


# ── _repair_and_parse ────────────────────────────────────────────────

class TestRepairAndParse:
    def test_trailing_comma(self):
        result = _repair_and_parse('{"items": [1, 2, 3,]}')
        assert result == {"items": [1, 2, 3]}

    def test_array_input(self):
        result = _repair_and_parse('[{"id": "1"}]')
        assert isinstance(result["scopeItems"], list)
        assert result["scopeItems"][0]["id"] == "1"


# ── Full parse_extraction_result ──────────────────────────────────────

class TestParseExtractionResult:
    def test_valid_output(self):
        raw = json.dumps({
            "scopeItems": [
                {
                    "id": "item-1",
                    "category": "Cabinetry",
                    "description": "Wall cabinet 17 l/f",
                    "quantity": 17,
                    "unit": "lf",
                    "confidence": 0.95,
                    "pageRefs": [1],
                    "textSnippets": ["Wall cabinet 17 LF"],
                }
            ],
            "specSheet": [
                {
                    "key": "Cabinet Finish",
                    "value": "PL-1",
                    "confidence": 0.9,
                    "pageRefs": [1],
                    "textSnippets": ["Finish: PL-1"],
                }
            ],
            "specifications": [
                {
                    "finish_code": "PL-1",
                    "finish_name": "Plastic Laminate",
                    "category": "Casework",
                    "manufacturer": "Wilsonart",
                    "description": "High pressure laminate",
                    "finish": "Matte",
                }
            ],
        })

        result = parse_extraction_result(raw)
        assert isinstance(result, ExtractionResult)
        assert len(result.scopeItems) == 1
        assert result.scopeItems[0].unit == UnitType.LF
        assert result.scopeItems[0].quantity == 17
        assert result.scopeItems[0].confidence == 0.95
        assert len(result.specSheet) == 1
        assert result.specSheet[0].key == "Cabinet Finish"
        assert len(result.specifications) == 1

    def test_uppercase_units_normalized(self):
        raw = json.dumps({
            "scopeItems": [
                {
                    "category": "Fixtures",
                    "description": "Toilet",
                    "quantity": 5,
                    "unit": "EA",
                    "confidence": 0.9,
                }
            ],
            "specSheet": [],
            "specifications": [],
        })
        result = parse_extraction_result(raw)
        assert result.scopeItems[0].unit == UnitType.EA

    def test_confidence_100_normalized(self):
        raw = json.dumps({
            "scopeItems": [
                {
                    "category": "Lighting",
                    "description": "Recessed light",
                    "quantity": 12,
                    "unit": "ea",
                    "confidence": 95,  # 0-100 instead of 0-1
                }
            ],
            "specSheet": [],
            "specifications": [],
        })
        result = parse_extraction_result(raw)
        assert result.scopeItems[0].confidence == 0.95

    def test_empty_output(self):
        result = parse_extraction_result("I couldn't find any items.")
        assert result.scopeItems == []
        assert result.specSheet == []
        assert result.specifications == []

    def test_code_fence_wrapped(self):
        inner = json.dumps({"scopeItems": [
            {"category": "Cabinetry", "description": "Base cab", "quantity": 5, "unit": "lf", "confidence": 0.8}
        ], "specSheet": [], "specifications": []})

        raw = f"Here is the analysis:\n```json\n{inner}\n```\nDone."
        result = parse_extraction_result(raw)
        assert len(result.scopeItems) == 1

    def test_missing_fields_default(self):
        raw = json.dumps({
            "scopeItems": [
                {"category": "Hardware", "description": "Pull", "quantity": 10, "unit": "ea", "confidence": 0.7}
            ],
        })
        result = parse_extraction_result(raw)
        assert result.specSheet == []
        assert result.specifications == []

    def test_feet_inches_quantity(self):
        raw = json.dumps({
            "scopeItems": [
                {
                    "category": "Countertops",
                    "description": "Counter",
                    "quantity": "13'-6\"",
                    "unit": "lf",
                    "confidence": 0.85,
                }
            ],
            "specSheet": [],
            "specifications": [],
        })
        result = parse_extraction_result(raw)
        assert result.scopeItems[0].quantity == pytest.approx(13.5)
