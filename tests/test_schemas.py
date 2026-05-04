"""
Tests for the Pydantic extraction schemas.
"""
import pytest
from pydantic import ValidationError

from schemas.extraction import (
    ScopeItem,
    SpecSheetEntry,
    Specification,
    ExtractionResult,
    UnitType,
    Trade,
)


class TestScopeItem:
    def test_valid_item(self):
        item = ScopeItem(
            category="Cabinetry",
            description="Wall cabinet 17 l/f",
            quantity=17,
            unit="lf",
            confidence=0.95,
        )
        assert item.unit == UnitType.LF
        assert item.id.startswith("item-")

    def test_uppercase_unit_lowered(self):
        item = ScopeItem(
            category="Hardware",
            description="Pulls",
            quantity=10,
            unit="EA",
            confidence=0.8,
        )
        assert item.unit == UnitType.EA

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            ScopeItem(
                category="Test",
                description="Test",
                quantity=1,
                unit="ea",
                confidence=1.5,
            )

    def test_negative_quantity(self):
        with pytest.raises(ValidationError):
            ScopeItem(
                category="Test",
                description="Test",
                quantity=-1,
                unit="ea",
                confidence=0.5,
            )


class TestExtractionResult:
    def test_empty(self):
        result = ExtractionResult()
        assert result.scopeItems == []
        assert result.specSheet == []
        assert result.specifications == []

    def test_full(self):
        result = ExtractionResult(
            scopeItems=[
                ScopeItem(
                    category="Cabinetry",
                    description="Base cab",
                    quantity=5,
                    unit="lf",
                    confidence=0.9,
                )
            ],
            specSheet=[
                SpecSheetEntry(
                    key="Finish",
                    value="PL-1",
                    confidence=0.9,
                )
            ],
            specifications=[
                Specification(
                    finish_code="PL-1",
                    finish_name="Plastic Laminate",
                    category="Casework",
                )
            ],
        )
        assert len(result.scopeItems) == 1
        assert result.specSheet[0].value == "PL-1"


class TestTrade:
    def test_valid_trades(self):
        for t in ("millwork", "plumbing", "electrical", "flooring"):
            assert Trade(t).value == t

    def test_invalid_trade(self):
        with pytest.raises(ValueError):
            Trade("hvac")
