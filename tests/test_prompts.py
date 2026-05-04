"""
Tests for the trade prompt builder.
"""
from prompts.trade_prompts import build_prompt, SYSTEM_PROMPTS, TRADE_CATEGORIES


class TestBuildPrompt:
    def test_returns_tuple(self):
        sys, usr = build_prompt("millwork")
        assert isinstance(sys, str)
        assert isinstance(usr, str)

    def test_all_trades(self):
        for trade in ("millwork", "plumbing", "electrical", "flooring"):
            sys, usr = build_prompt(trade)
            assert trade in usr.lower()
            assert "scopeItems" in usr
            assert "unit" in usr

    def test_ocr_text_included(self):
        _, usr = build_prompt("millwork", "Wall cabinet 17 LF")
        assert "Wall cabinet 17 LF" in usr

    def test_empty_ocr(self):
        _, usr = build_prompt("millwork", "")
        assert "OCR-EXTRACTED TEXT" not in usr

    def test_unknown_trade_defaults_to_millwork(self):
        sys, _ = build_prompt("unknown_trade")
        assert "millwork" in sys.lower() or "cabinetry" in sys.lower()

    def test_categories_in_prompt(self):
        for trade, cats in TRADE_CATEGORIES.items():
            _, usr = build_prompt(trade)
            for cat in cats:
                assert cat in usr
