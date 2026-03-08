"""Tests for response parsing utilities."""

import pytest

from litmus.parsing import extract_answer, parse_float, parse_label, parse_multilabel


class TestExtractAnswer:
    def test_with_tags(self):
        response = "Some reasoning.\n<answer>42.0</answer>"
        assert extract_answer(response) == "42.0"

    def test_with_tags_multiline(self):
        response = "Reasoning.\n<answer>\nhigh\n</answer>\nMore text."
        assert extract_answer(response) == "high"

    def test_fallback_last_line(self):
        response = "Some reasoning.\nThe answer is 3.14"
        assert extract_answer(response) == "The answer is 3.14"

    def test_empty_response(self):
        assert extract_answer("") is None

    def test_whitespace_only(self):
        assert extract_answer("   \n  \n  ") is None

    def test_tags_with_surrounding_text(self):
        response = "I think <answer>low</answer> because..."
        assert extract_answer(response) == "low"


class TestParseFloat:
    def test_simple_float(self):
        assert parse_float("3.14") == pytest.approx(3.14)

    def test_negative_float(self):
        assert parse_float("-2.5") == pytest.approx(-2.5)

    def test_integer(self):
        assert parse_float("42") == pytest.approx(42.0)

    def test_scientific_notation(self):
        assert parse_float("1.2e-3") == pytest.approx(1.2e-3)

    def test_embedded_float(self):
        assert parse_float("The value is 3.14 units") == pytest.approx(3.14)

    def test_no_number(self):
        assert parse_float("high") is None

    def test_empty_string(self):
        assert parse_float("") is None


class TestParseLabel:
    def test_exact_match(self):
        assert parse_label("high", ["high", "low"]) == "high"

    def test_case_insensitive(self):
        assert parse_label("HIGH", ["high", "low"]) == "high"

    def test_substring_in_answer(self):
        assert parse_label("I think it is high", ["high", "low"]) == "high"

    def test_no_match(self):
        assert parse_label("maybe", ["high", "low"]) is None

    def test_empty_answer(self):
        assert parse_label("", ["high", "low"]) is None

    def test_empty_choices(self):
        assert parse_label("high", []) is None

    def test_whitespace_handling(self):
        assert parse_label("  high  ", ["high", "low"]) == "high"


class TestParseMultilabel:
    def test_comma_separated(self):
        result = parse_multilabel("cytoplasm, nucleus", ["cytoplasm", "nucleus", "membrane"])
        assert result == ["cytoplasm", "nucleus"]

    def test_semicolon_separated(self):
        result = parse_multilabel("cytoplasm; nucleus", ["cytoplasm", "nucleus", "membrane"])
        assert result == ["cytoplasm", "nucleus"]

    def test_no_match(self):
        result = parse_multilabel("unknown", ["cytoplasm", "nucleus"])
        assert result == []

    def test_empty(self):
        result = parse_multilabel("", ["cytoplasm"])
        assert result == []

    def test_deduplication(self):
        result = parse_multilabel("cytoplasm, cytoplasm", ["cytoplasm", "nucleus"])
        assert result == ["cytoplasm"]
