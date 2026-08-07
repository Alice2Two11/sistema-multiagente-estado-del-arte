from __future__ import annotations

import unittest

from src.tools.draft_writing.numeric_literals import (
    historical_number_exists_in_text,
    numeric_literal_exists_in_text,
    parse_numeric_literal,
)
from src.tools.draft_writing.validation import number_exists_in_text as validator_historical_rule


class NumericLiteralTests(unittest.TestCase):
    def test_parse_decimal_and_percentage(self):
        literal = parse_numeric_literal(" -10,25 % ")
        self.assertIsNotNone(literal)
        self.assertEqual(literal.canonical_value, "-10.25%")
        self.assertTrue(literal.is_percentage)

    def test_decimal_comma_and_point_are_equivalent(self):
        self.assertTrue(numeric_literal_exists_in_text("1.34", "RMSE = 1,34 MJ"))
        self.assertTrue(numeric_literal_exists_in_text("1,34", "RMSE = 1.34 MJ"))

    def test_percentage_allows_spacing_before_symbol(self):
        self.assertTrue(numeric_literal_exists_in_text("10%", "error below 10 %"))
        self.assertTrue(numeric_literal_exists_in_text("10 %", "error below 10%"))

    def test_percentage_does_not_match_nonpercentage_context(self):
        for text in ("10 min", "10 hours", "10 neurons", "10,000 iterations"):
            with self.subTest(text=text):
                self.assertFalse(numeric_literal_exists_in_text("10%", text))

    def test_nonpercentage_does_not_match_percentage_context(self):
        self.assertFalse(numeric_literal_exists_in_text("10", "error below 10%"))

    def test_does_not_match_inside_larger_decimal(self):
        self.assertFalse(numeric_literal_exists_in_text("1.3", "value = 11.3"))

    def test_does_not_match_inside_larger_integer(self):
        self.assertFalse(numeric_literal_exists_in_text("10", "value = 100"))
        self.assertFalse(numeric_literal_exists_in_text("7", "value = 70"))

    def test_signed_numbers_require_the_same_sign(self):
        self.assertTrue(numeric_literal_exists_in_text("-1.3", "MBE = -1,3"))
        self.assertFalse(numeric_literal_exists_in_text("-1.3", "MBE = 1.3"))

    def test_invalid_numeric_input_is_rejected(self):
        for value in (None, "", "approximately ten", "1.2.3"):
            with self.subTest(value=value):
                self.assertFalse(numeric_literal_exists_in_text(value, "1.2.3"))

    def test_historical_helper_exactly_matches_current_validator_on_fixtures(self):
        fixtures = (
            ("1.34", "RMSE = 1.34 MJ"),
            ("1,34", "RMSE = 1.34 MJ"),
            ("0.96", "R = 0,96"),
            ("58.7%", "58.7% of predictions"),
            ("99%", "largest error was 99%"),
            ("6.11%", "MAPE = 6.11%"),
            ("1.34", "RMSE = 2.46 MJ"),
            ("10%", "10 neurons"),
        )
        for value, text in fixtures:
            with self.subTest(value=value, text=text):
                self.assertEqual(
                    historical_number_exists_in_text(value, text),
                    validator_historical_rule(value, text),
                )

    def test_strict_utility_preserves_historical_valid_matches(self):
        fixtures = (
            ("1.34", "RMSE = 1.34 MJ"),
            ("1,34", "RMSE = 1.34 MJ"),
            ("0.96", "R = 0,96"),
            ("58.7%", "58.7% of predictions"),
            ("99%", "largest error was 99%"),
            ("6.11%", "MAPE = 6.11%"),
        )
        for value, text in fixtures:
            with self.subTest(value=value, text=text):
                self.assertTrue(validator_historical_rule(value, text))
                self.assertTrue(numeric_literal_exists_in_text(value, text))

    def test_both_rules_reject_percentage_in_nonpercentage_context(self):
        self.assertFalse(validator_historical_rule("10%", "10 neurons"))
        self.assertFalse(numeric_literal_exists_in_text("10%", "10 neurons"))

    def test_documented_incompatibility_removes_historical_containment_false_positives(self):
        fixtures = (
            ("1.3", "value = 11.3"),
            ("10", "value = 100"),
            ("7", "value = 70"),
        )
        for value, text in fixtures:
            with self.subTest(value=value, text=text):
                self.assertTrue(validator_historical_rule(value, text))
                self.assertFalse(numeric_literal_exists_in_text(value, text))

    def test_documented_incompatibility_accepts_percentage_spacing(self):
        self.assertFalse(validator_historical_rule("10%", "error below 10 %"))
        self.assertTrue(numeric_literal_exists_in_text("10%", "error below 10 %"))


if __name__ == "__main__":
    unittest.main()
