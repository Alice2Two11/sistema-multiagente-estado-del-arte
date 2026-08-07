from __future__ import annotations

import copy
import unittest

from src.config.draft_writing_policy_config import get_draft_writing_policy
from src.tools.draft_writing.source_aware_budgets import (
    assign_source_aware_section_budgets,
    classify_section_for_budget,
)


def section(section_id: str, title: str = "Body", **values):
    result = {"section_id": section_id, "section_title": title}
    result.update(values)
    return result


class SourceAwareBudgetTests(unittest.TestCase):
    def setUp(self):
        self.policy = get_draft_writing_policy()

    def assign(self, sections, total=1000, policy=None):
        return assign_source_aware_section_budgets(
            sections,
            total,
            policy=self.policy if policy is None else policy,
        )

    def test_no_sections_with_zero_target_returns_empty_budget(self):
        self.assertEqual(self.assign([], total=0), {})

    def test_no_sections_with_positive_target_fails_explicitly(self):
        with self.assertRaisesRegex(
            ValueError,
            "^SOURCE_AWARE_BUDGET_IMPOSSIBLE:target_words_without_sections$",
        ):
            self.assign([], total=100)

    def test_one_substantive_section_receives_the_full_target(self):
        budgets = self.assign(
            [section("S1", papers_to_use=["paper.pdf"])],
            total=137,
        )
        self.assertEqual(budgets["S1"]["target_words"], 137)
        self.assertEqual(budgets["S1"]["budget_type"], "source_aware_substantive")

    def test_zero_organizational_sections_distribute_exactly(self):
        sections = [
            section("S1", papers_to_use=["a.pdf"]),
            section("S2", papers_to_use=["b.pdf"]),
            section("S3", papers_to_use=["c.pdf"]),
        ]
        budgets = self.assign(sections, total=1000)
        self.assertEqual([budgets[key]["target_words"] for key in budgets], [334, 333, 333])

    def test_one_organizational_section_uses_preferred_target(self):
        sections = [
            section("S0", "Introducción", papers_to_use=[]),
            section("S1", papers_to_use=["a.pdf"]),
            section("S2", papers_to_use=["b.pdf"]),
        ]
        budgets = self.assign(sections, total=1000)
        self.assertEqual(budgets["S0"]["target_words"], 40)
        self.assertEqual(budgets["S1"]["target_words"], 480)
        self.assertEqual(budgets["S2"]["target_words"], 480)

    def test_multiple_organizational_sections(self):
        sections = [
            section("S0", requires_sources=False),
            section("S1", source_requirement="none"),
            section("S2", papers_to_use=["a.pdf"]),
        ]
        budgets = self.assign(sections, total=1000)
        self.assertEqual(budgets["S0"]["target_words"], 40)
        self.assertEqual(budgets["S1"]["target_words"], 40)
        self.assertEqual(budgets["S2"]["target_words"], 920)

    def test_only_organizational_sections_distribute_exactly(self):
        sections = [
            section("S0", requires_sources=False),
            section("S1", section_type="conclusion"),
            section("S2", source_requirement="source_free"),
        ]
        budgets = self.assign(sections, total=100)
        self.assertEqual([budgets[key]["target_words"] for key in budgets], [34, 33, 33])
        self.assertTrue(all(item["budget_type"] == "source_free_organizational" for item in budgets.values()))

    def test_only_organizational_sections_fail_above_configured_maximum(self):
        sections = [section("S0", requires_sources=False), section("S1", requires_sources=False)]
        with self.assertRaisesRegex(
            ValueError,
            "^SOURCE_AWARE_BUDGET_IMPOSSIBLE:organizational_only_total_outside_configured_bounds$",
        ):
            self.assign(sections, total=1000)

    def test_exact_division(self):
        sections = [
            section("O1", requires_sources=False),
            section("O2", requires_sources=False),
            section("S1", papers_to_use=["a"]),
            section("S2", papers_to_use=["b"]),
        ]
        budgets = self.assign(sections, total=1000)
        self.assertEqual([budgets[key]["target_words"] for key in budgets], [40, 40, 460, 460])

    def test_residue_is_assigned_in_stable_section_order(self):
        sections = [
            section("O1", requires_sources=False),
            section("O2", requires_sources=False),
            section("S1", papers_to_use=["a"]),
            section("S2", papers_to_use=["b"]),
            section("S3", papers_to_use=["c"]),
        ]
        budgets = self.assign(sections, total=1000)
        self.assertEqual([budgets[key]["target_words"] for key in budgets], [40, 40, 307, 307, 306])

    def test_very_small_valid_target(self):
        budgets = self.assign(
            [section("S1", section_type="substantive")],
            total=1,
        )
        self.assertEqual(budgets["S1"]["target_words"], 1)
        self.assertEqual(budgets["S1"]["minimum_words"], 1)
        self.assertEqual(budgets["S1"]["maximum_words"], 1)

    def test_target_smaller_than_section_count_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "^SOURCE_AWARE_BUDGET_IMPOSSIBLE:target_total_words_smaller_than_section_count$",
        ):
            self.assign([section("S1"), section("S2")], total=1)

    def test_spanish_title_fallback(self):
        classification = classify_section_for_budget(section("S1", "Introducción"))
        self.assertTrue(classification.is_organizational)
        self.assertEqual(classification.reason, "title_fallback:organizational")

    def test_english_title_fallback(self):
        classification = classify_section_for_budget(section("S1", "Conclusion"))
        self.assertTrue(classification.is_organizational)

    def test_ambiguous_title_defaults_to_substantive(self):
        classification = classify_section_for_budget(
            section("S1", "Introduction to neural forecasting")
        )
        self.assertFalse(classification.is_organizational)
        self.assertEqual(classification.reason, "title_fallback:substantive_default")

    def test_no_papers_but_structurally_substantive(self):
        classification = classify_section_for_budget(
            section("S1", papers_to_use=[], requires_sources=True)
        )
        self.assertFalse(classification.is_organizational)

    def test_papers_assigned_but_structurally_organizational(self):
        classification = classify_section_for_budget(
            section("S1", papers_to_use=["a.pdf"], requires_sources=False)
        )
        self.assertTrue(classification.is_organizational)

    def test_structured_section_type_has_priority_over_title(self):
        classification = classify_section_for_budget(
            section("S1", "Introduction", section_type="substantive")
        )
        self.assertFalse(classification.is_organizational)

    def test_conflicting_structural_signals_fail_explicitly(self):
        with self.assertRaisesRegex(
            ValueError,
            "^SOURCE_AWARE_BUDGET_CONFLICTING_STRUCTURAL_SIGNALS:S1$",
        ):
            classify_section_for_budget(
                section("S1", requires_sources=True, section_type="organizational")
            )

    def test_results_are_deterministic(self):
        sections = [
            section("O1", requires_sources=False),
            section("S1", papers_to_use=["a"]),
            section("S2", papers_to_use=["b"]),
            section("S3", papers_to_use=["c"]),
        ]
        first = self.assign(copy.deepcopy(sections), total=1000)
        second = self.assign(copy.deepcopy(sections), total=1000)
        self.assertEqual(first, second)

    def test_target_sum_is_exact_across_representative_cases(self):
        cases = [
            ([section("S1", papers_to_use=["a"])], 17),
            ([section("S1", papers_to_use=["a"]), section("S2", papers_to_use=["b"])], 101),
            ([section("O1", requires_sources=False), section("S1", papers_to_use=["a"]), section("S2", papers_to_use=["b"])], 1000),
            ([section("O1", requires_sources=False), section("O2", requires_sources=False)], 79),
        ]
        for sections, total in cases:
            with self.subTest(total=total, sections=len(sections)):
                budgets = self.assign(sections, total=total)
                self.assertEqual(sum(item["target_words"] for item in budgets.values()), total)

    def test_minimum_and_maximum_limits_are_present_and_consistent(self):
        sections = [
            section("O1", requires_sources=False),
            section("S1", papers_to_use=["a"]),
        ]
        budgets = self.assign(sections, total=200)
        for item in budgets.values():
            self.assertLessEqual(item["minimum_words"], item["target_words"])
            self.assertGreaterEqual(item["maximum_words"], item["target_words"])

    def test_policy_bounds_are_applied_to_organizational_sections(self):
        policy = get_draft_writing_policy(
            {
                "organizational_target_words": 12,
                "organizational_minimum_words": 10,
                "organizational_maximum_words": 20,
            }
        )
        budgets = self.assign(
            [section("O1", requires_sources=False), section("S1", requires_sources=True)],
            total=100,
            policy=policy,
        )
        self.assertEqual(budgets["O1"], {
            "target_words": 12,
            "minimum_words": 10,
            "maximum_words": 20,
            "budget_type": "source_free_organizational",
            "classification_reason": "structured:requires_sources",
        })

    def test_invalid_policy_is_rejected_by_contractual_policy_validation(self):
        invalid_policy = dict(self.policy)
        invalid_policy["organizational_minimum_words"] = 50
        invalid_policy["organizational_target_words"] = 40
        with self.assertRaisesRegex(
            ValueError,
            "^DRAFT_POLICY_INVALID:organizational_minimum_words:must_not_exceed_organizational_target_words$",
        ):
            self.assign([section("S1")], total=100, policy=invalid_policy)

    def test_duplicate_section_ids_fail(self):
        with self.assertRaisesRegex(
            ValueError,
            "^SOURCE_AWARE_BUDGET_INVALID:section_id:must_be_unique$",
        ):
            self.assign([section("S1"), section("S1")], total=100)


if __name__ == "__main__":
    unittest.main()
