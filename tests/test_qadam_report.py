import json
import unittest
from dataclasses import replace

from qadam_engine import CASES, make_plans
from qadam_report import build_case_report


class CaseReportTests(unittest.TestCase):
    def test_report_contains_current_and_recommended_amortization(self) -> None:
        case = replace(
            CASES[0],
            income=1_000_000,
            income_drop=0.0,
            balance=12_000,
            annual_rate=12.0,
            current_payment=1_000,
            other_debt_payment=0.0,
            essential_expenses=100_000,
            remaining_months=2,
        )
        plans = make_plans(
            case,
            budget_share=1.0,
            max_extension_months=12,
            max_grace_months=1,
        )

        report = json.loads(
            build_case_report(
                case,
                plans,
                objective="affordable",
                budget_share=1.0,
                max_extension_months=12,
                max_grace_months=1,
            )
        )

        self.assertEqual(report["schema_version"], "2.1")
        self.assertEqual(len(report["amortization_schedules"]["current"]), 2)
        self.assertIsNotNone(report["recommendation"])
        self.assertNotEqual(report["recommendation"]["key"], "current")
        recommended_rows = report["amortization_schedules"]["recommended"]
        self.assertEqual(
            len(recommended_rows), report["recommendation"]["term_months"]
        )
        self.assertAlmostEqual(
            sum(row["payment"] for row in recommended_rows),
            report["recommendation"]["total_repayment"],
        )

    def test_report_omits_recommendation_amortization_when_none_is_feasible(self) -> None:
        case = replace(
            CASES[0],
            income=100_000,
            income_drop=0.0,
            essential_expenses=100_000,
            other_debt_payment=0.0,
        )
        plans = make_plans(case, budget_share=1.0, max_extension_months=0)

        report = json.loads(
            build_case_report(
                case,
                plans,
                objective="balanced",
                budget_share=1.0,
                max_extension_months=0,
                max_grace_months=3,
            )
        )

        self.assertIsNone(report["recommendation"])
        self.assertIsNone(report["amortization_schedules"]["recommended"])
        self.assertEqual(len(report["amortization_schedules"]["current"]), case.remaining_months)


if __name__ == "__main__":
    unittest.main()
