from dataclasses import replace
import unittest

from qadam_engine import (
    CASES,
    amortization_schedule,
    annuity_payment,
    choose_plan,
    free_cash,
    make_plans,
    monthly_budget,
    monthly_schedule,
    sort_plans,
    stress_details,
)


class ScheduleOptimizerTests(unittest.TestCase):
    def test_demo_cases_cover_pause_balloon_deficit_and_zero_rate(self) -> None:
        cases = {case.case_id: case for case in CASES}
        recovery = choose_plan(make_plans(cases["Q-1301"]), "cost")
        self.assertEqual(recovery.grace_months, 3)

        balloon = make_plans(cases["Q-1320"], max_extension_months=0)
        self.assertLess(balloon[0].monthly_payment, balloon[0].budget_limit)
        self.assertEqual(balloon[0].budget_check_month, 12)
        self.assertFalse(balloon[0].is_feasible)
        self.assertEqual(choose_plan(balloon, "cost").key, "term-12-grace-0")

        self.assertLess(free_cash(cases["Q-1334"]), 0.0)
        self.assertIsNone(choose_plan(make_plans(cases["Q-1334"])))

        zero_rate = make_plans(cases["Q-1370"])
        self.assertTrue(zero_rate[0].is_feasible)
        self.assertEqual(zero_rate[0].stress_passes, 4)
        self.assertEqual(choose_plan(zero_rate, "cost").key, "current")
        self.assertLess(choose_plan(zero_rate, "affordable").peak_monthly_payment, zero_rate[0].peak_monthly_payment)

    def test_annuity_handles_zero_rate_and_positive_rate(self) -> None:
        self.assertAlmostEqual(annuity_payment(120_000, 0.0, 12), 10_000)
        self.assertGreater(annuity_payment(120_000, 24.0, 12), 10_000)

    def test_search_enumerates_each_term_and_grace_pair(self) -> None:
        case = replace(CASES[0], remaining_months=4)
        plans = make_plans(case, max_extension_months=2, max_grace_months=1)
        self.assertEqual(len(plans), 7)
        self.assertEqual(sum(plan.key == "current" for plan in plans), 1)
        self.assertEqual(len({plan.key for plan in plans}), len(plans))

    def test_recalculation_at_same_term_can_be_the_only_feasible_plan(self) -> None:
        case = replace(
            CASES[0], income=100_000, income_drop=0.0, essential_expenses=80_000,
            other_debt_payment=0.0, balance=120_000, annual_rate=0.0,
            current_payment=5_000, remaining_months=12,
        )
        plans = make_plans(case, max_extension_months=0, max_grace_months=0)

        self.assertFalse(plans[0].is_feasible)
        chosen = choose_plan(plans)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.key, "term-12-grace-0")
        self.assertEqual(chosen.monthly_payment, 10_000)

    def test_pause_can_bridge_temporary_reduction_in_income(self) -> None:
        case = replace(
            CASES[0], income=120_000, income_drop=0.25, essential_expenses=80_000,
            other_debt_payment=0.0, balance=120_000, annual_rate=0.0,
            current_payment=20_000, remaining_months=6,
        )
        settings = dict(budget_share=1.0, max_extension_months=0, max_grace_months=2)
        self.assertIsNone(choose_plan(make_plans(case, **settings)))

        recovering = replace(case, income_recovery_months=2)
        self.assertEqual([month.income for month in monthly_budget(recovering, 1.0, 4)], [90_000, 105_000, 120_000, 120_000])
        plans = make_plans(recovering, **settings)
        chosen = choose_plan(plans, "cost")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.grace_months, 1)
        self.assertEqual(chosen.budget_check_month, 2)
        self.assertEqual(chosen.budget_limit, 25_000)
        self.assertEqual(chosen.budget_check_payment, 24_000)
        self.assertAlmostEqual(chosen.payment_share_of_free_cash, 0.96)
        two_month_pause = next(plan for plan in plans if plan.grace_months == 2)
        self.assertTrue(dict(two_month_pause.stress_results)["Расходы выше на 10%"])

        deficit = replace(recovering, essential_expenses=92_000)
        self.assertIsNone(choose_plan(make_plans(deficit, **settings)))

    def test_delayed_recovery_changes_the_necessary_pause_and_stress_month(self) -> None:
        case = replace(CASES[3], income_recovery_months=3, income_recovery_wait_months=3)
        self.assertEqual([round(month.payment_limit) for month in monthly_budget(case, 0.8, 6)],
                         [54_160, 54_160, 54_160, 90_240, 126_320, 162_400])
        plans = make_plans(case)
        selected = choose_plan(plans, "cost")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.key, "term-20-grace-3")
        checks = stress_details(case, selected, 0.8)
        self.assertEqual(tuple((check.name, check.passed) for check in checks), selected.stress_results)
        expense_check = next(check for check in checks if check.name == "Расходы выше на 10%")
        self.assertEqual(expense_check.shortfall_month, 4)
        self.assertAlmostEqual(expense_check.shortfall, 10_379, delta=1)

    def test_dominated_plans_do_not_change_balanced_choice(self) -> None:
        without_pause = make_plans(CASES[3], max_grace_months=0)
        with_pause = make_plans(CASES[3], max_grace_months=3)
        self.assertGreater(sum(plan.is_feasible for plan in with_pause), sum(plan.is_feasible for plan in without_pause))
        self.assertEqual(choose_plan(without_pause).key, "term-45-grace-0")
        self.assertEqual(choose_plan(with_pause).key, "term-45-grace-0")

    def test_final_balance_is_checked_in_its_own_recovered_month(self) -> None:
        case = replace(
            CASES[0], income=125_000, income_drop=0.2, essential_expenses=85_000,
            other_debt_payment=0.0, balance=45_000, annual_rate=0.0,
            current_payment=8_000, remaining_months=3,
        )
        self.assertFalse(make_plans(case, max_extension_months=0)[0].is_feasible)

        current = make_plans(replace(case, income_recovery_months=2), max_extension_months=0)[0]
        self.assertTrue(current.is_feasible)
        self.assertEqual(current.budget_check_month, 3)
        self.assertEqual(current.budget_check_payment, 29_000)
        self.assertEqual(current.budget_limit, 32_000)

    def test_budget_limit_is_a_hard_constraint(self) -> None:
        case = replace(
            CASES[0],
            income=300_000,
            income_drop=0.0,
            essential_expenses=100_000,
            other_debt_payment=0.0,
        )
        plans = make_plans(case, budget_share=0.5, max_extension_months=60, max_grace_months=2)
        chosen = choose_plan(plans, "balanced")
        self.assertIsNotNone(chosen)
        self.assertFalse(next(plan for plan in plans if plan.key == "current").is_feasible)
        for plan in plans:
            if plan.is_feasible:
                self.assertLessEqual(plan.peak_monthly_payment, max(0.0, free_cash(case)) * 0.5 + 0.01)

    def test_no_affordable_plan_returns_no_recommendation(self) -> None:
        case = replace(
            CASES[0],
            income=100_000,
            income_drop=0.0,
            essential_expenses=100_000,
            other_debt_payment=0.0,
        )
        plans = make_plans(case, max_extension_months=12)
        self.assertIsNone(choose_plan(plans, "balanced"))
        self.assertFalse(any(plan.is_pareto_optimal for plan in plans))

    def test_affordable_and_cost_objectives_have_distinct_tradeoffs(self) -> None:
        case = replace(
            CASES[4],
            income=1_000_000,
            income_drop=0.0,
            essential_expenses=100_000,
            other_debt_payment=0.0,
        )
        plans = make_plans(case, budget_share=1.0, max_extension_months=48)
        affordable = choose_plan(plans, "affordable")
        lower_cost = choose_plan(plans, "cost")
        self.assertIsNotNone(affordable)
        self.assertIsNotNone(lower_cost)
        self.assertLessEqual(affordable.monthly_payment, lower_cost.monthly_payment)
        self.assertLess(affordable.monthly_payment, lower_cost.monthly_payment)
        self.assertLessEqual(lower_cost.total_repayment, affordable.total_repayment)

    def test_robust_objective_prioritizes_stress_scenarios(self) -> None:
        case = replace(
            CASES[3],
            income=600_000,
            income_drop=0.0,
            essential_expenses=150_000,
            other_debt_payment=50_000,
        )
        plans = make_plans(case, budget_share=1.0, max_extension_months=48)
        chosen = choose_plan(plans, "robust")
        best_pass_count = max(plan.stress_passes for plan in plans if plan.is_feasible)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.stress_passes, best_pass_count)

    def test_current_schedule_includes_final_balance_payment(self) -> None:
        case = replace(
            CASES[0],
            balance=100_000,
            annual_rate=0.0,
            current_payment=1_000,
            remaining_months=3,
        )
        current = next(plan for plan in make_plans(case) if plan.key == "current")
        payments = monthly_schedule(case, current, 3)
        self.assertEqual(current.final_balance_payment, 97_000)
        self.assertEqual(payments, [1_000, 1_000, 98_000])
        self.assertAlmostEqual(sum(payments), current.total_repayment)

        rows = amortization_schedule(case, current)
        self.assertEqual([row.payment for row in rows], payments)
        self.assertAlmostEqual(rows[-1].principal_repaid, 98_000)
        self.assertAlmostEqual(rows[-1].closing_balance, 0.0)

    def test_grace_period_capitalizes_interest_before_payments_start(self) -> None:
        case = replace(
            CASES[0],
            balance=12_000,
            annual_rate=12.0,
            remaining_months=2,
        )
        plans = make_plans(case, budget_share=1.0, max_extension_months=0, max_grace_months=1)
        deferred = next(plan for plan in plans if plan.key == "term-2-grace-1")
        expected_payment = annuity_payment(12_120, 12.0, 1)

        self.assertAlmostEqual(deferred.monthly_payment, expected_payment)
        self.assertEqual(monthly_schedule(case, deferred, 2), [0.0, expected_payment])
        self.assertAlmostEqual(deferred.total_repayment, expected_payment)

        rows = amortization_schedule(case, deferred)
        self.assertEqual(rows[0].payment, 0.0)
        self.assertAlmostEqual(rows[0].accrued_interest, 120.0)
        self.assertEqual(rows[0].interest_paid, 0.0)
        self.assertAlmostEqual(rows[0].capitalized_interest, 120.0)
        self.assertAlmostEqual(rows[0].closing_balance, 12_120.0)
        self.assertAlmostEqual(rows[1].principal_repaid, 12_120.0)
        self.assertAlmostEqual(rows[1].closing_balance, 0.0)
        self.assertAlmostEqual(sum(row.payment for row in rows), deferred.total_repayment)

    def test_amortization_schedule_capitalizes_interest_shortfall(self) -> None:
        case = replace(
            CASES[0],
            balance=100_000,
            annual_rate=24.0,
            current_payment=500,
            remaining_months=2,
        )
        current = next(plan for plan in make_plans(case) if plan.key == "current")

        first_row = amortization_schedule(case, current)[0]

        self.assertAlmostEqual(first_row.accrued_interest, 2_000.0)
        self.assertAlmostEqual(first_row.interest_paid, 500.0)
        self.assertEqual(first_row.principal_repaid, 0.0)
        self.assertAlmostEqual(first_row.capitalized_interest, 1_500.0)
        self.assertAlmostEqual(first_row.closing_balance, 101_500.0)

    def test_final_balance_payment_is_included_in_budget_check(self) -> None:
        case = replace(
            CASES[0],
            income=100_000,
            income_drop=0.0,
            essential_expenses=90_000,
            other_debt_payment=0.0,
            balance=100_000,
            annual_rate=0.0,
            current_payment=1_000,
            remaining_months=3,
        )
        current = next(
            plan for plan in make_plans(case, budget_share=1.0, max_extension_months=0) if plan.key == "current"
        )
        self.assertEqual(current.monthly_payment, 1_000)
        self.assertGreater(current.peak_monthly_payment, current.budget_limit)
        self.assertFalse(current.is_feasible)

    def test_stress_result_has_one_flag_per_defined_scenario(self) -> None:
        plans = make_plans(CASES[3])
        for plan in plans:
            self.assertEqual(plan.stress_total, 4)
            self.assertEqual(len(plan.stress_results), 4)
            self.assertEqual(plan.stress_passes, sum(passed for _, passed in plan.stress_results))

    def test_sort_keeps_feasible_candidates_ahead_of_infeasible_ones(self) -> None:
        plans = make_plans(CASES[3])
        sorted_plans = sort_plans(plans, "balanced")
        first_infeasible = next((i for i, plan in enumerate(sorted_plans) if not plan.is_feasible), len(sorted_plans))
        self.assertTrue(all(plan.is_feasible for plan in sorted_plans[:first_infeasible]))
        self.assertTrue(all(not plan.is_feasible for plan in sorted_plans[first_infeasible:]))


if __name__ == "__main__":
    unittest.main()
