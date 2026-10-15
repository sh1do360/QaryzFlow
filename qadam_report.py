"""JSON report creation for the QaryzFlow demonstration app."""

from __future__ import annotations

from dataclasses import asdict
import json

from qadam_engine import (
    OBJECTIVES,
    STRESS_SCENARIOS,
    BorrowerCase,
    Plan,
    amortization_schedule,
    choose_plan,
    effective_income,
    free_cash,
    objective_explanation,
    sort_plans,
)


def build_case_report(
    case: BorrowerCase,
    plans: list[Plan],
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
) -> str:
    current = next(plan for plan in plans if plan.key == "current")
    recommended = choose_plan(plans, objective)
    ordered = sort_plans(plans, objective)
    frontier = [plan for plan in plans if plan.is_pareto_optimal]
    payload = {
        "project": "QaryzFlow",
        "schema_version": "2.1",
        "report_type": "synthetic_schedule_search",
        "data_notice": "All example profiles are fictional. This report contains no empirical default probability or real-customer data.",
        "case": asdict(case),
        "calculated_budget": {
            "usual_monthly_income": case.income,
            "income_decline": case.income_drop,
            "income_after_decline": effective_income(case),
            "free_cash_before_reviewed_payment": free_cash(case),
        },
        "search_settings": {
            "objective": objective,
            "objective_label": OBJECTIVES[objective],
            "objective_rule": objective_explanation(objective),
            "maximum_payment_share_of_free_cash": budget_share,
            "maximum_term_extension_months": max_extension_months,
            "maximum_grace_months": max_grace_months,
            "candidate_count": len(plans),
            "feasible_candidate_count": sum(plan.is_feasible for plan in plans),
        },
        "calculation_assumptions": {
            "payment_method": "level-payment annuity",
            "monthly_rate": "nominal annual rate divided by 12",
            "grace_period": "interest is capitalized; regular payments start after the grace period",
            "current_schedule": "entered payment is kept; any unpaid balance at the end is shown as a final balance payment",
            "fees_insurance_penalties_and_contract_rules": "not modeled",
        },
        "current_schedule": asdict(current),
        "recommendation": asdict(recommended) if recommended else None,
        "amortization_schedules": {
            "current": [asdict(row) for row in amortization_schedule(case, current)],
            "recommended": (
                [asdict(row) for row in amortization_schedule(case, recommended)]
                if recommended
                else None
            ),
        },
        "pareto_frontier": [
            asdict(plan)
            for plan in sorted(
                frontier,
                key=lambda item: (item.peak_monthly_payment, item.total_repayment),
            )
        ],
        "all_candidates_in_objective_order": [asdict(plan) for plan in ordered],
        "stress_scenarios": [
            {
                "name": name,
                "income_decline": income_shock,
                "essential_expense_increase": expense_shock,
            }
            for name, income_shock, expense_shock in STRESS_SCENARIOS
        ],
        "stress_notice": "These are deterministic affordability checks under stated assumptions, not probabilities or forecasts.",
        "notice": "Demonstration artifact only. It is not a credit, collection, restructuring, or contract decision.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
