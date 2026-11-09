"""Downloadable reports for the QaryzFlow demonstration app."""

from __future__ import annotations

from dataclasses import asdict
from html import escape
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
    monthly_budget,
    objective_explanation,
    sort_plans,
    stress_details,
)


def build_case_report(
    case: BorrowerCase,
    plans: list[Plan],
    objective: str,
    budget_share: float,
    max_extension_months: int,
    max_grace_months: int,
    *,
    review_plan_key: str | None = None,
) -> str:
    current = next(plan for plan in plans if plan.key == "current")
    recommended = choose_plan(plans, objective)
    ordered = sort_plans(plans, objective)
    frontier = [plan for plan in plans if plan.is_pareto_optimal]
    review_plan = recommended
    if review_plan_key is not None:
        review_plan = next((plan for plan in plans if plan.key == review_plan_key and plan.is_feasible), None)
        if review_plan is None:
            raise ValueError("Для обсуждения можно выгрузить только допустимый график.")
    payload = {
        "project": "QaryzFlow",
        "schema_version": "2.3",
        "report_type": "synthetic_schedule_search",
        "data_notice": "All example profiles are fictional. This report contains no empirical default probability or real-customer data.",
        "case": asdict(case),
        "calculated_budget": {
            "usual_monthly_income": case.income,
            "income_decline": case.income_drop,
            "income_after_decline": effective_income(case),
            "income_recovery_months": case.income_recovery_months,
            "income_recovery_wait_months": case.income_recovery_wait_months,
            "free_cash_before_reviewed_payment": free_cash(case),
        },
        "monthly_budget": [
            asdict(month)
            for month in monthly_budget(case, budget_share, max(plan.term_months for plan in plans))
        ],
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
            "income_recovery": "income stays reduced during the waiting period, then rises in equal monthly steps; zero recovery months keeps income reduced",
            "monthly_affordability": "free cash must be nonnegative and the payment must fit the limit in every month; budget_check_month identifies the smallest margin",
            "fees_insurance_penalties_and_contract_rules": "not modeled",
        },
        "current_schedule": asdict(current),
        "recommendation": asdict(recommended) if recommended else None,
        "selected_for_review": asdict(review_plan) if review_plan else None,
        "amortization_schedules": {
            "current": [asdict(row) for row in amortization_schedule(case, current)],
            "recommended": (
                [asdict(row) for row in amortization_schedule(case, recommended)]
                if recommended
                else None
            ),
            "selected_for_review": (
                [asdict(row) for row in amortization_schedule(case, review_plan)]
                if review_plan else None
            ),
        },
        "selected_stress_checks": (
            [asdict(check) for check in stress_details(case, review_plan, budget_share)]
            if review_plan else []
        ),
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


def build_review_summary(case: BorrowerCase, current: Plan, selected: Plan, budget_share: float) -> str:
    """Create a compact, printable comparison for a budget-approved schedule."""
    if not selected.is_feasible:
        raise ValueError("Для обсуждения можно выгрузить только допустимый график.")

    def amount(value: float) -> str:
        return f"{value:,.0f} ₸".replace(",", " ")

    rows = (
        ("Срок", f"{current.term_months} мес.", f"{selected.term_months} мес."),
        ("Пауза", f"{current.grace_months} мес.", f"{selected.grace_months} мес."),
        ("Обычный платёж", amount(current.monthly_payment), amount(selected.monthly_payment)),
        ("Максимальный платёж", amount(current.peak_monthly_payment), amount(selected.peak_monthly_payment)),
        ("Всего выплат", amount(current.total_repayment), amount(selected.total_repayment)),
    )
    table = "".join(f"<tr><th scope='row'>{label}</th><td>{before}</td><td>{after}</td></tr>" for label, before, after in rows)
    checks = []
    for check in stress_details(case, selected, budget_share):
        result = "Проходит" if check.passed else f"Месяц {check.shortfall_month}, не хватает {amount(check.shortfall)}"
        checks.append(f"<li>{escape(check.name)} — {result}</li>")
    income_path = "Доход постоянный." if case.income_drop <= 0 else "Снижение дохода сохраняется весь срок."
    if case.income_drop > 0 and case.income_recovery_months > 0:
        wait = max(1, case.income_recovery_wait_months)
        income_path = f"Доход растёт с месяца {wait + 1} и возвращается к обычному в месяце {wait + case.income_recovery_months}."

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>QaryzFlow — график для обсуждения</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; color: #132c26; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.8rem; margin: .3rem 0; }} h2 {{ font-size: 1.15rem; margin-top: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #cad7d2; }}
.note {{ color: #425850; }} @media print {{ body {{ margin: 0; padding: 0; }} table {{ break-inside: avoid; }} }}
</style></head><body>
<p class="note">QARYZFLOW · УЧЕБНЫЙ КЕЙС {escape(case.case_id)}</p>
<h1>График для обсуждения</h1>
<p>{escape(selected.title)}. Остаток долга {amount(case.balance)}. Номинальная ставка {case.annual_rate:g}% годовых.</p>
<table><thead><tr><th scope="col">Показатель</th><th scope="col">Текущий график</th><th scope="col">График для обсуждения</th></tr></thead>
<tbody>{table}</tbody></table>
<h2>Бюджет и стресс-проверки</h2>
<p>Лимит составляет {budget_share:.0%} свободного бюджета своего месяца. {income_path} В месяце {selected.budget_check_month} платёж {amount(selected.budget_check_payment)} при лимите {amount(selected.budget_limit)}.</p>
<ul>{''.join(checks)}</ul>
<p class="note">Учебный расчёт для обсуждения. Суммы округлены до тенге. Комиссии, страховки и штрафы в расчёт не входят.</p>
</body></html>"""
