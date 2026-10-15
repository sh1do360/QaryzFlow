"""Transparent repayment-schedule search for the QaryzFlow demonstration app.

The engine uses invented borrower profiles and deterministic arithmetic. It does
not estimate default probabilities or make decisions about real customers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypedDict


@dataclass(frozen=True)
class BorrowerCase:
    case_id: str
    label: str
    segment: str
    region: str
    income: float
    income_drop: float
    balance: float
    annual_rate: float
    current_payment: float
    other_debt_payment: float
    essential_expenses: float
    remaining_months: int
    overdue_months: int
    savings: float


@dataclass(frozen=True)
class Plan:
    key: str
    title: str
    description: str
    monthly_payment: float
    peak_monthly_payment: float
    grace_months: int
    term_months: int
    months_with_payment: int
    total_repayment: float
    interest_cost: float
    final_balance_payment: float
    monthly_cash_after_payment: float
    budget_limit: float
    payment_share_of_free_cash: float | None
    is_feasible: bool
    feasibility_gap: float
    stress_passes: int
    stress_total: int
    stress_results: tuple[tuple[str, bool], ...]
    is_pareto_optimal: bool = False


@dataclass(frozen=True)
class AmortizationRow:
    month: int
    opening_balance: float
    payment: float
    accrued_interest: float
    interest_paid: float
    principal_repaid: float
    capitalized_interest: float
    closing_balance: float


class PortfolioItem(TypedDict):
    case_id: str
    label: str
    region: str
    segment: str
    balance: float
    current_payment: float
    recommended_plan: str
    recommended_payment: float | None
    recommended_peak_payment: float | None
    recommended_total_repayment: float | None
    stress_passes: int | None
    stress_total: int
    has_feasible_plan: bool


CASES: tuple[BorrowerCase, ...] = (
    BorrowerCase("Q-1042", "Клиент Q-1042", "Наёмный сотрудник", "Алматы", 420_000, 0.22, 1_850_000, 22.5, 131_000, 95_000, 180_000, 26, 2, 130_000),
    BorrowerCase("Q-1088", "Клиент Q-1088", "Микропредприниматель", "Шымкент", 310_000, 0.06, 950_000, 26.5, 65_000, 48_000, 150_000, 18, 1, 300_000),
    BorrowerCase("Q-1121", "Клиент Q-1121", "Наёмный сотрудник", "Астана", 680_000, 0.10, 3_200_000, 21.0, 198_000, 112_000, 250_000, 30, 3, 500_000),
    BorrowerCase("Q-1176", "Клиент Q-1176", "Самозанятый", "Караганда", 410_000, 0.33, 1_200_000, 23.9, 104_000, 62_000, 145_000, 19, 4, 25_000),
    BorrowerCase("Q-1204", "Клиент Q-1204", "Наёмный сотрудник", "Актобе", 540_000, 0.00, 2_400_000, 19.9, 118_000, 210_000, 220_000, 27, 1, 420_000),
    BorrowerCase("Q-1249", "Клиент Q-1249", "Микропредприниматель", "Тараз", 390_000, 0.13, 1_500_000, 24.0, 95_000, 80_000, 175_000, 21, 2, 160_000),
)


OBJECTIVES: dict[str, str] = {
    "balanced": "Сбалансированный",
    "affordable": "Ниже ежемесячный платёж",
    "cost": "Меньше сумма выплат",
    "robust": "Устойчивый в стресс-сценариях",
}

STRESS_SCENARIOS: tuple[tuple[str, float, float], ...] = (
    ("Доход ниже на 10%", 0.10, 0.00),
    ("Доход ниже на 20%", 0.20, 0.00),
    ("Расходы выше на 10%", 0.00, 0.10),
    ("Доход ниже на 15%, расходы выше на 10%", 0.15, 0.10),
)


def effective_income(case: BorrowerCase) -> float:
    """Apply the entered income decline to the usual monthly income."""
    drop = max(0.0, min(0.80, float(case.income_drop)))
    return max(0.0, float(case.income)) * (1.0 - drop)


def free_cash(case: BorrowerCase) -> float:
    """Monthly income left after essential expenses and other debt payments."""
    return effective_income(case) - max(0.0, case.essential_expenses) - max(0.0, case.other_debt_payment)


def annuity_payment(balance: float, annual_rate: float, months: int) -> float:
    """Monthly payment for a level-payment annuity, excluding fees and insurance."""
    months = max(1, int(months))
    balance = max(0.0, float(balance))
    monthly_rate = max(0.0, float(annual_rate)) / 100.0 / 12.0
    if monthly_rate <= 1e-12:
        return balance / months
    factor = (1.0 + monthly_rate) ** months
    return balance * monthly_rate * factor / (factor - 1.0)


def _stress_results(case: BorrowerCase, payment: float, budget_share: float) -> tuple[tuple[str, bool], ...]:
    starting_income = effective_income(case)
    results: list[tuple[str, bool]] = []
    for name, income_shock, expense_shock in STRESS_SCENARIOS:
        stressed_income = starting_income * (1.0 - income_shock)
        stressed_expenses = max(0.0, case.essential_expenses) * (1.0 + expense_shock)
        stressed_cash = stressed_income - stressed_expenses - max(0.0, case.other_debt_payment)
        allowed_payment = max(0.0, stressed_cash) * budget_share
        results.append((name, stressed_cash >= 0.0 and payment <= allowed_payment + 0.01))
    return tuple(results)


def _make_plan(
    case: BorrowerCase,
    *,
    key: str,
    title: str,
    description: str,
    payment: float,
    term_months: int,
    grace_months: int,
    total_repayment: float,
    budget_share: float,
    final_balance_payment: float = 0.0,
    peak_monthly_payment: float | None = None,
) -> Plan:
    cash = free_cash(case)
    budget_limit = max(0.0, cash) * budget_share
    peak_payment = (
        payment + final_balance_payment
        if peak_monthly_payment is None
        else peak_monthly_payment
    )
    share = peak_payment / cash if cash > 0 else None
    stress_results = _stress_results(case, peak_payment, budget_share)
    return Plan(
        key=key,
        title=title,
        description=description,
        monthly_payment=payment,
        peak_monthly_payment=peak_payment,
        grace_months=grace_months,
        term_months=term_months,
        months_with_payment=max(0, term_months - grace_months),
        total_repayment=total_repayment,
        interest_cost=max(0.0, total_repayment - max(0.0, case.balance)),
        final_balance_payment=final_balance_payment,
        monthly_cash_after_payment=cash - payment,
        budget_limit=budget_limit,
        payment_share_of_free_cash=share,
        is_feasible=(cash >= 0.0 and peak_payment <= budget_limit + 0.01),
        feasibility_gap=max(0.0, peak_payment - budget_limit),
        stress_passes=sum(passed for _, passed in stress_results),
        stress_total=len(stress_results),
        stress_results=stress_results,
    )


def _current_plan(case: BorrowerCase, budget_share: float) -> Plan:
    """Keep the entered payment; show any residual balance as a final balloon."""
    balance = max(0.0, case.balance)
    monthly_rate = max(0.0, case.annual_rate) / 100.0 / 12.0
    payment = max(0.0, case.current_payment)
    total_paid = 0.0
    peak_payment = 0.0
    term_months = max(1, int(case.remaining_months))
    for month in range(1, term_months + 1):
        due = balance * (1.0 + monthly_rate)
        paid = min(payment, due)
        total_paid += paid
        balance = max(0.0, due - paid)
        peak_payment = max(peak_payment, paid)
        if month == term_months:
            final_payment = paid + balance
            peak_payment = max(peak_payment, final_payment)
    total_repayment = total_paid + balance
    return _make_plan(
        case,
        key="current",
        title="Текущий график",
        description="Введённый платёж; возможный остаток долга показан как платёж в конце срока.",
        payment=payment,
        term_months=term_months,
        grace_months=0,
        total_repayment=total_repayment,
        final_balance_payment=balance,
        budget_share=budget_share,
        peak_monthly_payment=peak_payment,
    )


def _mark_pareto_frontier(plans: list[Plan]) -> list[Plan]:
    feasible = [plan for plan in plans if plan.is_feasible]
    marked: list[Plan] = []
    for plan in plans:
        dominated = any(
            other.key != plan.key
            and other.peak_monthly_payment <= plan.peak_monthly_payment + 0.01
            and other.total_repayment <= plan.total_repayment + 0.01
            and other.stress_passes >= plan.stress_passes
            and (
                other.peak_monthly_payment < plan.peak_monthly_payment - 0.01
                or other.total_repayment < plan.total_repayment - 0.01
                or other.stress_passes > plan.stress_passes
                or other.key < plan.key
            )
            for other in feasible
        )
        marked.append(replace(plan, is_pareto_optimal=plan.is_feasible and not dominated))
    return marked


def make_plans(
    case: BorrowerCase,
    *,
    budget_share: float = 0.80,
    max_extension_months: int = 48,
    max_grace_months: int = 3,
) -> list[Plan]:
    """Enumerate every monthly term and grace-period pair in the search bounds."""
    if case.balance <= 0:
        raise ValueError("Остаток долга должен быть больше нуля.")
    if case.remaining_months < 1:
        raise ValueError("Оставшийся срок должен быть не меньше одного месяца.")

    budget_share = max(0.0, min(1.0, float(budget_share)))
    remaining = int(case.remaining_months)
    extension = max(0, int(max_extension_months))
    max_term = min(180, remaining + extension)
    max_grace = max(0, min(6, int(max_grace_months)))
    monthly_rate = max(0.0, float(case.annual_rate)) / 100.0 / 12.0

    plans = [_current_plan(case, budget_share)]
    for term in range(remaining, max_term + 1):
        for grace in range(0, min(max_grace, term - 1) + 1):
            if term == remaining and grace == 0:
                continue
            payment_months = term - grace
            balance_after_grace = max(0.0, case.balance) * ((1.0 + monthly_rate) ** grace)
            payment = annuity_payment(balance_after_grace, case.annual_rate, payment_months)
            total_repayment = payment * payment_months
            extension_label = f"+{term - remaining} мес. к сроку" if term > remaining else "срок без продления"
            pause_label = f"пауза {grace} мес." if grace else "без паузы"
            plans.append(
                _make_plan(
                    case,
                    key=f"term-{term}-grace-{grace}",
                    title=f"Срок {term} мес. · {pause_label}",
                    description=f"{extension_label}; равный платёж после периода отсрочки.",
                    payment=payment,
                    term_months=term,
                    grace_months=grace,
                    total_repayment=total_repayment,
                    budget_share=budget_share,
                )
            )
    return _mark_pareto_frontier(plans)


def _balanced_distance(plan: Plan, candidates: list[Plan]) -> float:
    min_payment = min(item.peak_monthly_payment for item in candidates)
    max_payment = max(item.peak_monthly_payment for item in candidates)
    min_total = min(item.total_repayment for item in candidates)
    max_total = max(item.total_repayment for item in candidates)
    payment_position = (
        (plan.peak_monthly_payment - min_payment) / (max_payment - min_payment)
        if max_payment > min_payment
        else 0.0
    )
    total_position = (
        (plan.total_repayment - min_total) / (max_total - min_total)
        if max_total > min_total
        else 0.0
    )
    return (payment_position + total_position) / 2.0


def sort_plans(plans: list[Plan], objective: str = "balanced") -> list[Plan]:
    feasible = [plan for plan in plans if plan.is_feasible]
    infeasible = [plan for plan in plans if not plan.is_feasible]
    if objective == "affordable":
        feasible.sort(key=lambda plan: (plan.peak_monthly_payment, plan.total_repayment, plan.term_months, plan.grace_months))
        infeasible.sort(key=lambda plan: (plan.feasibility_gap, plan.peak_monthly_payment, plan.total_repayment))
    elif objective == "cost":
        feasible.sort(key=lambda plan: (plan.total_repayment, plan.peak_monthly_payment, plan.term_months, plan.grace_months))
        infeasible.sort(key=lambda plan: (plan.feasibility_gap, plan.total_repayment, plan.monthly_payment))
    elif objective == "robust":
        balanced_scores = {plan.key: _balanced_distance(plan, feasible) for plan in feasible}
        feasible.sort(
            key=lambda plan: (
                -plan.stress_passes,
                balanced_scores[plan.key],
                plan.term_months,
                plan.grace_months,
            )
        )
        infeasible.sort(key=lambda plan: (-plan.stress_passes, plan.feasibility_gap, plan.peak_monthly_payment))
    else:
        feasible_scores = {plan.key: _balanced_distance(plan, feasible) for plan in feasible}
        infeasible_scores = {plan.key: _balanced_distance(plan, infeasible) for plan in infeasible}
        feasible.sort(key=lambda plan: (feasible_scores[plan.key], plan.term_months, plan.grace_months))
        infeasible.sort(key=lambda plan: (plan.feasibility_gap, infeasible_scores[plan.key], plan.term_months))
    return feasible + infeasible


def choose_plan(plans: list[Plan], objective: str = "balanced") -> Plan | None:
    """Select only among budget-feasible plans; return None if none qualify."""
    return next((plan for plan in sort_plans(plans, objective) if plan.is_feasible), None)


def objective_explanation(objective: str) -> str:
    if objective == "affordable":
        return "Среди графиков, укладывающихся в бюджетный лимит, выбран вариант с наименьшим максимальным платежом за месяц."
    if objective == "cost":
        return "Среди графиков, укладывающихся в бюджетный лимит, выбран вариант с наименьшей общей суммой выплат."
    if objective == "robust":
        return "Сначала выбран график, который выдерживает больше заданных стресс-сценариев; при равенстве сравниваются максимальный месячный платёж и общая сумма выплат."
    return "Сбалансированный вариант минимизирует среднее относительное положение по двум целям: максимальному платежу за месяц и общей сумме выплат."


def recommendation_reason(case: BorrowerCase, plan: Plan, budget_share: float) -> list[str]:
    reasons = [
        f"Максимальный платёж {plan.peak_monthly_payment:,.0f} ₸ укладывается в лимит {budget_share:.0%} свободного бюджета ({plan.budget_limit:,.0f} ₸).".replace(",", " "),
        f"Сумма выплат за оставшийся срок — {plan.total_repayment:,.0f} ₸; расчётная стоимость финансирования — {plan.interest_cost:,.0f} ₸.".replace(",", " "),
        f"График остаётся в бюджетном лимите в {plan.stress_passes} из {plan.stress_total} заданных стресс-сценариев.",
    ]
    if case.income_drop > 0:
        reasons.append(f"Для бюджета учтено снижение обычного дохода на {case.income_drop:.0%}.")
    elif plan.grace_months:
        reasons.append(f"Первые {plan.grace_months} мес. платежи приостановлены; проценты за паузу капитализируются.")
    else:
        reasons.append(f"Срок составляет {plan.term_months} мес., включая возможную паузу.")
    return reasons[:4]


def monthly_schedule(case: BorrowerCase, plan: Plan, horizon: int = 12) -> list[float]:
    """Return actual monthly cash payments, including a final current-plan balloon."""
    horizon = max(0, int(horizon))
    schedule: list[float] = []
    if plan.key != "current":
        for month in range(1, horizon + 1):
            if month > plan.term_months:
                schedule.append(0.0)
            elif month <= plan.grace_months:
                schedule.append(0.0)
            else:
                schedule.append(plan.monthly_payment)
        return schedule

    balance = max(0.0, case.balance)
    monthly_rate = max(0.0, case.annual_rate) / 100.0 / 12.0
    for month in range(1, horizon + 1):
        if month > plan.term_months:
            schedule.append(0.0)
            continue
        due = balance * (1.0 + monthly_rate)
        payment = min(max(0.0, case.current_payment), due)
        balance = max(0.0, due - payment)
        if month == plan.term_months:
            payment += balance
            balance = 0.0
        schedule.append(payment)
    return schedule


def amortization_schedule(case: BorrowerCase, plan: Plan) -> list[AmortizationRow]:
    """Return the full balance, interest, and payment breakdown for a plan."""
    balance = max(0.0, case.balance)
    monthly_rate = max(0.0, case.annual_rate) / 100.0 / 12.0
    rows: list[AmortizationRow] = []

    for month in range(1, max(0, int(plan.term_months)) + 1):
        opening_balance = balance
        accrued_interest = opening_balance * monthly_rate
        amount_due = opening_balance + accrued_interest

        if plan.key == "current":
            payment = min(max(0.0, case.current_payment), amount_due)
            if month == plan.term_months:
                payment = amount_due
        elif month <= plan.grace_months:
            payment = 0.0
        else:
            payment = min(max(0.0, plan.monthly_payment), amount_due)

        interest_paid = min(payment, accrued_interest)
        capitalized_interest = max(0.0, accrued_interest - interest_paid)
        principal_repaid = min(
            max(0.0, payment - interest_paid), opening_balance
        )
        closing_balance = max(
            0.0, opening_balance + capitalized_interest - principal_repaid
        )
        rows.append(
            AmortizationRow(
                month=month,
                opening_balance=opening_balance,
                payment=payment,
                accrued_interest=accrued_interest,
                interest_paid=interest_paid,
                principal_repaid=principal_repaid,
                capitalized_interest=capitalized_interest,
                closing_balance=closing_balance,
            )
        )
        balance = closing_balance

    return rows


def portfolio_summary(
    objective: str = "balanced",
    *,
    budget_share: float = 0.80,
    max_extension_months: int = 48,
    max_grace_months: int = 3,
) -> list[PortfolioItem]:
    summary: list[PortfolioItem] = []
    for case in CASES:
        plans = make_plans(
            case,
            budget_share=budget_share,
            max_extension_months=max_extension_months,
            max_grace_months=max_grace_months,
        )
        recommendation = choose_plan(plans, objective)
        summary.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "region": case.region,
                "segment": case.segment,
                "balance": case.balance,
                "current_payment": case.current_payment,
                "recommended_plan": recommendation.title if recommendation else "Нет доступного графика",
                "recommended_payment": recommendation.monthly_payment if recommendation else None,
                "recommended_peak_payment": recommendation.peak_monthly_payment if recommendation else None,
                "recommended_total_repayment": recommendation.total_repayment if recommendation else None,
                "stress_passes": recommendation.stress_passes if recommendation else None,
                "stress_total": len(STRESS_SCENARIOS),
                "has_feasible_plan": recommendation is not None,
            }
        )
    return summary
