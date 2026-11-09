"""Transparent repayment-schedule search for the QaryzFlow demonstration app.

The engine uses invented borrower profiles and deterministic arithmetic. It does
not estimate default probabilities or make decisions about real customers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum
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
    income_recovery_months: int = 0
    income_recovery_wait_months: int = 1
    demo_note: str = ""


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
    budget_check_month: int
    budget_check_payment: float
    budget_margin: float
    is_pareto_optimal: bool = False


@dataclass(frozen=True)
class BudgetMonth:
    month: int
    income: float
    free_cash: float
    payment_limit: float


@dataclass(frozen=True)
class StressCheck:
    name: str
    passed: bool
    shortfall_month: int | None = None
    shortfall: float = 0.0


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
    BorrowerCase(
        "Q-1042", "Q-1042 · Высокая нагрузка", "Наёмный сотрудник", "Алматы", 420_000, 0.22, 1_850_000, 22.5, 131_000, 95_000, 180_000, 26, 2, 130_000,
        demo_note="Свободный бюджет положительный, но лимит слишком мал для нового платежа. Попробуйте изменить срок поиска или долю бюджета.",
    ),
    BorrowerCase(
        "Q-1088", "Q-1088 · Небольшое падение дохода", "Микропредприниматель", "Шымкент", 310_000, 0.06, 950_000, 26.5, 65_000, 48_000, 150_000, 18, 1, 300_000,
        demo_note="Доход немного снизился. Сравните, сколько стоит уменьшение ежемесячного платежа за счёт продления срока.",
    ),
    BorrowerCase(
        "Q-1121", "Q-1121 · Крупный остаток долга", "Наёмный сотрудник", "Астана", 680_000, 0.10, 3_200_000, 21.0, 198_000, 112_000, 250_000, 30, 3, 500_000,
        demo_note="Нагрузка складывается из большого остатка и платежей по другим обязательствам. Сопоставьте цель по стоимости и по устойчивости.",
    ),
    BorrowerCase(
        "Q-1176", "Q-1176 · Сценарии восстановления", "Самозанятый", "Караганда", 410_000, 0.33, 1_200_000, 23.9, 104_000, 62_000, 145_000, 19, 4, 25_000,
        demo_note="Основной пример для показа. Установите три шага восстановления дохода и сравните рост со второго и с четвёртого месяца.",
    ),
    BorrowerCase(
        "Q-1204", "Q-1204 · Другие обязательства", "Наёмный сотрудник", "Актобе", 540_000, 0.00, 2_400_000, 19.9, 118_000, 210_000, 220_000, 27, 1, 420_000,
        demo_note="Доход стабилен, а бюджет сжимают выплаты по другим долгам. Посмотрите, как они меняют доступный платёж.",
    ),
    BorrowerCase(
        "Q-1249", "Q-1249 · Стоимость или запас", "Микропредприниматель", "Тараз", 390_000, 0.13, 1_500_000, 24.0, 95_000, 80_000, 175_000, 21, 2, 160_000,
        demo_note="Сравните дешёвый график с графиком, который оставляет больший запас при росте расходов.",
    ),
    BorrowerCase(
        "Q-1301", "Q-1301 · Пауза при позднем восстановлении", "Самозанятый", "Усть-Каменогорск", 430_000, 0.35, 900_000, 18.0, 76_000, 75_000, 170_000, 14, 1, 65_000,
        income_recovery_months=3, income_recovery_wait_months=3,
        demo_note="Доход остаётся сниженным три месяца, затем постепенно растёт. Проверьте, какой срок и пауза подходят при такой траектории.",
    ),
    BorrowerCase(
        "Q-1320", "Q-1320 · Крупный последний платёж", "Наёмный сотрудник", "Павлодар", 255_000, 0.00, 550_000, 16.5, 22_000, 38_000, 145_000, 12, 0, 85_000,
        demo_note="Текущий регулярный платёж доступен, но в конце остаётся крупный остаток. Откройте весь календарь и сравните с пересчётом на те же 12 месяцев.",
    ),
    BorrowerCase(
        "Q-1334", "Q-1334 · Дефицит даже в паузу", "Наёмный сотрудник", "Костанай", 300_000, 0.30, 720_000, 20.0, 48_000, 52_000, 188_000, 18, 2, 20_000,
        demo_note="Доход уже ниже обязательных расходов и других платежей. Пауза по этому кредиту сама по себе не закрывает нехватку средств.",
    ),
    BorrowerCase(
        "Q-1370", "Q-1370 · Нулевая ставка и запас", "Наёмный сотрудник", "Петропавловск", 320_000, 0.00, 360_000, 0.0, 30_000, 20_000, 160_000, 12, 0, 240_000,
        demo_note="Текущий график проходит бюджет и все стресс-проверки. При нулевой ставке продление меняет срок и платёж, сохраняя сумму выплат.",
    ),
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


def effective_income(case: BorrowerCase, month: int = 1) -> float:
    """Start at the reduced income and recover linearly when requested."""
    drop = max(0.0, min(0.80, float(case.income_drop)))
    recovery = max(0, int(case.income_recovery_months))
    if recovery:
        wait = max(1, int(case.income_recovery_wait_months))
        drop *= 1.0 - min(max(0, month - wait), recovery) / recovery
    return max(0.0, float(case.income)) * (1.0 - drop)


def free_cash(case: BorrowerCase) -> float:
    """First month's income left after essential expenses and other debts."""
    return effective_income(case) - max(0.0, case.essential_expenses) - max(0.0, case.other_debt_payment)


def monthly_budget(
    case: BorrowerCase,
    budget_share: float,
    horizon: int,
    *,
    income_shock: float = 0.0,
    expense_shock: float = 0.0,
) -> list[BudgetMonth]:
    """Compute the payment limit in each month of the assumed income path."""
    share = max(0.0, min(1.0, float(budget_share)))
    expenses = max(0.0, case.essential_expenses) * (1.0 + expense_shock)
    other_debt = max(0.0, case.other_debt_payment)
    rows: list[BudgetMonth] = []
    for month in range(1, horizon + 1):
        income = effective_income(case, month) * (1.0 - income_shock)
        cash = income - expenses - other_debt
        rows.append(BudgetMonth(month, income, cash, max(0.0, cash) * share))
    return rows


def annuity_payment(balance: float, annual_rate: float, months: int) -> float:
    """Monthly payment for a level-payment annuity, excluding fees and insurance."""
    months = max(1, int(months))
    balance = max(0.0, float(balance))
    monthly_rate = max(0.0, float(annual_rate)) / 100.0 / 12.0
    if monthly_rate <= 1e-12:
        return balance / months
    factor = (1.0 + monthly_rate) ** months
    return balance * monthly_rate * factor / (factor - 1.0)


def payment_after_grace(balance: float, annual_rate: float, term: int, grace: int) -> float:
    """Level payment on a term that includes the initial payment pause."""
    monthly_rate = max(0.0, float(annual_rate)) / 100.0 / 12.0
    balance_after_grace = max(0.0, balance) * (1.0 + monthly_rate) ** grace
    return annuity_payment(balance_after_grace, annual_rate, term - grace)


def _budget_margin(month: BudgetMonth, payment: float) -> float:
    return month.payment_limit - payment + min(0.0, month.free_cash)


def payment_within_limit(payment: float, limit: float) -> bool:
    """Check the same one-cent payment tolerance in the engine and the calendar."""
    return payment <= limit + 0.01


def _fits_month(month: BudgetMonth, payment: float) -> bool:
    return month.free_cash >= 0.0 and payment_within_limit(payment, month.payment_limit)


def _passes_budget(budget: list[BudgetMonth], payments: list[float]) -> bool:
    return all(_fits_month(month, payment) for month, payment in zip(budget, payments))


def _make_plan(
    case: BorrowerCase,
    *,
    key: str,
    title: str,
    description: str,
    payment: float,
    grace_months: int,
    payments: list[float],
    budget: list[BudgetMonth],
    stress_budgets: tuple[tuple[str, list[BudgetMonth]], ...],
    final_balance_payment: float = 0.0,
) -> Plan:
    tightest_month, tightest_payment = min(
        zip(budget, payments), key=lambda item: _budget_margin(*item)
    )
    margin = _budget_margin(tightest_month, tightest_payment)
    paying_months = [(month, paid) for month, paid in zip(budget, payments) if paid > 0.0]
    payment_share = None
    if all(month.free_cash > 0.0 for month, _ in paying_months):
        payment_share = max((paid / month.free_cash for month, paid in paying_months), default=None)
    total_repayment = fsum(payments)
    stress_results = tuple(
        (name, _passes_budget(stressed_budget, payments))
        for name, stressed_budget in stress_budgets
    )
    return Plan(
        key=key,
        title=title,
        description=description,
        monthly_payment=payment,
        peak_monthly_payment=max(payments),
        grace_months=grace_months,
        term_months=len(payments),
        months_with_payment=len(paying_months),
        total_repayment=total_repayment,
        interest_cost=max(0.0, total_repayment - max(0.0, case.balance)),
        final_balance_payment=final_balance_payment,
        monthly_cash_after_payment=min(month.free_cash - paid for month, paid in zip(budget, payments)),
        budget_limit=tightest_month.payment_limit,
        payment_share_of_free_cash=payment_share,
        is_feasible=_passes_budget(budget, payments),
        feasibility_gap=max(0.0, -margin),
        stress_passes=sum(passed for _, passed in stress_results),
        stress_total=len(stress_results),
        stress_results=stress_results,
        budget_check_month=tightest_month.month,
        budget_check_payment=tightest_payment,
        budget_margin=margin,
    )


def _current_payments(case: BorrowerCase, term_months: int) -> tuple[list[float], float]:
    """Keep the entered payment; show any residual balance as a final balloon."""
    balance = max(0.0, case.balance)
    monthly_rate = max(0.0, case.annual_rate) / 100.0 / 12.0
    payment = max(0.0, case.current_payment)
    payments: list[float] = []
    for _ in range(term_months):
        due = balance * (1.0 + monthly_rate)
        paid = min(payment, due)
        payments.append(paid)
        balance = max(0.0, due - paid)
    if payments:
        payments[-1] += balance
    return payments, balance


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

    remaining = int(case.remaining_months)
    extension = max(0, int(max_extension_months))
    max_term = min(180, remaining + extension)
    max_grace = max(0, min(6, int(max_grace_months)))
    horizon = max(remaining, max_term)
    budget = monthly_budget(case, budget_share, horizon)
    stress_budgets = tuple(
        (
            name,
            monthly_budget(case, budget_share, horizon, income_shock=income_shock, expense_shock=expense_shock),
        )
        for name, income_shock, expense_shock in STRESS_SCENARIOS
    )
    current_payments, final_balance = _current_payments(case, remaining)
    plans = [
        _make_plan(
            case,
            key="current",
            title="Текущий график",
            description="Введённый платёж; возможный остаток долга показан как платёж в конце срока.",
            payment=max(0.0, case.current_payment),
            grace_months=0,
            payments=current_payments,
            budget=budget,
            stress_budgets=stress_budgets,
            final_balance_payment=final_balance,
        )
    ]
    for term in range(remaining, max_term + 1):
        for grace in range(0, min(max_grace, term - 1) + 1):
            payment_months = term - grace
            payment = payment_after_grace(case.balance, case.annual_rate, term, grace)
            extension_label = f"+{term - remaining} мес. к сроку" if term > remaining else "срок без продления"
            pause_label = f"пауза {grace} мес." if grace else "без паузы"
            plans.append(
                _make_plan(
                    case,
                    key=f"term-{term}-grace-{grace}",
                    title=f"Срок {term} мес. · {pause_label}",
                    description=f"{extension_label}; равный платёж после периода отсрочки.",
                    payment=payment,
                    grace_months=grace,
                    payments=[0.0] * grace + [payment] * payment_months,
                    budget=budget,
                    stress_budgets=stress_budgets,
                )
            )
    return _mark_pareto_frontier(plans)


def _balanced_scores(plans: list[Plan], reference: list[Plan]) -> dict[str, float]:
    if not reference:
        return {}
    min_payment = min(plan.peak_monthly_payment for plan in reference)
    min_total = min(plan.total_repayment for plan in reference)
    payment_range = max(plan.peak_monthly_payment for plan in reference) - min_payment
    total_range = max(plan.total_repayment for plan in reference) - min_total
    return {
        plan.key: (
            ((plan.peak_monthly_payment - min_payment) / payment_range if payment_range else 0.0)
            + ((plan.total_repayment - min_total) / total_range if total_range else 0.0)
        ) / 2.0
        for plan in plans
    }


def sort_plans(plans: list[Plan], objective: str = "balanced") -> list[Plan]:
    feasible = [plan for plan in plans if plan.is_feasible]
    infeasible = [plan for plan in plans if not plan.is_feasible]
    if objective == "affordable":
        feasible.sort(key=lambda plan: (plan.peak_monthly_payment, plan.total_repayment, plan.term_months, plan.grace_months))
        infeasible.sort(key=lambda plan: (plan.feasibility_gap, plan.peak_monthly_payment, plan.total_repayment))
    elif objective == "cost":
        feasible.sort(
            key=lambda plan: (
                round(plan.total_repayment, 2),
                plan.key != "current",
                round(plan.peak_monthly_payment, 2),
                plan.term_months,
                plan.grace_months,
            )
        )
        infeasible.sort(key=lambda plan: (plan.feasibility_gap, plan.total_repayment, plan.monthly_payment))
    else:
        frontier = [plan for plan in feasible if plan.is_pareto_optimal]
        scores = _balanced_scores(feasible, frontier or feasible)
        robust = objective == "robust"
        feasible.sort(
            key=lambda plan: (
                -plan.stress_passes if robust else 0,
                not plan.is_pareto_optimal,
                scores[plan.key],
                plan.term_months,
                plan.grace_months,
            )
        )
        if robust:
            infeasible.sort(key=lambda plan: (-plan.stress_passes, plan.feasibility_gap, plan.peak_monthly_payment))
        else:
            infeasible_scores = _balanced_scores(infeasible, infeasible)
            infeasible.sort(key=lambda plan: (plan.feasibility_gap, infeasible_scores[plan.key], plan.term_months))
    return feasible + infeasible


def choose_plan(plans: list[Plan], objective: str = "balanced") -> Plan | None:
    """Select only among budget-feasible plans; return None if none qualify."""
    return next((plan for plan in sort_plans(plans, objective) if plan.is_feasible), None)


def objective_explanation(objective: str) -> str:
    if objective == "affordable":
        return "Среди графиков, укладывающихся в бюджетный лимит, выбран вариант с наименьшим максимальным платежом за месяц."
    if objective == "cost":
        return "Суммы выплат допустимых графиков сравниваются до 0,01 ₸. При равной минимальной сумме предпочтение получает текущий график."
    if objective == "robust":
        return "Приоритет у графика, который выдерживает больше стресс-сценариев. При равенстве используется баланс платежа и общей суммы среди лучших компромиссов."
    return "Среди лучших компромиссов максимальный платёж и общая сумма выплат сравниваются с равными весами. Варианты, которые уступают другому по этим показателям и стресс-проверкам, не меняют шкалу."


def recommendation_reason(case: BorrowerCase, plan: Plan, budget_share: float) -> list[str]:
    reasons = [
        (
            f"Платёж проходит лимит {budget_share:.0%} свободного бюджета в каждом месяце. "
            f"Наименьший запас — в месяце {plan.budget_check_month}: платёж {plan.budget_check_payment:,.0f} ₸ "
            f"при лимите {plan.budget_limit:,.0f} ₸."
        ).replace(",", " "),
        f"Сумма выплат за оставшийся срок — {plan.total_repayment:,.0f} ₸; расчётная стоимость финансирования — {plan.interest_cost:,.0f} ₸.".replace(",", " "),
        f"График остаётся в бюджетном лимите в {plan.stress_passes} из {plan.stress_total} заданных стресс-сценариев.",
    ]
    if plan.grace_months:
        reasons.append(f"Первые {plan.grace_months} мес. платежи приостановлены; проценты за паузу капитализируются.")
    elif case.income_drop > 0 and case.income_recovery_months > 0:
        recovery_end = max(1, case.income_recovery_wait_months) + case.income_recovery_months
        reasons.append(f"По сценарию обычный доход возвращается в месяце {recovery_end}.")
    else:
        reasons.append(f"Срок составляет {plan.term_months} мес., включая возможную паузу.")
    return reasons[:4]


def monthly_schedule(case: BorrowerCase, plan: Plan, horizon: int = 12) -> list[float]:
    """Return actual monthly cash payments, including a final current-plan balloon."""
    horizon = max(0, int(horizon))
    if plan.key == "current":
        payments, _ = _current_payments(case, plan.term_months)
        return (payments + [0.0] * max(0, horizon - len(payments)))[:horizon]
    return [
        plan.monthly_payment if plan.grace_months < month <= plan.term_months else 0.0
        for month in range(1, horizon + 1)
    ]


def stress_details(case: BorrowerCase, plan: Plan, budget_share: float) -> list[StressCheck]:
    """Report the largest shortfall and its month for each failed scenario."""
    payments = monthly_schedule(case, plan, plan.term_months)
    checks = []
    for name, income_shock, expense_shock in STRESS_SCENARIOS:
        budget = monthly_budget(
            case, budget_share, plan.term_months, income_shock=income_shock, expense_shock=expense_shock
        )
        tightest = min(
            ((month, paid) for month, paid in zip(budget, payments) if not _fits_month(month, paid)),
            key=lambda item: _budget_margin(*item),
            default=None,
        )
        if tightest is None:
            checks.append(StressCheck(name, passed=True))
        else:
            month, paid = tightest
            checks.append(StressCheck(name, passed=False, shortfall_month=month.month, shortfall=-_budget_margin(month, paid)))
    return checks


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
