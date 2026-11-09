"""Cross-check the demo against a separate decimal cash-flow calculation."""

from dataclasses import dataclass, replace
from decimal import Decimal
from statistics import median
from time import perf_counter

from qadam_engine import (
    CASES,
    STRESS_SCENARIOS,
    BorrowerCase,
    Plan,
    amortization_schedule,
    choose_plan,
    make_plans,
    monthly_budget,
    monthly_schedule,
)


@dataclass(frozen=True)
class Example:
    name: str
    case: BorrowerCase
    income_by_month: tuple[int, ...]  # The last value applies to the rest of the term.
    extension: int = 48
    grace: int = 3


CASES_BY_ID = {case.case_id: case for case in CASES}
BASE = CASES_BY_ID["Q-1176"]
EXAMPLES = (
    Example("fixed_income", BASE, (274_700,)),
    Example("early_recovery", replace(BASE, income_recovery_months=3), (274_700, 319_800, 364_900, 410_000)),
    Example(
        "delayed_recovery",
        replace(BASE, income_recovery_months=3, income_recovery_wait_months=3),
        (274_700, 274_700, 274_700, 319_800, 364_900, 410_000),
    ),
    Example("household_deficit", replace(CASES_BY_ID["Q-1042"], income_drop=0.70), (126_000,)),
    Example(
        "same_term_recalculation",
        replace(
            BASE, income=100_000, income_drop=0.0, essential_expenses=80_000, other_debt_payment=0.0,
            balance=120_000, annual_rate=0.0, current_payment=5_000, remaining_months=12,
        ),
        (100_000,), extension=0, grace=0,
    ),
    Example("final_balloon", CASES_BY_ID["Q-1320"], (255_000,)),
    Example("keep_current_at_equal_cost", CASES_BY_ID["Q-1370"], (320_000,)),
)
BUDGET_SHARE = Decimal("0.80")
CENT = Decimal("0.01")


def reference_cashflows(example: Example) -> dict[str, list[Decimal]]:
    case = example.case
    debt = Decimal(str(case.balance))
    factor = 1 + Decimal(str(case.annual_rate)) / 1200
    current_payment = Decimal(str(case.current_payment))
    opening = debt
    current = []
    for month in range(1, case.remaining_months + 1):
        due = opening * factor
        payment = due if month == case.remaining_months else min(current_payment, due)
        current.append(payment)
        opening = due - payment

    cashflows = {"current": current}
    for term in range(case.remaining_months, case.remaining_months + example.extension + 1):
        for pause in range(min(example.grace, term - 1) + 1):
            # The present value of payments in months pause+1 through term equals the original debt.
            present_value_of_unit_payments = sum(factor ** -month for month in range(pause + 1, term + 1))
            payment = debt / present_value_of_unit_payments
            cashflows[f"term-{term}-grace-{pause}"] = [Decimal(0)] * pause + [payment] * (term - pause)
    return cashflows


def reference_available(
    example: Example, horizon: int, income_shock: float = 0.0, expense_shock: float = 0.0,
) -> list[Decimal]:
    expenses = Decimal(str(example.case.essential_expenses)) * (1 + Decimal(str(expense_shock)))
    obligations = expenses + Decimal(str(example.case.other_debt_payment))
    income_multiplier = 1 - Decimal(str(income_shock))
    incomes = example.income_by_month
    return [
        income_multiplier * incomes[min(index, len(incomes) - 1)] - obligations
        for index in range(horizon)
    ]


def fits_budget(available_by_month: list[Decimal], payments: list[Decimal]) -> bool:
    return all(
        available >= 0 and payment <= BUDGET_SHARE * available + CENT
        for available, payment in zip(available_by_month, payments)
    )


def check_months(example: Example, plan: Plan, payments: list[Decimal]) -> Decimal:
    """Check the plotted payments and the exported ledger against the reference."""
    case = example.case
    ledger = amortization_schedule(case, plan)
    plotted = monthly_schedule(case, plan, len(payments))
    balance = Decimal(str(case.balance))
    rate = Decimal(str(case.annual_rate)) / 1200
    max_error = Decimal(0)

    for month, (payment, line_amount, row) in enumerate(zip(payments, plotted, ledger, strict=True), start=1):
        interest = balance * rate
        interest_paid = min(interest, payment)
        closing = max(Decimal(0), balance + interest - payment)
        fields = (
            ("plotted payment", line_amount, payment),
            ("payment", row.payment, payment),
            ("opening balance", row.opening_balance, balance),
            ("accrued interest", row.accrued_interest, interest),
            ("interest paid", row.interest_paid, interest_paid),
            ("capitalized interest", row.capitalized_interest, interest - interest_paid),
            ("principal repaid", row.principal_repaid, min(balance, payment - interest_paid)),
            ("closing balance", row.closing_balance, closing),
        )
        if row.month != month:
            raise AssertionError(f"Month index mismatch: {example.name} / {plan.key} / month {month}")
        for field, actual, expected in fields:
            error = abs(Decimal(str(actual)) - expected)
            if error > CENT:
                raise AssertionError(f"{field} mismatch: {example.name} / {plan.key} / month {month}")
            max_error = max(max_error, error)
        balance = closing
    return max_error


def solve(example: Example) -> list[Plan]:
    return make_plans(
        example.case, budget_share=float(BUDGET_SHARE),
        max_extension_months=example.extension, max_grace_months=example.grace,
    )


def main() -> None:
    total_plans = 0
    total_months = 0
    max_error = Decimal(0)
    for example in EXAMPLES:
        expected = reference_cashflows(example)
        plans = solve(example)
        if {plan.key for plan in plans} != expected.keys():
            raise AssertionError(f"Candidate sets differ for {example.name}")
        horizon = max(map(len, expected.values()))
        available = reference_available(example, horizon)
        actual_budget = monthly_budget(example.case, float(BUDGET_SHARE), horizon)
        for month, free_cash in zip(actual_budget, available, strict=True):
            limit = max(Decimal(0), free_cash) * BUDGET_SHARE
            error = abs(Decimal(str(month.payment_limit)) - limit)
            if error > CENT:
                raise AssertionError(f"Payment limit mismatch: {example.name} / month {month.month}")
            max_error = max(max_error, error)
        stressed = [
            (name, reference_available(example, horizon, income, expense))
            for name, income, expense in STRESS_SCENARIOS
        ]

        feasible = []
        for plan in plans:
            payments = expected[plan.key]
            total = sum(payments)
            peak = max(payments)
            fits = fits_budget(available, payments)
            error = max(abs(Decimal(str(plan.total_repayment)) - total), abs(Decimal(str(plan.peak_monthly_payment)) - peak))
            if error > CENT or plan.is_feasible != fits:
                raise AssertionError(f"Cash-flow or budget mismatch: {example.name} / {plan.key}")
            if plan.stress_results != tuple((name, fits_budget(cash, payments)) for name, cash in stressed):
                raise AssertionError(f"Stress check mismatch: {example.name} / {plan.key}")
            if fits:
                feasible.append((
                    total.quantize(CENT), plan.key != "current", peak.quantize(CENT),
                    plan.term_months, plan.grace_months, plan.key,
                ))
            max_error = max(max_error, error, check_months(example, plan, payments))
            total_months += len(payments)

        choice = choose_plan(plans, "cost")
        expected_key = min(feasible)[-1] if feasible else None
        if (choice.key if choice else None) != expected_key:
            raise AssertionError(f"Cost choice differs for {example.name}")
        total_plans += len(plans)
        print(f"{example.name}: {len(plans)} plans, cost choice={expected_key}")

    timings = []
    for _ in range(7):
        start = perf_counter()
        for example in EXAMPLES:
            choose_plan(solve(example), "cost")
        timings.append((perf_counter() - start) * 1_000)
    print(f"Verified {total_plans} plans and {total_months} monthly rows; max amount error {max_error:.9f} KZT")
    print(f"Median local engine time for all {len(EXAMPLES)} cases: {median(timings):.1f} ms (7 repeats)")


if __name__ == "__main__":
    main()
