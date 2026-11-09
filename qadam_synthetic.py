"""Fictional loan offers and cash-flow outcomes for the ML demonstration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qadam_engine import BorrowerCase, Plan, annuity_payment, effective_income, free_cash, payment_after_grace


DEFAULT_ROWS = 8_000
DEFAULT_SEED = 20261012
MAX_SEED = 2_147_483_647
OBSERVED_PAYMENTS = 6
MAX_GRACE = 6
SCENARIOS = (
    "Стабильный доход",
    "Снижение без восстановления",
    "Раннее восстановление",
    "Отложенное восстановление",
)

# Each column defines the same input for synthetic training and for the case screen.
FEATURE_LABELS = {
    "payment_share_of_free_cash": "Наибольшая доля платежа в свободном бюджете",
    "balance_to_monthly_income": "Остаток долга / месячный доход",
    "other_debt_share": "Платежи по другим долгам / доход",
    "annual_rate_pct": "Номинальная годовая ставка, %",
    "term_months": "Срок графика, мес.",
    "income_drop_share": "Снижение дохода, доля",
    "overdue_months": "Текущая просрочка, мес.",
    "reserve_months": "Резерв, мес. расходов",
    "grace_months": "Пауза, мес.",
    "recovery_wait_months": "До первого роста дохода, мес.",
    "recovery_months": "Восстановление дохода, мес.",
}
FEATURE_NAMES = tuple(FEATURE_LABELS)


@dataclass(frozen=True)
class SyntheticLoan:
    case: BorrowerCase
    scenario: str
    term_months: int
    grace_months: int
    payment: float
    payment_share: float


@dataclass(frozen=True)
class SyntheticDataset:
    loans: tuple[SyntheticLoan, ...]
    features: np.ndarray
    labels: np.ndarray
    missed_month: np.ndarray
    missed_amount: np.ndarray


def _feature_row(case: BorrowerCase, term: int, grace: int, payment_share: float) -> tuple[float, ...]:
    income = max(effective_income(case), 1.0)
    recovery = case.income_recovery_months if case.income_drop > 0 else 0
    return (
        max(0.0, min(1.25, payment_share)),
        max(0.0, min(30.0, case.balance / income)),
        max(0.0, min(0.70, case.other_debt_payment / income)),
        max(0.0, min(45.0, case.annual_rate)),
        max(1, min(180, term)),
        max(0.0, min(0.70, case.income_drop)),
        max(0, min(6, case.overdue_months)),
        max(0.0, min(8.0, case.savings / max(case.essential_expenses, 1.0))),
        max(0, min(MAX_GRACE, grace)),
        max(1, min(12, case.income_recovery_wait_months)) if recovery > 0 else 0,
        max(0, min(12, recovery)),
    )


def plan_features(case: BorrowerCase, plan: Plan) -> np.ndarray:
    """Build the same decision-time inputs that are available to the generator."""
    share = plan.payment_share_of_free_cash
    if share is None:
        share = plan.peak_monthly_payment / max(free_cash(case), 1.0)
    return np.asarray([_feature_row(case, plan.term_months, plan.grace_months, share)], dtype=float)


def _draw_loan(rng: np.random.Generator, index: int, later: bool) -> SyntheticLoan | None:
    pressure = rng.beta(2.0, 2.2)
    scenario = int(rng.choice(4, p=(0.19, 0.31, 0.21, 0.29) if later else (0.29, 0.24, 0.27, 0.20)))
    income = round(min(1_200_000, max(140_000, rng.lognormal(np.log(380_000), 0.43))) / 1_000) * 1_000
    drop = rng.uniform(0.0, 0.06) if scenario == 0 else rng.uniform(0.13, 0.39) + 0.025 * later
    recovery = int(rng.integers(2, 9)) if scenario >= 2 else 0
    wait = int(rng.integers(3, 7)) if scenario == 3 else 1
    expenses = round(income * max(0.26, min(0.62, 0.29 + 0.21 * pressure + rng.normal(0, 0.035))))
    other_debt = round(income * max(0.0, min(0.33, 0.015 + 0.21 * pressure + rng.normal(0, 0.035))))
    savings = round(expenses * min(5.0, rng.gamma(1.4, 0.65) * (1.0 - 0.53 * pressure) * (0.61 if later else 1.0)))
    if rng.random() < (0.32 if later else 0.22):
        savings = 0

    term = int(rng.integers(6, 121))
    grace = int(rng.integers(0, min(MAX_GRACE, term - 1) + 1)) if rng.random() < (0.70 if recovery else 0.23) else 0
    remaining = min(84, term - int(rng.integers(0, min(term - 3, 36) + 1)))
    balance = round(income * min(18.0, max(0.8, rng.lognormal(1.30 + 0.63 * pressure, 0.60))) / 1_000) * 1_000
    rate = 0.0 if rng.random() < 0.03 else round(max(5.0, min(45.0, 12 + 22 * pressure + rng.normal(0, 3.5) + 3.0 * later)), 1)
    overdue = int(rng.integers(1, 7)) if rng.random() < (0.12 + 0.45 * pressure + 0.08 * later) else 0
    case = BorrowerCase(
        f"SIM-{index + 1:05d}", "", "", "", income, drop, balance, rate,
        annuity_payment(balance, rate, remaining), other_debt, expenses, remaining, overdue, savings,
        income_recovery_months=recovery, income_recovery_wait_months=wait,
    )
    payment = payment_after_grace(balance, rate, term, grace)
    # The forecast is nondecreasing, so the first paying month has the highest load.
    first_payment_cash = effective_income(case, grace + 1) - expenses - other_debt
    if free_cash(case) < 0 or payment > first_payment_cash:
        return None
    return SyntheticLoan(case, SCENARIOS[scenario], term, grace, payment, payment / first_payment_cash)


def first_missed_payment(
    payments: np.ndarray, available_cash: np.ndarray, savings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the month and amount of the first missed instalment in each row."""
    reserve = savings.copy()
    first_month = np.zeros(len(payments), dtype=int)
    first_amount = np.zeros(len(payments), dtype=float)
    for index, (payment, cash) in enumerate(zip(payments.T, available_cash.T, strict=True), start=1):
        due_shortfall = np.maximum(0.0, payment - np.maximum(0.0, cash + reserve))
        first_failure = (first_month == 0) & (due_shortfall > 0.01)
        first_month[first_failure] = index
        first_amount[first_failure] = due_shortfall[first_failure]
        # A pause can still consume savings through other household expenses.
        reserve = np.maximum(0.0, reserve + np.minimum(0.0, cash - payment))
    return first_month, first_amount


def _simulate_payments(
    loans: list[SyntheticLoan], later: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    months = np.arange(1, MAX_GRACE + OBSERVED_PAYMENTS + 1)
    shape = (len(loans), len(months))
    delays = rng.binomial(4, np.where(later, 0.48, 0.19))
    realized_cases = (
        replace(loan.case, income_recovery_wait_months=loan.case.income_recovery_wait_months + int(delay))
        for loan, delay in zip(loans, delays, strict=True)
    )
    planned_with_delay = np.asarray([
        [effective_income(case, int(month)) for month in months] for case in realized_cases
    ])
    income_shock = (rng.random(len(loans)) < np.where(later, 0.49, 0.27)) * rng.uniform(0.13, 0.36, len(loans))
    actual_income = planned_with_delay * np.clip(1.0 - income_shock[:, None] + rng.normal(0, 0.10, shape), 0.30, 1.25)
    extra_expense = (rng.random(shape) < np.where(later[:, None], 0.25, 0.12)) * rng.uniform(0.15, 0.60, shape)
    expense_multiplier = np.clip(1.0 + np.where(later[:, None], 0.08, 0.02) + rng.normal(0, 0.06, shape) + extra_expense, 0.80, None)
    expenses = np.array([loan.case.essential_expenses for loan in loans])[:, None] * expense_multiplier
    other_debt = np.array([loan.case.other_debt_payment for loan in loans])[:, None]
    pause = np.array([loan.grace_months for loan in loans])[:, None]
    term = np.array([loan.term_months for loan in loans])[:, None]
    payment = np.array([loan.payment for loan in loans])[:, None]
    payments = np.where((months > pause) & (months <= np.minimum(term, pause + OBSERVED_PAYMENTS)), payment, 0.0)
    return first_missed_payment(payments, actual_income - expenses - other_debt, np.array([loan.case.savings for loan in loans]))


def simulate_dataset(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> SyntheticDataset:
    """Generate one affordable annuity per profile, then observe its cash flow."""
    if not isinstance(rows, (int, np.integer)) or rows < 100:
        raise ValueError("Для генератора нужно целое число от 100 искусственных строк.")
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > MAX_SEED:
        raise ValueError(f"Seed должен быть целым числом от 0 до {MAX_SEED}.")

    rng = np.random.default_rng(seed)
    later = np.arange(rows) >= int(rows * 0.80)
    loans: list[SyntheticLoan] = []
    for index, is_later in enumerate(later):
        for _ in range(1_000):
            loan = _draw_loan(rng, index, bool(is_later))
            if loan is not None:
                loans.append(loan)
                break
        else:
            raise RuntimeError("Не удалось получить допустимый график для искусственного примера.")

    features = np.asarray([
        _feature_row(loan.case, loan.term_months, loan.grace_months, loan.payment_share) for loan in loans
    ], dtype=float)
    missed_month, missed_amount = _simulate_payments(loans, later, rng)
    return SyntheticDataset(tuple(loans), features, (missed_month == 0).astype(int), missed_month, missed_amount)


def generate_synthetic_dataset(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return decision-time features and binary simulated payment outcomes."""
    dataset = simulate_dataset(rows, seed)
    return dataset.features, dataset.labels
