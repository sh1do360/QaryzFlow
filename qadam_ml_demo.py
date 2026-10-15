"""Synthetic-only model comparison for the QaryzFlow demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qadam_engine import BorrowerCase, Plan, free_cash


DEFAULT_ROWS = 8_000
DEFAULT_SEED = 20261012
MIN_TRAINING_ROWS = 500
MAX_SEED = 2_147_483_647
LOGISTIC_KEY = "logistic"
BOOSTING_KEY = "boosting"
BASELINE_KEY = "baseline"
RANKING_MODEL_KEYS = (LOGISTIC_KEY, BOOSTING_KEY)
RANKING_MODEL_LABELS = {
    LOGISTIC_KEY: "Логистическая регрессия",
    BOOSTING_KEY: "Градиентный бустинг",
}

FEATURE_NAMES = (
    "payment_share_of_free_cash",
    "balance_to_monthly_income",
    "other_debt_share",
    "annual_rate_pct",
    "term_months",
    "income_drop_share",
    "overdue_months",
    "reserve_months",
    "grace_months",
)
FEATURE_LABELS = {
    "payment_share_of_free_cash": "Максимальный платёж / свободный бюджет",
    "balance_to_monthly_income": "Остаток долга / месячный доход",
    "other_debt_share": "Платежи по другим долгам / доход",
    "annual_rate_pct": "Номинальная годовая ставка, %",
    "term_months": "Срок графика, мес.",
    "income_drop_share": "Снижение дохода, доля",
    "overdue_months": "Текущая просрочка, мес.",
    "reserve_months": "Резерв, мес. расходов",
    "grace_months": "Пауза, мес.",
}


@dataclass(frozen=True)
class CalibrationPoint:
    mean_score: float
    observed_success: float
    rows: int


@dataclass(frozen=True)
class ModelEvaluation:
    key: str
    label: str
    estimator: Any
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss_value: float
    mean_score: float
    calibration: tuple[CalibrationPoint, ...]
    permutation_importance: tuple[tuple[str, float], ...]
    burden_segments: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SyntheticModelDemo:
    models: tuple[ModelEvaluation, ...]
    rows_total: int
    rows_train: int
    rows_test: int
    train_success_share: float
    test_success_share: float
    seed: int
    holdout_examples: tuple[dict[str, object], ...]

    def get_model(self, key: str) -> ModelEvaluation:
        for model in self.models:
            if model.key == key:
                return model
        raise KeyError(f"Неизвестная демонстрационная модель: {key}")


def _synthetic_success_probability(features: np.ndarray) -> np.ndarray:
    """Nonlinear teaching rule; it is not estimated from borrower outcomes."""
    (
        payment_share,
        balance_ratio,
        other_debt_share,
        annual_rate,
        term,
        income_drop,
        overdue,
        reserve,
        grace,
    ) = features.T
    payment_pressure = np.maximum(payment_share - 0.60, 0.0)
    log_odds = (
        3.45
        - 2.65 * payment_share
        - 0.043 * balance_ratio
        - 0.70 * other_debt_share
        - 0.026 * annual_rate
        - 0.005 * term
        - 1.00 * income_drop
        - 0.23 * overdue
        + 0.11 * reserve
        - 0.09 * grace
        - 2.4 * payment_pressure**2
        - 1.8 * payment_pressure * income_drop
    )
    return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -30.0, 30.0)))


def generate_synthetic_dataset(
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create correlated fictional records with a shifted later cohort."""
    if not isinstance(rows, (int, np.integer)):
        raise ValueError("Количество строк должно быть целым числом.")
    if rows < 100:
        raise ValueError("Для генератора нужно не меньше 100 искусственных строк.")
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > MAX_SEED:
        raise ValueError(f"Seed должен быть целым числом от 0 до {MAX_SEED}.")

    rng = np.random.default_rng(seed)
    future_cohort = np.arange(rows) >= int(rows * 0.80)
    financial_pressure = rng.beta(2.1, 2.4, rows)

    payment_share = (
        0.06
        + 1.03 * financial_pressure
        + rng.normal(0.0, 0.11, rows)
        + 0.055 * future_cohort
    )
    balance_ratio = (
        1.5
        + 23.0 * financial_pressure
        + rng.normal(0.0, 3.8, rows)
    )
    other_debt_share = (
        0.015
        + 0.39 * financial_pressure
        + rng.normal(0.0, 0.045, rows)
    )
    annual_rate = (
        7.0
        + 25.0 * financial_pressure
        + rng.normal(0.0, 4.0, rows)
        + 3.0 * future_cohort
    )
    term = np.rint(
        12.0 + 145.0 * financial_pressure + rng.normal(0.0, 13.0, rows)
    )
    income_drop = (
        rng.beta(1.35, 8.0, rows) * 0.70
        + 0.10 * future_cohort
    )
    overdue_probability = np.clip(
        0.035 + 0.50 * financial_pressure + 0.08 * future_cohort,
        0.0,
        0.90,
    )
    overdue_active = rng.random(rows) < overdue_probability
    overdue_months = np.where(overdue_active, rng.integers(1, 7, rows), 0)
    reserve_months = (
        7.3 * (1.0 - financial_pressure)
        + rng.normal(0.0, 0.95, rows)
        - 0.65 * future_cohort
    )
    grace_months = np.rint(
        5.0 * financial_pressure + rng.normal(0.0, 1.4, rows)
    )

    features = np.column_stack(
        (
            np.clip(payment_share, 0.05, 1.25),
            np.clip(balance_ratio, 0.5, 30.0),
            np.clip(other_debt_share, 0.0, 0.50),
            np.clip(annual_rate, 0.0, 45.0),
            np.clip(term, 1.0, 180.0),
            np.clip(income_drop, 0.0, 0.70),
            overdue_months,
            np.clip(reserve_months, 0.0, 8.0),
            np.clip(grace_months, 0.0, 6.0),
        )
    ).astype(float)

    probabilities = _synthetic_success_probability(features)
    labels = rng.binomial(1, probabilities).astype(int)
    return features, labels


def _calibration_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 8,
) -> tuple[CalibrationPoint, ...]:
    ordered = np.argsort(probabilities)
    groups = (group for group in np.array_split(ordered, bins) if len(group))
    return tuple(
        CalibrationPoint(
            mean_score=float(probabilities[group].mean()),
            observed_success=float(labels[group].mean()),
            rows=int(len(group)),
        )
        for group in groups
    )


def _permutation_auc_importance(
    estimator: Any,
    features: np.ndarray,
    labels: np.ndarray,
    baseline_auc: float,
    seed: int,
) -> tuple[tuple[str, float], ...]:
    rng = np.random.default_rng(seed)
    importances: list[tuple[str, float]] = []
    for index, name in enumerate(FEATURE_NAMES):
        shuffled = features.copy()
        shuffled[:, index] = rng.permutation(shuffled[:, index])
        shuffled_scores = estimator.predict_proba(shuffled)[:, 1]
        shuffled_auc = roc_auc_score(labels, shuffled_scores)
        importances.append((FEATURE_LABELS[name], float(baseline_auc - shuffled_auc)))
    return tuple(sorted(importances, key=lambda item: item[1], reverse=True))


def _burden_segments(
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, object], ...]:
    payment_share = features[:, 0]
    bands = (
        ("Низкая нагрузка", payment_share <= 0.35),
        ("Умеренная", (payment_share > 0.35) & (payment_share <= 0.60)),
        ("Высокая", (payment_share > 0.60) & (payment_share <= 0.85)),
        ("Очень высокая", payment_share > 0.85),
    )
    rows: list[dict[str, object]] = []
    for label, mask in bands:
        if not np.any(mask):
            continue
        rows.append(
            {
                "Группа по платёжной нагрузке": label,
                "Примеров": int(mask.sum()),
                "Доля успеха в симуляции": float(labels[mask].mean()),
                "Средний синтетический скор": float(probabilities[mask].mean()),
                "Brier score": float(
                    brier_score_loss(labels[mask], probabilities[mask])
                ),
            }
        )
    return tuple(rows)


def train_synthetic_model(
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> SyntheticModelDemo:
    """Fit three models and compare them on a later synthetic cohort."""
    if rows < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Для обучения и контрольной когорты нужно не меньше {MIN_TRAINING_ROWS} строк."
        )

    features, labels = generate_synthetic_dataset(rows, seed)
    split_at = int(rows * 0.80)
    x_train, x_test = features[:split_at], features[split_at:]
    y_train, y_test = labels[:split_at], labels[split_at:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("В одной из синтетических когорт не хватает обоих классов.")

    candidates: tuple[tuple[str, str, Any, bool], ...] = (
        (
            LOGISTIC_KEY,
            RANKING_MODEL_LABELS[LOGISTIC_KEY],
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1_000, random_state=seed),
                    ),
                ]
            ),
            True,
        ),
        (
            BOOSTING_KEY,
            RANKING_MODEL_LABELS[BOOSTING_KEY],
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=seed,
            ),
            True,
        ),
        (
            BASELINE_KEY,
            "Базовый уровень: частота успеха",
            DummyClassifier(strategy="prior"),
            False,
        ),
    )

    evaluations: list[ModelEvaluation] = []
    holdout_probabilities: dict[str, np.ndarray] = {}
    for key, label, estimator, explainable in candidates:
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_test)[:, 1]
        holdout_probabilities[key] = probabilities
        auc = float(roc_auc_score(y_test, probabilities))
        evaluations.append(
            ModelEvaluation(
                key=key,
                label=label,
                estimator=estimator,
                roc_auc=auc,
                average_precision=float(
                    average_precision_score(y_test, probabilities)
                ),
                brier_score=float(brier_score_loss(y_test, probabilities)),
                log_loss_value=float(log_loss(y_test, probabilities, labels=[0, 1])),
                mean_score=float(probabilities.mean()),
                calibration=_calibration_points(y_test, probabilities),
                permutation_importance=(
                    _permutation_auc_importance(
                        estimator,
                        x_test,
                        y_test,
                        auc,
                        seed + len(evaluations) + 1,
                    )
                    if explainable
                    else ()
                ),
                burden_segments=(
                    _burden_segments(x_test, y_test, probabilities)
                    if explainable
                    else ()
                ),
            )
        )

    example_rows: list[dict[str, object]] = []
    for index, (feature_row, label) in enumerate(
        zip(x_test[:8], y_test[:8], strict=True)
    ):
        row: dict[str, object] = {
            FEATURE_LABELS[name]: round(float(value), 3)
            for name, value in zip(FEATURE_NAMES, feature_row, strict=True)
        }
        row["Искусственная метка"] = "Успех" if label else "Неуспех"
        row["Скор логистической регрессии"] = round(
            float(holdout_probabilities[LOGISTIC_KEY][index]), 3
        )
        row["Скор градиентного бустинга"] = round(
            float(holdout_probabilities[BOOSTING_KEY][index]), 3
        )
        example_rows.append(row)

    return SyntheticModelDemo(
        models=tuple(evaluations),
        rows_total=rows,
        rows_train=len(y_train),
        rows_test=len(y_test),
        train_success_share=float(y_train.mean()),
        test_success_share=float(y_test.mean()),
        seed=seed,
        holdout_examples=tuple(example_rows),
    )


def plan_features(case: BorrowerCase, plan: Plan) -> np.ndarray:
    """Build model inputs, bounding ratios to the synthetic generator range."""
    cash = free_cash(case)
    income = max(0.0, case.income) * (1.0 - max(0.0, min(0.70, case.income_drop)))
    payment_share = plan.peak_monthly_payment / max(cash, 1.0)
    balance_ratio = max(0.0, case.balance) / max(income, 1.0)
    other_debt_share = max(0.0, case.other_debt_payment) / max(income, 1.0)
    reserve_months = max(0.0, case.savings) / max(case.essential_expenses, 1.0)
    return np.asarray(
        [[
            payment_share,
            max(0.5, min(30.0, balance_ratio)),
            max(0.0, min(0.50, other_debt_share)),
            max(0.0, case.annual_rate),
            plan.term_months,
            max(0.0, min(0.70, case.income_drop)),
            max(0, min(6, case.overdue_months)),
            max(0.0, min(8.0, reserve_months)),
            plan.grace_months,
        ]],
        dtype=float,
    )


def score_plan(
    demo: SyntheticModelDemo,
    case: BorrowerCase,
    plan: Plan,
    model_key: str = LOGISTIC_KEY,
) -> float:
    """Return a synthetic-only score for one candidate and selected model."""
    estimator = demo.get_model(model_key).estimator
    return float(estimator.predict_proba(plan_features(case, plan))[0, 1])
