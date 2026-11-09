"""Synthetic-only model comparison for the QaryzFlow demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
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

from qadam_engine import BorrowerCase, Plan
from qadam_synthetic import (
    DEFAULT_ROWS,
    DEFAULT_SEED,
    FEATURE_LABELS,
    FEATURE_NAMES,
    OBSERVED_PAYMENTS,
    SCENARIOS,
    generate_synthetic_dataset,
    plan_features,
    simulate_dataset,
)


MIN_TRAINING_ROWS = 500
LOGISTIC_KEY = "logistic"
BOOSTING_KEY = "boosting"
BASELINE_KEY = "baseline"
BURDEN_BASELINE_KEY = "burden_baseline"
RANKING_MODEL_KEYS = (LOGISTIC_KEY, BOOSTING_KEY)
RANKING_MODEL_LABELS = {
    LOGISTIC_KEY: "Логистическая регрессия",
    BOOSTING_KEY: "Градиентный бустинг",
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
    scenario_summary: tuple[dict[str, object], ...]
    holdout_examples: tuple[dict[str, object], ...]

    def get_model(self, key: str) -> ModelEvaluation:
        for model in self.models:
            if model.key == key:
                return model
        raise KeyError(f"Неизвестная демонстрационная модель: {key}")




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
    """Compare two full models and two baselines on a later synthetic cohort."""
    if rows < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Для обучения и контрольной когорты нужно не меньше {MIN_TRAINING_ROWS} строк."
        )

    dataset = simulate_dataset(rows, seed)
    features, labels = dataset.features, dataset.labels
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
            BURDEN_BASELINE_KEY,
            "Базовый ориентир: только доля платежа",
            Pipeline(
                [
                    (
                        "payment_share",
                        ColumnTransformer([("burden", "passthrough", [FEATURE_NAMES.index("payment_share_of_free_cash")])]),
                    ),
                    ("classifier", LogisticRegression(max_iter=1_000, random_state=seed)),
                ]
            ),
            False,
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
        loan = dataset.loans[split_at + index]
        missed_month = int(dataset.missed_month[split_at + index])
        row: dict[str, object] = {
            "Учебная запись": loan.case.case_id,
            "Профиль дохода": loan.scenario,
            "Остаток долга, ₸": loan.case.balance,
            "Расчётный платёж, ₸": round(loan.payment, 2),
            "Платежей в горизонте": min(OBSERVED_PAYMENTS, loan.term_months - loan.grace_months),
            "Первый пропуск, месяц графика": missed_month or None,
            "Нехватка на платёж, ₸": round(float(dataset.missed_amount[split_at + index]), 2) if missed_month else None,
            "Исход в симуляции": "Все платежи вовремя" if label else "Неполный платёж",
            **{
                FEATURE_LABELS[name]: round(float(value), 3)
                for name, value in zip(FEATURE_NAMES, feature_row, strict=True)
            },
        }
        row["Скор логистической регрессии"] = round(
            float(holdout_probabilities[LOGISTIC_KEY][index]), 3
        )
        row["Скор градиентного бустинга"] = round(
            float(holdout_probabilities[BOOSTING_KEY][index]), 3
        )
        example_rows.append(row)

    scenarios = np.asarray([loan.scenario for loan in dataset.loans])
    scenario_summary = []
    for scenario in SCENARIOS:
        train_matches = scenarios[:split_at] == scenario
        test_matches = scenarios[split_at:] == scenario
        scenario_summary.append({
            "Профиль дохода": scenario,
            "Обучение": int(train_matches.sum()),
            "Поздняя проверка": int(test_matches.sum()),
            "Без пропусков в проверке, %": round(float(y_test[test_matches].mean() * 100), 1) if np.any(test_matches) else None,
        })

    return SyntheticModelDemo(
        models=tuple(evaluations),
        rows_total=rows,
        rows_train=len(y_train),
        rows_test=len(y_test),
        train_success_share=float(y_train.mean()),
        test_success_share=float(y_test.mean()),
        seed=seed,
        scenario_summary=tuple(scenario_summary),
        holdout_examples=tuple(example_rows),
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


def rank_feasible_plans(
    demo: SyntheticModelDemo,
    case: BorrowerCase,
    plans: list[Plan],
    model_key: str,
) -> list[tuple[float, Plan]]:
    """Rank the budget-approved candidates in one model call."""
    eligible = [plan for plan in plans if plan.is_feasible]
    if not eligible:
        return []
    features = np.concatenate([plan_features(case, plan) for plan in eligible])
    scores = demo.get_model(model_key).estimator.predict_proba(features)[:, 1]
    return sorted(
        zip(map(float, scores), eligible, strict=True),
        key=lambda item: (-item[0], item[1].term_months, item[1].grace_months),
    )


def score_sensitivity(
    demo: SyntheticModelDemo,
    case: BorrowerCase,
    rule_plan: Plan,
    ml_plan: Plan,
    model_key: str,
) -> list[tuple[str, float]]:
    """Measure score changes when replacing one input of the rule's plan at a time."""
    original = plan_features(case, rule_plan)[0]
    alternative = plan_features(case, ml_plan)[0]
    changed = np.flatnonzero(original != alternative)
    inputs = np.tile(original, (len(changed) + 1, 1))
    inputs[np.arange(1, len(inputs)), changed] = alternative[changed]
    scores = demo.get_model(model_key).estimator.predict_proba(inputs)[:, 1]
    return sorted(
        (
            (FEATURE_LABELS[FEATURE_NAMES[index]], float(score - scores[0]))
            for index, score in zip(changed, scores[1:], strict=True)
        ),
        key=lambda item: -abs(item[1]),
    )
