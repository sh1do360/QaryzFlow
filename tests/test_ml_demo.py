import unittest
from dataclasses import replace

import numpy as np

from qadam_engine import CASES, make_plans
from qadam_ml_demo import (
    BOOSTING_KEY,
    FEATURE_NAMES,
    LOGISTIC_KEY,
    generate_synthetic_dataset,
    score_plan,
    train_synthetic_model,
)


class SyntheticModelDemoTests(unittest.TestCase):
    def test_synthetic_dataset_is_reproducible_and_binary(self) -> None:
        first_features, first_labels = generate_synthetic_dataset(rows=512, seed=73)
        second_features, second_labels = generate_synthetic_dataset(rows=512, seed=73)

        np.testing.assert_array_equal(first_features, second_features)
        np.testing.assert_array_equal(first_labels, second_labels)
        self.assertEqual(first_features.shape, (512, len(FEATURE_NAMES)))
        self.assertEqual(set(np.unique(first_labels)), {0, 1})

    def test_training_reports_holdout_metrics_and_examples(self) -> None:
        model = train_synthetic_model(rows=1_000, seed=73)

        self.assertEqual(model.rows_total, 1_000)
        self.assertEqual(model.rows_train, 800)
        self.assertEqual(model.rows_test, 200)
        self.assertEqual(
            {item.key for item in model.models},
            {LOGISTIC_KEY, BOOSTING_KEY, "baseline"},
        )
        self.assertEqual(len(model.holdout_examples), 8)
        self.assertGreater(model.train_success_share, 0.0)
        self.assertLess(model.train_success_share, 1.0)
        self.assertGreater(model.test_success_share, 0.0)
        self.assertLess(model.test_success_share, 1.0)
        for item in model.models:
            with self.subTest(model=item.key):
                self.assertGreaterEqual(item.roc_auc, 0.0)
                self.assertLessEqual(item.roc_auc, 1.0)
                self.assertGreaterEqual(item.average_precision, 0.0)
                self.assertLessEqual(item.average_precision, 1.0)
                self.assertGreaterEqual(item.brier_score, 0.0)
                self.assertLessEqual(item.brier_score, 1.0)
                self.assertGreaterEqual(item.log_loss_value, 0.0)
                self.assertEqual(sum(point.rows for point in item.calibration), 200)
        self.assertEqual(
            len(model.get_model(LOGISTIC_KEY).permutation_importance),
            len(FEATURE_NAMES),
        )
        self.assertEqual(
            sum(int(row["Примеров"]) for row in model.get_model(LOGISTIC_KEY).burden_segments),
            model.rows_test,
        )

    def test_later_synthetic_cohort_has_documented_stress_shift(self) -> None:
        features, _ = generate_synthetic_dataset(rows=2_000, seed=73)
        split_at = 1_600

        self.assertGreater(features[split_at:, 0].mean(), features[:split_at, 0].mean())
        self.assertGreater(features[split_at:, 3].mean(), features[:split_at, 3].mean())
        self.assertLess(features[split_at:, 7].mean(), features[:split_at, 7].mean())

    def test_training_requires_a_minimum_cohort_size(self) -> None:
        with self.assertRaises(ValueError):
            train_synthetic_model(rows=499, seed=73)

    def test_plan_score_is_bounded(self) -> None:
        case = replace(
            CASES[0],
            income=1_000_000,
            income_drop=0.0,
            essential_expenses=100_000,
            other_debt_payment=0.0,
        )
        plans = make_plans(
            case,
            budget_share=1.0,
            max_extension_months=24,
            max_grace_months=2,
        )
        feasible_plan = next(plan for plan in plans if plan.is_feasible)
        model = train_synthetic_model(rows=1_000, seed=73)

        for model_key in (LOGISTIC_KEY, BOOSTING_KEY):
            with self.subTest(model=model_key):
                score = score_plan(model, case, feasible_plan, model_key)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
