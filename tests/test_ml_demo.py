import unittest
from dataclasses import replace

import numpy as np

from qadam_engine import CASES, make_plans
from qadam_ml_demo import (
    BOOSTING_KEY,
    BURDEN_BASELINE_KEY,
    FEATURE_NAMES,
    LOGISTIC_KEY,
    generate_synthetic_dataset,
    rank_feasible_plans,
    score_plan,
    train_synthetic_model,
)
from qadam_synthetic import OBSERVED_PAYMENTS, first_missed_payment, plan_features, simulate_dataset


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
            {LOGISTIC_KEY, BOOSTING_KEY, "baseline", BURDEN_BASELINE_KEY},
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
        baseline = model.get_model(BURDEN_BASELINE_KEY).estimator
        share = FEATURE_NAMES.index("payment_share_of_free_cash")
        examples = np.zeros((2, len(FEATURE_NAMES)))
        examples[:, share] = (0.3, 0.8)
        changed = examples + 1
        changed[:, share] = examples[:, share]
        probabilities = baseline.predict_proba(examples)[:, 1]
        np.testing.assert_allclose(probabilities, baseline.predict_proba(changed)[:, 1])
        self.assertGreater(probabilities[0], probabilities[1])

    def test_later_synthetic_cohort_has_documented_stress_shift(self) -> None:
        features, labels = generate_synthetic_dataset(rows=2_000, seed=73)
        split_at = 1_600
        rate = FEATURE_NAMES.index("annual_rate_pct")
        reserve = FEATURE_NAMES.index("reserve_months")
        self.assertGreater(features[split_at:, rate].mean(), features[:split_at, rate].mean())
        self.assertLess(features[split_at:, reserve].mean(), features[:split_at, reserve].mean())
        self.assertLess(labels[split_at:].mean(), labels[:split_at].mean())

    def test_simulated_features_match_the_engine_and_misses_follow_the_pause(self) -> None:
        dataset = simulate_dataset(rows=256, seed=73)
        index, loan = next((i, loan) for i, loan in enumerate(dataset.loans) if loan.grace_months > 0)
        plans = make_plans(
            loan.case,
            budget_share=1.0,
            max_extension_months=loan.term_months - loan.case.remaining_months,
            max_grace_months=loan.grace_months,
        )
        plan = next(plan for plan in plans if plan.key == f"term-{loan.term_months}-grace-{loan.grace_months}")
        self.assertTrue(plan.is_feasible)
        self.assertAlmostEqual(plan.monthly_payment, loan.payment)
        np.testing.assert_allclose(plan_features(loan.case, plan)[0], dataset.features[index])
        self.assertTrue(any(item > 0 for item in dataset.missed_month))
        for simulated_loan, label, month, amount in zip(
            dataset.loans, dataset.labels, dataset.missed_month, dataset.missed_amount, strict=True
        ):
            self.assertEqual(label, int(month == 0))
            if month:
                self.assertGreater(month, simulated_loan.grace_months)
                self.assertLessEqual(month, min(simulated_loan.term_months, simulated_loan.grace_months + OBSERVED_PAYMENTS))
                self.assertGreater(amount, 0.0)
                self.assertLessEqual(amount, simulated_loan.payment)

    def test_savings_can_cover_a_payment_but_a_pause_can_drain_the_reserve(self) -> None:
        scheduled = np.array([[0, 80, 80, 0], [0, 80, 80, 0], [0, 0, 70, 70]], dtype=float)
        cash = np.array([[-30, 60, 95, 0], [-30, 60, 95, 0], [-60, -10, 100, 65]], dtype=float)
        months, amounts = first_missed_payment(scheduled, cash, np.array([40, 55, 20], dtype=float))
        np.testing.assert_array_equal(months, [2, 0, 4])
        np.testing.assert_array_equal(amounts, [10, 0, 5])

    def test_training_requires_a_minimum_cohort_size(self) -> None:
        with self.assertRaises(ValueError):
            train_synthetic_model(rows=499, seed=73)

    def test_ml_ranks_only_budget_approved_plans(self) -> None:
        case = replace(CASES[3], income_recovery_months=3)
        plans = make_plans(case, max_extension_months=24, max_grace_months=2)
        eligible = {plan.key for plan in plans if plan.is_feasible}
        self.assertTrue(0 < len(eligible) < len(plans))
        model = train_synthetic_model(rows=1_000, seed=73)

        for model_key in (LOGISTIC_KEY, BOOSTING_KEY):
            with self.subTest(model=model_key):
                ranked = rank_feasible_plans(model, case, plans, model_key)
                self.assertEqual({plan.key for _, plan in ranked}, eligible)
                scores = [score for score, _ in ranked]
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
                best_score, best_plan = ranked[0]
                self.assertAlmostEqual(best_score, score_plan(model, case, best_plan, model_key))


if __name__ == "__main__":
    unittest.main()
