import json
from pathlib import Path
import unittest

from pyarrow.ipc import open_stream
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppRenderingTests(unittest.TestCase):
    def test_all_pages_render_without_exceptions(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio(key="page").options, ["Подбор графика", "Проверка проекта"])
        app.radio(key="page").set_value("Проверка проекта").run(timeout=30)
        for section, metric in (
            ("Портфель", "Вымышленные кейсы"),
            ("Как считается", "Перебрано графиков"),
            ("ML-лаборатория", "Искусственных примеров"),
        ):
            with self.subTest(section=section):
                app.segmented_control(key="project_section").set_value(section).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertIn(metric, [item.label for item in app.metric])
                if section == "ML-лаборатория":
                    examples = next(table.value for table in app.dataframe if "Первый пропуск, месяц графика" in table.value.columns)
                    self.assertEqual(len(examples), 8)

    def test_current_schedule_is_explicit_when_no_savings_are_available(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1370")
        app.selectbox(key="schedule_objective").set_value("cost").run(timeout=30)

        self.assertTrue(any("График для обсуждения · Текущий график" in block.value for block in app.markdown))
        keep = app.button(key="review_plan_Q-1370_cost")
        self.assertEqual(keep.label, "Оставить текущий график")
        keep.click().run(timeout=30)
        self.assertEqual(app.selectbox(key="review_plan_Q-1370").value, "current")

    def test_affordable_plans_can_all_fail_stress_checks(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1042")
        app.slider(key="budget_share").set_value(100).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        self.assertGreater(len(app.selectbox(key="review_plan_Q-1042").options), 1)
        self.assertTrue(any("Ни один доступный график" in caption.value for caption in app.caption))
        self.assertTrue(any("наибольшая нехватка" in caption.value and "₸" in caption.value for caption in app.caption))

    def test_no_affordable_plan_is_rendered_without_a_recommendation(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1042").run(timeout=30)
        app.slider(key="drop_Q-1042").set_value(70)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("подходящего графика нет" in warning.value for warning in app.warning))
        self.assertFalse(any(widget.key == "review_plan_Q-1042" for widget in app.selectbox))
        self.assertFalse(any(button.label == "Скачать краткое сравнение" for button in app.get("download_button")))
        self.assertTrue(app.toggle(key="case_ml_comparison").proto.disabled)

    def test_case_compares_ml_with_the_rule_under_income_recovery(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1176")
        app.selectbox(key="schedule_objective").set_value("cost").run(timeout=30)
        self.assertTrue(any("Срок 30 мес. · без паузы" in block.value for block in app.markdown))

        app.slider(key="recovery_Q-1176").set_value(3).run(timeout=30)
        recommendation = next(block.value for block in app.markdown if "График для обсуждения ·" in block.value)
        self.assertIn("Срок 19 мес. · пауза 1 мес.", recommendation)

        app.toggle(key="case_ml_comparison").set_value(True).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertIn(recommendation, [block.value for block in app.markdown])
        comparison = next(table.value for table in app.dataframe if "Учебный балл (0–100)" in table.value.columns)
        self.assertEqual(comparison["Подход"].tolist(), ["По правилам", "По ML в симуляции"])

    def test_manual_selection_updates_calendar_and_resets_if_plan_is_removed(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1176")
        app.selectbox(key="schedule_objective").set_value("cost").run(timeout=30)
        app.slider(key="recovery_Q-1176").set_value(3).run(timeout=30)
        app.slider(key="recovery_wait_Q-1176").set_value(3).run(timeout=30)
        self.assertTrue(any("График для обсуждения · Срок 20 мес. · пауза 3 мес." in block.value for block in app.markdown))

        app.button(key="review_plan_Q-1176_affordable").click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox(key="review_plan_Q-1176").value, "term-67-grace-0")
        self.assertTrue(any("График для обсуждения · Срок 67 мес." in block.value for block in app.markdown))
        self.assertTrue(app.button(key="review_plan_Q-1176_affordable").proto.disabled)
        payment = next(metric for metric in app.metric if metric.label == "Максимальный платёж")
        self.assertIn("меньше", payment.delta)
        self.assertTrue({"Календарь выплат", "Стресс-проверки", "Все варианты", "ML-сравнение"} <= {tab.label for tab in app.get("tab")})
        ledgers = [table.value for table in app.dataframe if "Долг на конец" in table.value.columns]
        self.assertEqual([len(ledger) for ledger in ledgers], [19, 67])
        self.assertTrue(any(button.label == "Скачать краткое сравнение" for button in app.get("download_button")))

        app.slider(key="max_extension_months").set_value(15).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox(key="review_plan_Q-1176").value, "auto")
        self.assertTrue(any("вышел за пределы" in message.value for message in app.info))

    def test_case_edits_survive_switching_profiles_and_can_be_reset(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual(app.selectbox(key="case_selector").value, "Q-1176")
        app.slider(key="recovery_Q-1176").set_value(3)
        app.slider(key="max_extension_months").set_value(40).run(timeout=30)

        app.selectbox(key="case_selector").set_value("Q-1320").run(timeout=30)
        self.assertEqual(app.slider(key="recovery_Q-1320").value, 0)
        app.selectbox(key="case_selector").set_value("Q-1176").run(timeout=30)
        self.assertEqual(app.slider(key="recovery_Q-1176").value, 3)

        app.button(key="reset_case_Q-1176").click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.slider(key="recovery_Q-1176").value, 0)
        self.assertEqual(app.slider(key="max_extension_months").value, 40)

    def test_calendar_marks_a_final_payment_above_the_limit(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.selectbox(key="case_selector").set_value("Q-1320").run(timeout=30)
        self.assertEqual(len(app.exception), 0)

        def above_limit_months() -> list[tuple[int, float]]:
            chart = app.get("vega_lite_chart")[0].proto
            spec = json.loads(chart.spec)
            self.assertEqual(spec["layer"][1]["mark"]["shape"], "diamond")
            self.assertIn("Превышение", spec["layer"][1]["transform"][0]["filter"])
            data = chart.data.data or chart.datasets[0].data.data
            rows = open_stream(data).read_all().to_pylist()
            return [(row["Месяц"], row["Превышение"]) for row in rows if row["Превышение"] > 0]

        marked = above_limit_months()
        self.assertEqual([month for month, _ in marked], [12])
        self.assertAlmostEqual(marked[0][1], 327_428.47, places=2)
        current_ledger = next(table.value for table in app.dataframe if "Выше лимита" in table.value.columns)
        self.assertEqual(current_ledger.iloc[-1]["Выше лимита"], "327 428,47 ₸")

        app.slider(key="term_Q-1320").set_value(18).run(timeout=30)
        self.assertEqual(above_limit_months(), [])
        self.assertTrue(any("Показать весь срок" in message.value for message in app.info))
        app.toggle(key="calendar_full_Q-1320").set_value(True).run(timeout=30)
        self.assertEqual([month for month, _ in above_limit_months()], [18])


if __name__ == "__main__":
    unittest.main()
