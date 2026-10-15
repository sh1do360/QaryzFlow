from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppRenderingTests(unittest.TestCase):
    def test_all_pages_render_without_exceptions(self) -> None:
        pages = (
            "Кейс клиента",
            "Портфель",
            "Оптимизатор и контроль",
            "ML-лаборатория",
        )
        for page in pages:
            with self.subTest(page=page):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                if page != pages[0]:
                    app.radio[0].set_value(page)
                    app.run(timeout=30)
                self.assertEqual(len(app.exception), 0)

    def test_no_affordable_plan_is_rendered_without_a_recommendation(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.slider[1].set_value(70)
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.warning), 0)


if __name__ == "__main__":
    unittest.main()
