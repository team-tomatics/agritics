import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


HTML = Path("serving_app/static/index.html").read_text(encoding="utf-8")


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])


class DashboardTest(unittest.TestCase):
    def test_required_panels_have_unique_ids(self):
        parser = IdCollector()
        parser.feed(HTML)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for element_id in (
            "metric-requests", "metric-latency", "metric-success", "metric-rmse",
            "metric-drift", "pipeline-track", "model-history", "alert-list",
            "upload-input", "drift-result",
        ):
            self.assertIn(element_id, parser.ids)

    def test_dashboard_uses_real_endpoints_without_reference_fake_values(self):
        for endpoint in ("/metrics?minutes=", "/health", "/logs/aiops.log", "/report", "/data/status"):
            self.assertIn(endpoint, HTML)
        for fake_value in ("1,338", "150 ms", "100.0%", "HAIC_Predictor"):
            self.assertNotIn(fake_value, HTML)

    def test_shared_constants_remain_machine_checkable(self):
        for name in ("SEQ_LEN", "WINDOW_SIZE", "RMSE_THRESHOLD", "WAPE_THRESHOLD"):
            self.assertRegex(HTML, rf"const {name} = [0-9.]+;")

    def test_drift_threshold_matches_server_and_is_shown_as_wape(self):
        detector = Path("serving_app/monitoring/drift_detector.py").read_text(encoding="utf-8")
        server = float(re.search(r"^WAPE_THRESHOLD = ([0-9.]+)", detector, re.M).group(1))
        html = float(re.search(r"const WAPE_THRESHOLD = ([0-9.]+);", HTML).group(1))
        self.assertEqual(html, server)
        self.assertIn('["감지", "WAPE 판정"]', HTML)
        self.assertIn("data.drift_check?.wape", HTML)

    def test_retrain_failure_is_visible_in_alerts_and_pipeline(self):
        self.assertGreaterEqual(HTML.count('event.line.includes("[FAIL]")'), 2)
        self.assertIn('last.line.includes("[FAIL]")', HTML)


if __name__ == "__main__":
    unittest.main()
