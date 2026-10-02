import ast
import csv
import math
import re
import statistics
import unittest
from pathlib import Path


class TomatoDriftScenarioTests(unittest.TestCase):
    def test_committed_data_and_dashboard_use_same_baseline(self):
        with Path("data/tomato_prices.csv").open(encoding="utf-8") as source:
            closes = [float(row["Close"]) for row in csv.DictReader(source)]
        mean = statistics.fmean(closes)
        sigma = statistics.pstdev(math.log(b / a) for a, b in zip(closes, closes[1:]))

        tree = ast.parse(Path("scripts/simulate_drift.py").read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertAlmostEqual(mean, constants["DEFAULT_BASE_PRICE"], places=2)
        self.assertAlmostEqual(sigma, constants["NORMAL_SIGMA"], places=6)

        html = Path("serving_app/static/index.html").read_text(encoding="utf-8")
        html_base = float(re.search(r"const DEFAULT_BASE_PRICE = ([0-9.]+)", html).group(1))
        html_sigma = float(re.search(r"const NORMAL_SIGMA = ([0-9.]+)", html).group(1))
        self.assertEqual(html_base, constants["DEFAULT_BASE_PRICE"])
        self.assertEqual(html_sigma, constants["NORMAL_SIGMA"])


if __name__ == "__main__":
    unittest.main()
