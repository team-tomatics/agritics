import re
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import serving_app
from serving_app.monitoring import drift_detector, retrain_trigger


class RetrainWindowTests(unittest.TestCase):
    def test_window_is_cleared_only_after_promotion(self):
        features = types.ModuleType("data.features")
        features.SEQ_LEN = 25
        features.load_rows = lambda _path: [{}] * 50
        storage = types.ModuleType("data.storage")
        storage.latest_upload = lambda: "unused.csv"

        for promoted in (True, False):
            with self.subTest(promoted=promoted):
                records = [{"predicted": 1, "actual": 2}] * 15
                trainer = types.ModuleType("serving_app.train_and_register")
                trainer.fine_tune = lambda rows, promoted=promoted: {
                    "promoted": promoted,
                    "rmse": 100.0,
                    **({"version": "2"} if promoted else {}),
                }
                loader = SimpleNamespace(_model_cache="old-model")

                with (
                    patch.object(retrain_trigger, "is_drift", return_value=True),
                    patch.object(serving_app, "model_loader", loader, create=True),
                    patch.dict(sys.modules, {
                        "data.features": features,
                        "data.storage": storage,
                        "serving_app.train_and_register": trainer,
                    }),
                ):
                    with self.assertLogs("aiops", level="WARNING") as logs:
                        result = retrain_trigger.check_and_trigger(records)

                self.assertEqual(result["promoted"], promoted)
                self.assertEqual(records, [] if promoted else [{"predicted": 1, "actual": 2}] * 15)
                self.assertEqual(loader._model_cache, None if promoted else "old-model")
                failure_logs = [message for message in logs.output if "[FAIL]" in message]
                if promoted:
                    self.assertEqual(failure_logs, [])
                else:
                    self.assertEqual(
                        failure_logs,
                        ["WARNING:aiops:[FAIL] new_rmse=100.00 - gate/regression failed, keep current Production"],
                    )


class WapeDriftTests(unittest.TestCase):
    """#105 드리프트 판정 = 최근 15건 WAPE > 18%."""

    def test_compute_wape_matches_definition(self):
        records = [{"predicted": 3300, "actual": 3000}, {"predicted": 3600, "actual": 4000}]
        self.assertAlmostEqual(drift_detector.compute_wape(records), 10.0)
        self.assertEqual(drift_detector.compute_wape([]), 0.0)

    def test_is_drift_uses_last_window_and_strict_threshold(self):
        def window(wape_pct, n=drift_detector.WINDOW_SIZE):
            return [{"predicted": 100 + wape_pct, "actual": 100}] * n

        self.assertFalse(drift_detector.is_drift(window(30, n=drift_detector.WINDOW_SIZE - 1)))
        self.assertFalse(drift_detector.is_drift(window(drift_detector.WAPE_THRESHOLD)))
        self.assertTrue(drift_detector.is_drift(window(drift_detector.WAPE_THRESHOLD + 0.5)))
        # 오래된 큰 오차는 창 밖 — 최근 15건만 본다
        self.assertFalse(drift_detector.is_drift(window(90, n=5) + window(5)))

    def test_ok_response_reports_window_wape(self):
        records = [{"predicted": 106, "actual": 100}] * drift_detector.WINDOW_SIZE
        self.assertEqual(retrain_trigger.check_and_trigger(records), {"status": "ok", "wape": 6.0})


class RetrainLogFormatTests(unittest.TestCase):
    """#105 [WARN] 에 wape, [OK] · [FAIL] 뒤에 새 모델 mae · wape · bias — 앞부분은 대시보드 · 보고서가 읽는 형식 그대로."""

    def _run(self, promoted):
        features = types.ModuleType("data.features")
        features.SEQ_LEN = 25
        features.load_rows = lambda _path: [{}] * 50
        storage = types.ModuleType("data.storage")
        storage.latest_upload = lambda: "unused.csv"
        trainer = types.ModuleType("serving_app.train_and_register")
        trainer.fine_tune = lambda rows: {
            "promoted": promoted, "rmse": 540.04, "mae": 412.34, "wape": 11.23, "bias": -1.6,
            **({"version": "2"} if promoted else {}),
        }
        records = [{"predicted": 125, "actual": 100}] * 15  # WAPE 25%
        with (
            patch.object(serving_app, "model_loader", SimpleNamespace(_model_cache="old"), create=True),
            patch.dict(sys.modules, {
                "data.features": features,
                "data.storage": storage,
                "serving_app.train_and_register": trainer,
            }),
            self.assertLogs("aiops", level="INFO") as logs,
        ):
            result = retrain_trigger.check_and_trigger(records)
        return result, [line.split(":", 2)[2] for line in logs.output]

    def test_promoted_logs_keep_parsed_prefix(self):
        result, lines = self._run(promoted=True)
        self.assertEqual(result, {"status": "retrain_triggered", "promoted": True, "rmse": 540.04, "wape": 25.0})
        self.assertEqual(lines[0], "[WARN] drift detected - triggering retrain (wape=25.00%)")
        ok = lines[-1]
        self.assertEqual(
            ok,
            "[OK] new_rmse=540.04 - production promoted: Tomato_Price_Predictor v2 (mae=412.3 wape=11.2% bias=-1.6%)",
        )
        # report_sidecar/generator.py 의 승격 정규식이 그대로 읽힌다
        match = re.search(r"\[OK\] new_rmse=([\d.]+) - production promoted: (\S+) v(\d+)", ok)
        self.assertEqual(match.groups(), ("540.04", "Tomato_Price_Predictor", "2"))

    def test_failed_logs_append_metrics(self):
        result, lines = self._run(promoted=False)
        self.assertEqual(result["wape"], 25.0)
        self.assertEqual(
            lines[-1],
            "[FAIL] new_rmse=540.04 - gate/regression failed, keep current Production (mae=412.3 wape=11.2% bias=-1.6%)",
        )


if __name__ == "__main__":
    unittest.main()
