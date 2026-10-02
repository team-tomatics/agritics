import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import serving_app
from serving_app.monitoring import retrain_trigger


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
                    result = retrain_trigger.check_and_trigger(records)

                self.assertEqual(result["promoted"], promoted)
                self.assertEqual(records, [] if promoted else [{"predicted": 1, "actual": 2}] * 15)
                self.assertEqual(loader._model_cache, None if promoted else "old-model")


if __name__ == "__main__":
    unittest.main()
