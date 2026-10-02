import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_app.monitoring import promotion_log


class PromotionLogTests(unittest.TestCase):
    def test_writes_initial_promotion_when_app_logger_is_not_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiops.log"
            isolated_logger = logging.Logger("test-aiops")
            with (
                patch.object(promotion_log, "LOG_PATH", str(path)),
                patch.object(promotion_log, "logger", isolated_logger),
            ):
                promotion_log.log_promotion(540.04, "Tomato_Price_Predictor", "1")

            content = path.read_text(encoding="utf-8")
            self.assertIn(
                "[INFO] [OK] new_rmse=540.04 - production promoted: Tomato_Price_Predictor v1",
                content,
            )


if __name__ == "__main__":
    unittest.main()
