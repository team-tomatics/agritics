import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from report_sidecar.observe import check_report

try:
    from serving_app import history_log
except ModuleNotFoundError:  # 로컬 최소 환경에는 FastAPI가 없을 수 있다.
    history_log = None


class ObserveTests(unittest.TestCase):
    @patch("report_sidecar.observe.logger.warning")
    def test_report_warnings_are_specific_and_do_not_leak_error(self, warning):
        messages = check_report(
            "보고서 본문",
            drift_detected=True,
            elapsed_s=30.1,
            source="template",
            error="RuntimeError: secret api key",
        )

        self.assertEqual(
            messages,
            [
                "[WARN] report generation failed",
                "[WARN] report slow",
                "[WARN] low-confidence text missing",
            ],
        )
        logged = " ".join(call.args[0] for call in warning.call_args_list)
        self.assertNotIn("secret api key", logged)

    @patch("report_sidecar.observe.logger.warning")
    def test_normal_report_has_no_warning(self, warning):
        self.assertEqual(check_report("가격 급변 감지 — 오늘 예측 신뢰도 낮음", False, 1.2), [])
        warning.assert_not_called()


class ApiExceptionLogTests(unittest.TestCase):
    @unittest.skipIf(history_log is None, "FastAPI 의존성이 없는 로컬 최소 환경")
    def test_middleware_logs_path_and_exception_type_then_reraises(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/predict"),
            method="GET",
            state=SimpleNamespace(),
        )

        async def fail(_request):
            raise ValueError("do not log this message")

        async def run():
            with patch.object(history_log, "_write") as write, patch.object(history_log.logger, "warning") as warning:
                with self.assertRaises(ValueError):
                    await history_log.history_middleware(request, fail)
                warning.assert_called_once()
                warning_args = warning.call_args.args
                self.assertIn("/predict", warning_args)
                self.assertIn("ValueError", warning_args)
                self.assertNotIn("do not log this message", str(warning_args))
                self.assertEqual(write.call_args.args[0]["status"], 500)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
