"""LLM 보고서 결과를 운영 경고로 변환한다 (기획서 ③ 3-2)."""

import logging
import os


LOG_PATH = "logs/aiops.log"
REPORT_TIMEOUT_S = 30.0
LOW_CONFIDENCE_TEXT = "신뢰도 낮음"
logger = logging.getLogger("aiops")


def _ensure_log_handler() -> None:
    """서빙 앱과 사이드카가 같은 로그 파일을 쓰도록 한 번만 연결한다."""
    if logger.handlers:
        return
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def check_report(
    report_text: str,
    drift_detected: bool,
    elapsed_s: float,
    source: str = "llm",
    error: str | None = None,
) -> list[str]:
    """보고서 이상을 감지하고 기록한 경고 문장을 반환한다.

    ``source``와 ``error``는 generator가 템플릿 폴백 사유를 전달할 때 사용한다.
    로그에는 API 키나 예외 전문을 남기지 않고, 운영자가 구분할 수 있는 유형만 남긴다.
    """
    _ensure_log_handler()
    warnings: list[str] = []

    if source == "template" or error or not report_text.strip():
        warnings.append("[WARN] report generation failed")
    if elapsed_s > REPORT_TIMEOUT_S:
        warnings.append("[WARN] report slow")
    if drift_detected and LOW_CONFIDENCE_TEXT not in report_text:
        warnings.append("[WARN] low-confidence text missing")

    for message in warnings:
        logger.warning(message)
    return warnings
