"""Production 승격 로그를 학습 경로와 재학습 경로에서 같은 형식으로 남긴다."""

import logging
import os


LOG_PATH = "logs/aiops.log"
logger = logging.getLogger("aiops")


def log_promotion(rmse: float, model_name: str, version: str) -> None:
    message = f"[OK] new_rmse={rmse:.2f} - production promoted: {model_name} v{version}"
    if logger.handlers:
        logger.info(message)
        return

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info(message)
    finally:
        logger.removeHandler(handler)
        handler.close()
