"""
이력 로그 (기획서 ② 핵심 기능 · ③ 3-2 응답 지연 · 에러율)

작성자: 박유진
버전: v1.3.0 (2026-10-01)
변경: #4 최초 작성 · #20 버전 표시 · #26 /predict 기준 집계 · #32 P95·1% 기준

요청마다 logs/history.jsonl 에 JSON 한 줄을 남긴다. 보고서(GET /report)와
응답 지연(P95 > 500ms) · 에러율(5xx > 1%) 지표의 원천이다.

한 줄 형식 (D 모니터링 · report_sidecar 가 이 키를 읽는다 — 바꾸면 B · D 둘 다 리뷰):
    {"ts": "2026-10-01T16:00:00+09:00", "method": "POST", "path": "/predict", "status": 200,
     "latency_ms": 23.4, "model_version": "production-v3", "predicted": 3812.5,
     "input_last": {"close": 3750.0, "volume": 120}}

- model_version · predicted · input_last 는 라우터가 request.state.history 에 넣어 준 경우만 있다
  (422 처럼 라우터까지 못 간 요청은 ts · method · path · status · latency_ms 만)
- 에러율 = 5xx 비율 (서버가 못 준 것). 4xx 는 client_error_rate 로 따로 센다
"""
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from fastapi import Request

logger = logging.getLogger("aiops")

HISTORY_PATH = "logs/history.jsonl"
TRACKED_PREFIXES = ("/predict", "/report")  # 정적 파일 · /docs · /health 는 기록하지 않는다

KST = timezone(timedelta(hours=9))
_lock = threading.Lock()


def _write(entry: dict, path: str = HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def history_middleware(request: Request, call_next):
    """main.py 에서 app.middleware("http")(history_middleware) 로 등록한다."""
    if not request.url.path.startswith(TRACKED_PREFIXES):
        return await call_next(request)

    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        logger.warning(
            "[WARN] api exception path=%s exception_type=%s",
            request.url.path,
            type(exc).__name__,
        )
        raise
    finally:
        # 라우터에서 예외가 나도 500 으로 한 줄 남기고, 예외는 그대로 올려 보낸다
        entry = {
            "ts": datetime.now(KST).isoformat(timespec="seconds"),
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        entry.update(getattr(request.state, "history", {}))
        _write(entry)


def read_entries(minutes: float | None = None, path: str = HISTORY_PATH, now: datetime | None = None) -> list[dict]:
    """최근 minutes 분의 기록 (None 이면 전부). 깨진 줄은 건너뛴다."""
    if not os.path.isfile(path):
        return []
    since = (now or datetime.now(KST)) - timedelta(minutes=minutes) if minutes is not None else None
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if since is None or datetime.fromisoformat(e["ts"]) >= since:
                    entries.append(e)
            except (ValueError, KeyError):
                continue
    return entries


METRIC_PATH = "/predict"  # 기획서 3-2 지연 · 에러율은 점주가 쓰는 예측 API 기준


def summarize(minutes: float = 5, path: str = HISTORY_PATH, now: datetime | None = None,
              target: str = METRIC_PATH) -> dict:
    """
    최근 minutes 분 집계 (기획서 3-2: 5분 집계). D 의 경고 판단과 GET /metrics 가 쓴다.
    - count · 지연 · 에러율은 target(기본 POST /predict) 만으로 계산한다.
      /predict/batch-test 는 드리프트 시뮬레이션이라 안에서 재학습(약 8초)이 돌아 섞으면 가짜 지연 경고가 난다 (#25)
    - total_count · by_path 로 전체 요청은 따로 본다
    - 요청이 없으면 지연 · 에러율은 None (0 으로 두면 "정상"으로 오해한다)
    """
    all_entries = read_entries(minutes, path, now)
    entries = [e for e in all_entries if e.get("path") == target]
    by_path: dict[str, int] = {}
    for e in all_entries:
        by_path[e.get("path", "?")] = by_path.get(e.get("path", "?"), 0) + 1
    n = len(entries)
    latencies = sorted(e["latency_ms"] for e in entries)
    return {
        "window_min": minutes,
        "target": target,
        "count": n,
        "avg_latency_ms": round(sum(latencies) / n, 1) if n else None,
        "p95_latency_ms": latencies[math.ceil(n * 0.95) - 1] if n else None,  # nearest-rank
        "error_rate": round(sum(e["status"] >= 500 for e in entries) / n, 3) if n else None,
        "client_error_rate": round(sum(400 <= e["status"] < 500 for e in entries) / n, 3) if n else None,
        "total_count": len(all_entries),
        "by_path": by_path,
    }
