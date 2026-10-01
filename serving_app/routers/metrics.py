"""
[소유: B 서빙 · 박유진]  GET /metrics — 운영 지표 (기획서 ③ 3-2 응답 지연 · 에러율)

logs/history.jsonl 을 최근 N분(기본 5분) 집계해 돌려준다.
D 모니터링의 경고 판단 · 대시보드 · 발표 ⑤ API 명세에서 쓴다.
"""
from fastapi import APIRouter, Query

from serving_app import history_log

router = APIRouter()

LATENCY_LIMIT_MS = 500  # 기획서 3-2: 평균 > 500ms 경고 (② 응답 1초 목표의 절반)
ERROR_RATE_LIMIT = 0.05  # 기획서 3-2: 에러율 > 5% 경고


@router.get("/metrics")
def metrics(minutes: float = Query(5, gt=0, le=24 * 60, description="집계 구간(분)")):
    s = history_log.summarize(minutes)
    s["limits"] = {"avg_latency_ms": LATENCY_LIMIT_MS, "error_rate": ERROR_RATE_LIMIT}
    s["alerts"] = [
        name
        for name, value, limit in [
            ("avg_latency_ms", s["avg_latency_ms"], LATENCY_LIMIT_MS),
            ("error_rate", s["error_rate"], ERROR_RATE_LIMIT),
        ]
        if value is not None and value > limit
    ]
    return s
