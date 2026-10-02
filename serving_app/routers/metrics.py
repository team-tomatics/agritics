"""
[소유: B 서빙 · 박유진]  GET /metrics — 운영 지표 (기획서 ③ 3-2 응답 지연 · 에러율)

작성자 : 박유진 (B 서빙 · Git)
버전   : 1.1.0
작성일 : 2026-10-01
변경 이력
  1.0.0  2026-10-01  최초 작성 — GET /metrics 5분 집계 · 경고 (#4)
  1.1.0  2026-10-01  경고 기준 P95 > 500ms · 5xx 에러율 > 1% (#32)

logs/history.jsonl 을 최근 N분(기본 5분) 집계해 돌려준다.
D 모니터링의 경고 판단 · 대시보드 · 발표 ⑤ API 명세에서 쓴다.

경고 기준 (기획서 3-2, #31)
- 응답 지연: P95 > 500ms — 평균은 느린 요청 1건을 묻는다. 실측 평소 약 20ms · Eager 첫 요청 150~200ms ·
  Lazy 첫 요청 약 5,300ms · 목표 1,000ms → 정상에선 안 울리고 1초 위반 전에 알린다
- 에러율: 5xx > 1% — ② 가용성 99% 의 오차 예산. 요청이 적으면 사실상 "5xx 1건이면 경고"
"""
from fastapi import APIRouter, Query

from serving_app import history_log

router = APIRouter()

P95_LATENCY_LIMIT_MS = 500
ERROR_RATE_LIMIT = 0.01


@router.get("/metrics")
def metrics(minutes: float = Query(5, gt=0, le=24 * 60, description="집계 구간(분)")):
    s = history_log.summarize(minutes)
    s["limits"] = {"p95_latency_ms": P95_LATENCY_LIMIT_MS, "error_rate": ERROR_RATE_LIMIT}
    s["alerts"] = [
        name
        for name, value, limit in [
            ("p95_latency_ms", s["p95_latency_ms"], P95_LATENCY_LIMIT_MS),
            ("error_rate", s["error_rate"], ERROR_RATE_LIMIT),
        ]
        if value is not None and value > limit
    ]
    return s
