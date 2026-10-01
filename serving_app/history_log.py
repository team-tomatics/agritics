"""
[소유: B 서빙 · 박유진]  이력 로그 (기획서 ② 핵심 기능 · ③ 3-2 응답 지연 · 에러율)

요청마다 logs/history.jsonl 에 JSON 한 줄을 남긴다. 보고서(GET /report)와
응답 지연(평균 > 500ms) · 에러율(> 5%) 지표의 원천이다.

한 줄 형식 (D 모니터링 · report_sidecar 가 이 키를 읽는다 — 바꾸면 슬랙 공지):
    {"ts": "2026-10-01T15:00:00+09:00", "path": "/predict", "status": 200,
     "latency_ms": 23.4, "model_version": "production", "input_last": {...}, "predicted": 3812.5}

TODO
- [ ] FastAPI 미들웨어로 /predict · /predict/batch-test 요청을 감싸 latency_ms 측정
- [ ] main.py 에 미들웨어 등록
- [ ] 5분 집계 함수 (평균 지연 · 에러율) — D 가 aiops.log 경고에 씀
"""
HISTORY_PATH = "logs/history.jsonl"
