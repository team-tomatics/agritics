# ⑤ API 명세 — 농산품 시세 예보 AIOps

> 소유: B 서빙 · 박유진 (`/data/*` 는 A 심준용, `/logs` 는 D 민영은 파일 — 동작이 바뀌면 소유자가 이 문서도 고친다)
> 기준: 입력 25거래일 · 판정 윈도우 15건 (기획서 3-6). Swagger: `http://localhost:8077/docs` (컨테이너 `8099`)
> 예시 값은 토마토 데이터(`data/tomato_prices.csv` 903거래일 · 반입량 포함)로 학습한 Production 모델(RMSE 540.04)에 실제로 보낸 응답이다 (10/2).

## 한눈에

| 메서드 · 경로 | 하는 일 | 성공 | 실패 | 기획서 |
|---|---|---|---|---|
| `POST /predict` | 최근 25거래일 → 다음날 도매가격 | 200 | **422** 입력 길이 · 값 범위 | ② 핵심 기능 · 3-1 입력 스키마 검증 |
| `POST /predict/batch-test` | 가격 40개 → 슬라이딩 예측 15건 → 드리프트 판정 · 재학습 | 200 | 422 | 3-2 예측 오차 · 3-3 드리프트 대응 |
| `GET /health` | 서버 상태 · 모델 로드 여부 · 로딩 모드 | 200 | — | ② 가용성 |
| `GET /metrics` | 최근 N분 지연 · 에러율 · 경고 (이력 로그 집계) | 200 | 422 `minutes` 범위 | 3-2 응답 지연 · 에러율 |
| `GET /report` | 최신 일일 보고서 (LLM / 템플릿) | 200 | **404** 아직 없음 | ② GET /report · 3-2 LLM 보고서 |
| `POST /data/upload` | 시세 CSV 업로드 | 200 | **400** 열 · 행 수 · 날짜 형식 · 중복 · 순서 · 값 범위 · 인코딩 | 3-1 업로드 검증 |
| `GET /data/status` | 최신 업로드 CSV 요약 | 200 | — | — |
| `GET /logs` | `logs/` 파일 목록 | 200 | — | 3-3 aiops.log |
| `GET /logs/{filename}` | 로그 파일 내용 | 200 | 404 없는 파일 · 경로가 섞인 이름 | 3-3 |

공통: 요청 · 응답은 JSON (업로드만 `multipart/form-data`). 422 는 FastAPI(Pydantic) 표준 형식 `{"detail": [...]}`.
`/predict` · `/predict/batch-test` · `/report` 는 요청마다 `logs/history.jsonl` 에 한 줄 기록된다 (아래 "이력 로그").

---

## POST /predict

최근 **25거래일** (가격, 반입량) → 다음날 도매가격.

요청
```json
{"sequence": [{"close": 4599.0, "volume": 158977}, {"close": 3817.0, "volume": 210513}, ... 25개]}
```
| 필드 | 타입 | 규칙 |
|---|---|---|
| `sequence` | 배열 | 정확히 25개, 오래된 날 → 최근 날 |
| `sequence[].close` | float | > 0 (도매가격 원/kg) |
| `sequence[].volume` | int | ≥ 0 (가락시장 반입량 kg) |

응답 200
```json
{"predicted_close": 3925.86, "model_version": "production-v1"}
```
(입력: 실제 2026-08-29 ~ 09-29 25거래일. 다음날 09-30 실제 가격은 3,042원/kg — 예측이 틀린 날의 예시이기도 하다)
- `model_version`: MLflow 로 불러오면 `production-vN` (재학습 승격 후 번호가 바뀜 · PR #20), 로컬 파일이면 `v1-local`

에러 422 — 24개 · 26개 · 20개, `close = 0`, `volume = -1` 모두 실제 호출로 422 확인 (10/1, 응답 원문 · `input` 생략)
```json
{"detail": [{"type": "too_short", "loc": ["body", "sequence"],
             "msg": "List should have at least 25 items after validation, not 24",
             "ctx": {"field_type": "List", "min_length": 25, "actual_length": 24}}]}
{"detail": [{"type": "greater_than", "loc": ["body", "sequence", 24, "close"],
             "msg": "Input should be greater than 0", "input": 0.0, "ctx": {"gt": 0.0}}]}
{"detail": [{"type": "greater_than_equal", "loc": ["body", "sequence", 24, "volume"],
             "msg": "Input should be greater than or equal to 0", "input": -1, "ctx": {"ge": 0}}]}
```

## POST /predict/batch-test

드리프트 시뮬레이션용. 가격 `SEQ_LEN + N` 개를 보내면 25칸 창을 한 칸씩 밀며 N 번 예측하고, 최근 15건 RMSE 가 612원/kg 을 넘으면 드리프트로 판정한다. 반입량은 최근 업로드 CSV 의 중앙값(토마토 136,449kg, #49).

요청 (보통 40개 = 25 + 15, `scripts/simulate_drift.py`)
```json
{"prices": [2833.0, 2850.0, 2920.0, ... 40개]}
```
응답 200 — 평온 구간 (실제 2025-03-28 ~ 05-13 40거래일)
```json
{"predictions": [2539.5, 2492.9, 2439.9, ... 15개],
 "drift_check": {"status": "ok"}}
```
드리프트면 그 자리에서 재학습(최근 25 + 25 = 50행 fine-tuning) → 게이트(612원/kg) · 회귀 테스트 → 통과하면 승격. 아래는 실제 2026-08-13 ~ 09-30 급등 구간 — 재학습한 모델도 게이트를 못 넘어 **기존 Production 유지**
```json
{"predictions": [4230.1, 4628.2, 4998.2, ... 15개],
 "drift_check": {"status": "retrain_triggered", "promoted": false, "rmse": 1258.41}}
```
통과하면 `"promoted": true` 와 새 `"rmse"` · 다음 `/predict` 의 `model_version` 이 `production-v2` 로 바뀐다
(10/2 토마토 Production 모델 실측)
- 승격되면 서버 모델 캐시를 비워 **다음 `/predict` 부터 새 버전**으로 응답한다 (기획서 3-4)
- 재학습은 요청 안에서 동기로 돈다 (약 8초) — 기획서 3-4 #8 한계

## GET /health

```json
{"status": "ok", "model_loaded": true, "loading_mode": "eager"}
```
- Lazy 면 첫 `/predict` 전까지 `model_loaded: false`

## GET /metrics?minutes=5

`logs/history.jsonl` 최근 N분 집계 (기본 5분, 0 < N ≤ 1440 — `minutes=0` 은 422).
지연 · 에러율은 **`POST /predict` 만으로** 계산한다 — batch-test 는 안에서 재학습(약 8초)이 돌아 섞으면 가짜 지연 경고가 난다 (#25).
```json
{"window_min": 5.0, "target": "/predict", "count": 3, "avg_latency_ms": 79.6, "p95_latency_ms": 194.7,
 "error_rate": 0.0, "client_error_rate": 0.0,
 "total_count": 3, "by_path": {"/predict": 3},
 "limits": {"p95_latency_ms": 500, "error_rate": 0.01}, "alerts": []}
```
(main 10/1 실측 — Eager 서버에 `/predict` 3회. 첫 요청 약 195ms 가 P95 로 잡혀도 500ms 아래라 경고 없음)

batch-test 는 `count` 에서 빠진다: `/predict` 2회 + batch-test 2회(그중 재학습 **7,554ms**)에서 `count 2 · avg 129.0 · p95 200.7 · alerts []` — 고치기 전(#25)엔 같은 시나리오에서 평균 2,054ms · 지연 경고.

| 필드 | 뜻 |
|---|---|
| `count` · 지연 · 에러율 | `target`(`/predict`) 요청만 |
| `total_count` · `by_path` | 기록된 전체 요청 (batch-test · report 포함) |
| `error_rate` | 5xx 비율 (서버가 응답을 못 준 것) |
| `client_error_rate` | 4xx 비율 (422 · 404 — 잘못된 요청) |
| `alerts` | 기획서 3-2 임계값(**P95 > 500ms · 5xx 에러율 > 1%**)을 넘은 지표 이름 (#31) |
| 요청 0건 | 지연 · 에러율 `null` (0 으로 두면 "정상"으로 오해) |

## GET /report

LLM 사이드카(`python -m report_sidecar.generator --at 06:00`)가 **매일 06:00 KST**(토마토 02:00 경매 기준) · 사이드카 시작 시 만든 최신 보고서. 서빙 프로세스는 LLM 을 부르지 않는다 (PR #16).
```json
{"date": "2026-10-02", "source": "template", "error": "RuntimeError: OPENAI_API_KEY 없음", "elapsed_s": 0.0,
 "item_name": "토마토", "model_version": "production-v1", "drift_detected": false,
 "markdown": "## 토마토 시세 일일 보고서\n- 내일 예측가: **3,926원/kg** (오늘 3,350원/kg, +17.2%)\n- 예측 모델: production-v1\n..."}
```
(위 `/predict` 1건 뒤 키 없이 생성한 실제 응답. LLM 이 성공하면 `source: "llm"`, `error: null`, `markdown` 은 `report_sidecar/prompts/report.md` 규칙으로 쓴 LLM 문장. 잘못된 키면 `error: "HTTPError: 401 ..."`)
- `source: "template"` — LLM 실패(키 없음 · 30초 초과 · 오류) 시 같은 숫자로 만든 정해진 양식, `error` 에 사유
- `drift_detected: true` 면 본문 첫 줄이 "가격 급변 감지 — 오늘 예측 신뢰도 낮음"
- 404 `{"detail": "아직 생성된 보고서가 없습니다 (python -m report_sidecar.generator)"}`

## POST /data/upload  ·  GET /data/status   (A 심준용)

`multipart/form-data` 의 `file` — UTF-8 CSV, 열 `Date, Close, Volume`, **최소 40행** (입력 25 + 윈도우 15). 저장 전에 `data/validation.py` 로 검증한다 (#18)
```json
{"filename": "tomato_1790839653.csv", "rows": 903}
```
400 — 아래는 업로드 검증 오류 예시다. 40행 정상 CSV는 200과 `{"filename": "tomato_1790844594.csv", "rows": 40}`을 반환한다.

| 보낸 CSV | `detail` |
|---|---|
| 39행 | `최소 40행 이상의 데이터가 필요합니다. 현재 39행입니다.` |
| `Volume` 열 없음 | `CSV에 ['Volume'] 컬럼이 없습니다. 필수 컬럼: ['Close', 'Date', 'Volume']` |
| 날짜 `2026/01/06` | `7행의 Date는 YYYY-MM-DD 형식이어야 합니다: '2026/01/06'` |
| 같은 날짜 두 번 | `중복 날짜가 있습니다: 2026-01-06` |
| 날짜 순서 뒤바뀜 | `Date는 오름차순이어야 합니다: 2026-01-08 다음에 2026-01-07` |
| `Close` 0 | `5행의 Close는 0보다 커야 합니다.` |
| `Volume` −1 | `5행의 Volume은 0 이상이어야 합니다.` |
| `Close` 빈 칸 | `5행의 Close 값이 비어 있습니다.` |
| UTF-8 아님 | `UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.` (코드의 문구 — 이번 테스트에선 안 보냄) |

행 번호는 헤더를 1행으로 센 CSV 줄 번호다.

`GET /data/status`
```json
{"exists": true, "filename": "tomato_1790839653.csv", "rows": 903,
 "start_date": "2023-10-02", "end_date": "2026-09-30",
 "min_close": 1178.0, "max_close": 14717.0, "price_unit": "원/kg",
 "min_volume": 43250.0, "max_volume": 363668.0,
 "avg_volume": 144429.16611295682, "volume_unit": "kg"}
```
업로드가 없으면 `{"exists": false}`. 학습 · 재학습은 항상 **가장 최근 업로드** 파일을 쓴다.

## GET /logs  ·  GET /logs/{filename}   (D 민영은)

```json
[{"name": "aiops.log", "size": 0}, {"name": "history.jsonl", "size": 324}]
```
```json
{"name": "aiops.log", "content": "2026-10-01 17:01:04,... [WARNING] [WARN] drift detected - triggering retrain\n..."}
```
`aiops.log` 순서: `[WARN] drift detected - triggering retrain` → `[INFO] retrain triggered (window=last_25_days)` → 승격하면 `[OK] new_rmse=... - production promoted: Tomato_Price_Predictor vN`, 게이트 · 회귀 테스트에서 떨어지면 `[FAIL] new_rmse=... - gate/regression failed, keep current Production`.
컨테이너 빌드 중 최초 Production 등록도 같은 `[OK]` 형식으로 기록되므로 첫 기동부터 대시보드에 v1 RMSE와 승격 이력이 표시된다.

---

## 이력 로그 한 줄 (`logs/history.jsonl`)

```json
{"ts": "2026-10-02T09:43:19+09:00", "method": "POST", "path": "/predict", "status": 200,
 "latency_ms": 153.0, "model_version": "production-v1", "predicted": 3925.86,
 "input_last": {"close": 3350.0, "volume": 159212}}
```
- 422 처럼 라우터까지 못 간 요청은 `ts · method · path · status · latency_ms` 만
- batch-test 는 `n_predictions` · `drift_status` 추가
- 이 키를 D(경고) · 보고서 사이드카가 읽는다 — 바꾸면 B · D 둘 다 리뷰

---

## 응답 시간 — Lazy vs Eager (기획서 ② "첫 요청 포함 1초 이내")

`python scripts/measure_latency.py` — 모드마다 서버를 새 프로세스로 띄워 3회 측정 (10/1, MODEL_SOURCE=mlflow, 입력 25일, MacBook CPU)

| 모드 | 기동 | 기동 직후 model_loaded | 첫 요청 | 두 번째 요청 | 1초 목표 |
|---|---|---|---|---|---|
| Lazy | 0.68초 | false | **5.270초** (4.19 ~ 7.27) | 0.022초 | ❌ 첫 요청 위반 |
| Eager | 3.73초 | true | **0.155초** (0.151 ~ 0.160) | 0.022초 | ✅ |

- Lazy 는 기동이 빠른 대신 **첫 손님이 모델 로딩 5초를 떠안는다** — 아침 06–10시 첫 발주 요청이 시간 초과될 수 있다 (기획서 ① "응답 지연")
- Eager 는 기동이 3초 늘지만 요청은 처음부터 0.15초 → **컨테이너는 Eager** (`Dockerfile` `LOADING_MODE=eager`)
- 이력 로그에도 같은 차이가 그대로 남는다: Lazy 첫 요청 `latency_ms: 5470.7` → 이후 `18.8` (PR #4)
- Lazy #1(7.27초)이 특히 느린 건 디스크 캐시가 비어 있던 첫 실행이라서다
