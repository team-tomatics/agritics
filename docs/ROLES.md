# 역할 분담 · 시작 가이드

> 이 문서 하나 읽고 바로 개발 시작할 수 있게 썼습니다. 규칙 상세는 [CONTRIBUTING.md](../CONTRIBUTING.md), 근거는 [기획서.md](기획서.md).
>
> **개발 마감: 10/2(금) 점심** · **발표: 10/2 14:00–16:30, 조별 20분 이내**
> 결과물은 하나의 파이프라인입니다 (가이드 p.5). 예측은 LSTM 1개(시계열)만, 생성형 AI 는 예측에 끼지 않고 이력을 읽어 보고서만 만듭니다.
>
> **서비스 범위 = 농수산물 시세 예보, 구현 · 발표 품목 = 토마토.** 품목은 사계절용 / 계절용 두 그룹으로 나누고 설정값(`config/items.yaml`)으로 바꿉니다 — 아래 0-1.

---

## 0. 시작 5분 (전원)

```bash
git clone https://github.com/team-tomatics/agritics.git
cd agritics
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env

# 지금은 HAIC 샘플로 파이프라인이 끝까지 도는지 확인 (토마토 CSV 는 A 가 머지한 뒤)
.venv/bin/uvicorn serving_app.main:app --port 8077          # 터미널 1 → localhost:8077 에서 data/sample_haic_prices.csv 업로드
.venv/bin/python scripts/train_baseline_v1.py                # 터미널 2 → scaler.pkl + haic_v1.keras
.venv/bin/python serving_app/train_and_register.py           # MLflow 등록 + [GATE PASSED]
# 터미널 1 재시작: MODEL_SOURCE=mlflow LOADING_MODE=eager .venv/bin/uvicorn serving_app.main:app --port 8077
.venv/bin/python scripts/simulate_drift.py                   # [WARN] → 재학습 → [OK]

# 컨테이너 (빌드 때 학습까지 끝남, 8099)
docker compose -f serving_app/docker-compose.yml up --build
```

그 다음:
1. GitHub 이슈 생성 (아래 **5. 초기 이슈** 표에서 자기 것 골라 템플릿으로) → 번호 확인
2. `git switch -c feat/<번호>-<slug>`
3. 자기 파일만 고치고 → PR

베이스 코드는 교수님 최종 배포본(빈칸 11곳 채움 + 재학습 후 캐시 비우기 포함)입니다. 원본 설명은 [skeleton_README.md](skeleton_README.md).

---

## 0-1. 서비스 범위와 품목 그룹 (교수님 피드백 10/1)

| 그룹 | 기준 | 품목 | 이번 구현 |
|---|---|---|---|
| **사계절용** | 연중 출하 — 25거래일(가락시장 한 달) 연속 시퀀스가 끊기지 않는다 | **토마토** · 토마토 완숙 · 방울토마토 | **토마토 1개 모델** (학습 · 서빙 · 드리프트 · 재학습 · 보고서 전부) |
| **계절용** | 출하 철에만 거래 — 철마다 새로 시작하는 짧은 시계열 | 토마토 대저 · 딸기 | 설계 · 발표로만 |

**기획은 범용, 구현은 토마토만**
- 기획서 · 발표: 농수산물 전체 + 사계절용 / 계절용 그룹핑 + 품목 확장 경로까지 "된다"고 설계로 보여 준다
- 코드: 토마토 하나로 업로드 → 학습 · 게이트 → 서빙 → 드리프트 → 재학습 → 보고서가 끝까지 도는 데만 집중한다. 품목 선택 · 그룹별 학습 같은 확장 코드는 만들지 않는다
- `config/items.yaml` 은 코드가 읽지 않는 **설계표**다 (발표 자료 · 확장 규칙 근거)
- 기존 스켈레톤의 HAIC 값은 3장 공유 상수 PR 로 토마토 값으로 한 번에 바꾼다

**발표에서 그룹을 보여 주는 방법**: `config/items.yaml` 표 → 토마토가 사계절용 대표로 끝까지 도는 데모 → "같은 설정으로 완숙 · 방울토마토는 바로, 계절용은 철 단위 시퀀스로" 확장 경로 한 장.

---

## 1. 누가 무엇을

| | 이름 | 역할 | 소유 파일 | 내 AIOps 신호 | 기획서 · 발표 |
|---|---|---|---|---|---|
| **A** | 심준용 | 데이터 오너 | `data/*` · `routers/data.py` · `scripts/train_baseline_v1.py` · `scripts/fetch_*.py` · `config/items.yaml` · `report_sidecar/prompts/` | 입력 데이터 드리프트 — 가격 · 반입량 분포 (PSI · KS) | ① Pain Point · 업로드 · `/data/status` · 분포 비교 · 보고서 예시 |
| **B** | 박유진 | 서빙 오너 (+ Git) | `schemas.py` · `model_loader.py` · `main.py` · `history_log.py` · `routers/predict.py` `health.py` `report.py` · `report_sidecar/generator.py` | 응답 지연 · 에러율 (지연 > 500ms · 에러율 > 5%) | ② 운영 목표 · ⑤ API 명세(/report 포함) · 지연 측정표 |
| **C** | 황재원 | MLOps 오너 | `train_and_register.py` · `lstm_model.py` · `Dockerfile` · `docker-compose.yml` · `requirements.txt` · `report_sidecar/Dockerfile` | 모델 버전 관리 — 전환 · 실패 시 기존 유지 · 재학습 뒤 캐시 비우기 | ③ 게이트 절 · ④ 아키텍처 · MLflow 버전 전 · 후 · 컨테이너 기동 |
| **D** | 민영은 | 모니터링 오너 | `monitoring/*` · `scripts/simulate_drift.py` · `static/index.html` · `routers/logs.py` · `report_sidecar/observe.py` | 예측 드리프트(21건 윈도우) + 생성형 출력 이상 | ③ 드리프트 대응 절 · 드리프트 → [WARN] → 재학습 → 보고서 라이브 데모 |

공용 (바꾸기 전에 슬랙): `.github/` · `.env.example` · `docs/` · `README.md` · `scripts/check_constants.py`

**하지 않는 것** (발표에서는 설계로만): 그룹 · 품목별 모델 여러 개 실제 학습 — 구현은 사계절용 토마토 1개 · 계절용 그룹 파이프라인 구현 · LLM 이 예측값을 고치거나 RAG 를 붙이는 것.

---

## 2. 파트별 — 할 일

### A 심준용 — 데이터

| 할 일 | 파일 | 근거 |
|---|---|---|
| KAMIS 가락시장 토마토(등급 "상") 3년치 → `Date,Close,Volume` CSV. 휴장일 빼고 거래일로 이어 붙이기 | `scripts/fetch_kamis.py` → `data/tomato_prices.csv` | 기획서 3-5 |
| 반입량 — 서울시농수산식품공사 OpenAPI 승인 대기. 늦으면 KAMIS 소매가격을 보조 피처로 (열 이름은 그대로 Volume) | 같은 스크립트 | 3-5 |
| 단순 예측(내일 = 오늘) RMSE 재확인 → **임계값 제안** (기획서: 612원/kg 미만) → 슬랙 공지 | baseline 결과 | ② 품질 · 3-1 |
| 품목 그룹 설계표 확정 (기획 · 발표용) — 사계절용 · 계절용 기준(거래일 수 · 공백 기간), 후보 품목 · `status` 갱신 | `config/items.yaml` (기획서 표 옮겨 둠) | 「품목 그룹」 |
| 발표용 그룹 근거 — 품목별 3년 거래일 수 · 최대 공백 표 (예: 토마토 908일 · 대저 288일) | 표 · 캡처 | 「품목 그룹」 |
| 보고서 프롬프트 — 점주 · 본사가 읽을 문장, 드리프트 시 "신뢰도 낮음" 규칙 | `report_sidecar/prompts/report.md` (뼈대 있음) | ② GET /report |
| (선택) 입력 분포 PSI > 0.2 | 새 파일 | 3-2 |

**먼저 할 것**: 토마토 CSV PR — **C(Dockerfile 시드) · D(기준 가격) · 전원(임계값)이 이걸 기다립니다.**

### B 박유진 — 서빙

| 할 일 | 파일 | 근거 |
|---|---|---|
| 이력 로그 미들웨어 — 요청마다 `logs/history.jsonl` 한 줄 (ts · path · status · latency_ms · model_version · predicted) | `serving_app/history_log.py` (뼈대 있음) · `main.py` 등록 | ② 이력 로그 |
| 5분 집계 (평균 지연 · 에러율) → D 가 `aiops.log` 경고에 사용 | `history_log.py` | 3-2 |
| `GET /report` — 사이드카가 만든 최신 보고서 반환 (지금은 501) | `routers/report.py` | ② |
| LLM 호출 코드 — 로그 읽기 → 프롬프트 → LLM → `outputs/reports/` | `report_sidecar/generator.py` | ② |
| (수준 2 결정 시) 스키마 필드 이름 `close` → `price` 등, Swagger 설명 원/kg | `schemas.py` · `routers/predict.py` · `model_loader.py` | 6장 매핑 |
| Lazy/Eager 지연 측정표 (⑤ 와 ② 응답 1초 근거) | — | ② 응답 시간 |

**먼저 할 것**: `history.jsonl` 한 줄 형식 확정 PR — D(LLM 관측) · 사이드카가 이 키를 읽습니다.

### C 황재원 — MLOps

| 할 일 | 파일 | 근거 |
|---|---|---|
| 모델 이름 `HAIC_Predictor` → `Tomato_Price_Predictor` — 규칙 `<품목>_Price_Predictor`, `config/items.yaml` 의 `model_name` 과 같게 (공유 상수 — 3장) | `train_and_register.py` · `model_loader.py`(B 리뷰) · `retrain_trigger.py` 로그 문구(D 리뷰) | 「품목 그룹」 |
| 게이트 임계값 원/kg (공유 상수) | `train_and_register.py` | 3-1 |
| Dockerfile 시드 CSV 를 토마토로 교체 | `Dockerfile` | 6장 주의 |
| 사이드카 컨테이너를 compose 에 추가 · API 키는 `env_file: .env` · `logs/` `outputs/` 볼륨 공유 | `docker-compose.yml` · `report_sidecar/Dockerfile` | 7장 |
| 승격 시 이전 Production → Archived (롤백 대상 명확) | `train_and_register.py` | 3-4 |
| (선택) 성능 회귀 테스트 — 새 RMSE ≤ 현재 Production RMSE | `train_and_register.py` | 3-1 |

**먼저 할 것**: 컨테이너가 HAIC 그대로 한 번 뜨는지 확인 → 모델 이름 PR.

### D 민영은 — 모니터링

| 할 일 | 파일 | 근거 |
|---|---|---|
| 드리프트 임계값 원/kg (공유 상수 — 게이트와 같은 값) | `drift_detector.py` · `index.html` | 3-2 |
| 폭염 시나리오 — 기준 가격 165 → 토마토 평균(3,727원/kg), 급등형 배치 | `simulate_drift.py` · `index.html` 복제 상수 | 3-3 |
| 승격 시 판정 윈도우 초기화 (팀 결정) | `retrain_trigger.py` | 3-4 #7 |
| 지연 > 500ms · 에러율 > 5% → `aiops.log` 경고 (B 의 집계 함수 사용) | `monitoring/` | 3-2 |
| LLM 관측 — 생성 실패 · 30초 초과 · 드리프트인데 "신뢰도 낮음" 누락 → 경고 | `report_sidecar/observe.py` (뼈대 있음) | 3-2 |
| 대시보드 문구 HAIC → "농수산물 시세 예보 · 토마토" (서비스명 + 현재 품목) | `index.html` | 6장 · 「품목 그룹」 |
| (선택) 재학습을 BackgroundTasks 로 분리 또는 한계로 명시 | `retrain_trigger.py` · `routers/predict.py`(B 리뷰) | 3-4 #8 |

**먼저 할 것**: 시나리오 함수에 기준 가격 · 변동성을 인자로 빼 두기 → A 의 CSV 가 오면 값만 바꾸면 되게.

---

## 3. 공유 상수 — 같이 바꿔야 하는 곳

**한 사람이 정해 공지하고, 표의 위치를 한 PR 에서 전부 바꿉니다.** `python3 scripts/check_constants.py` 가 CI 에서 일치 여부를 검사합니다 (✔ 표시).

| 상수 | 지금 (HAIC) | 구현 품목 (토마토) | 정하는 사람 | 위치 |
|---|---|---|---|---|
| 입력 길이 `SEQ_LEN` ✔ | 20 | **25** ✅ #5 (기획서 3-6) | 전원 (기획서) | `data/features.py:16` · `index.html` `const SEQ_LEN` |
| 판정 윈도우 `WINDOW_SIZE` ✔ | 21 | **15** ✅ #5 | 전원 (기획서) | `monitoring/drift_detector.py:24` · `index.html` `const WINDOW_SIZE` · `simulate_drift.py` `BATCH_N = 40` (= 25 + 15) · `routers/predict.py` `[-WINDOW_SIZE:]` |
| 재학습 정답 일수 `RETRAIN_DAYS` | 21 (숫자) | **25** → 50행 ✅ #5 | 전원 (기획서) | `monitoring/retrain_trigger.py:31` (판정 윈도우와 별개) |
| RMSE 임계값 ✔ | 4.00 ($) | **팀 결정** (< 612원/kg) | A 제안 → 전원 합의 | `train_and_register.py:37` `RMSE_GATE` · `drift_detector.py:22` `RMSE_THRESHOLD` · `index.html` `const RMSE_THRESHOLD` |
| 모델 이름 ✔ | `HAIC_Predictor` | `Tomato_Price_Predictor` (= `items.yaml` `model_name`) | C | `train_and_register.py:38` · `model_loader.py:40` `MLFLOW_MODEL_URI` · `retrain_trigger.py:90` 로그 문구 |
| 시나리오 기준 가격 | 165.0 | 3,727 (3년 평균) | A 데이터로 D 가 | `simulate_drift.py:75,80` `base=` · `index.html` `DEFAULT_BASE_PRICE` |
| 변동성 | 0.012 / ×3 | **팀 결정** (토마토 일변동) | A 데이터로 D 가 | `simulate_drift.py:66-67` · `index.html` `NORMAL_SIGMA` `DRIFT_SIGMA` |
| 시드 CSV | `sample_haic_prices.csv` | `tomato_prices.csv` (= `items.yaml` `data`) ✅ 커밋됨, Volume 열 대기 | A 파일 · C 교체 | `Dockerfile:28` · `simulate_drift.py:42` `SAMPLE_CSV` · 안내 문구(`storage.py` · `index.html`) |
| 열 · 필드 이름 | `Close` / `Volume` | **팀 결정** — 수준 1(그대로, 의미만 재정의) / 수준 2(이름까지) | 전원 | CSV · `features.py` · `schemas.py` · `routers/data.py` · `model_loader.py` · `routers/predict.py` |

> 대시보드(`index.html`)는 서버 상수를 복사해 씁니다. 서버만 고치면 대시보드 버튼은 예전 값으로 돕니다.
> 드리프트 → 재학습은 MLflow 에 Production 모델이 있어야 돕니다 — `train_and_register.py` 를 먼저 실행.

---

## 4. 의존 그래프

```
A 토마토 CSV ──┬──▶ C Dockerfile 시드 교체 ──▶ 컨테이너 데모
               ├──▶ A 임계값 제안 ──▶ [CHORE] 공유 상수 PR (C · D 파일) ──▶ 게이트 · 드리프트 데모
               └──▶ D 기준 가격 · 변동성 ──▶ 폭염 시나리오

B history.jsonl 형식 ──┬──▶ B 5분 집계 ──▶ D 지연 · 에러율 경고
                       └──▶ B generator.py ◀── A prompts/report.md
                                  │
                                  ├──▶ C 사이드카 compose ──▶ B GET /report
                                  └──▶ D observe.py (LLM 관측)

C 모델 이름 변경 ──▶ (B model_loader 리뷰 · D retrain_trigger 문구 리뷰)
```

**병목은 A 의 CSV 와 임계값**입니다. 그동안 B · C · D 는 HAIC 샘플로 각자 기능을 먼저 만들고, 값만 나중에 바꿉니다.

---

## 5. 초기 이슈

| # | 제목 | 파트 | 템플릿 | 막고 있는 것 |
|---|---|---|---|---|
| 1 | [CHORE] 협업 템플릿 · CI · 문서 | 공용 | chore | — (이 저장소 초기 커밋) |
| 2 | [FEAT] KAMIS 토마토 3년치 CSV | A | feature | C 시드 · D 시나리오 · 임계값 |
| 3 | [CHORE] 공유 상수 — RMSE 임계값 원/kg 확정 | A → 전원 | chore | 게이트 · 드리프트 데모 |
| 4 | [FEAT] 이력 로그 미들웨어 + 5분 집계 | B | feature | D 지연 · 에러율 · 사이드카 |
| 5 | [FEAT] 보고서 생성기 + GET /report | B | feature | 사이드카 데모 |
| 6 | [FEAT] 보고서 프롬프트 · 품목 그룹표 | A | feature | 5 |
| 7 | [CHORE] 모델 이름 Tomato_Price_Predictor | C | chore | — |
| 8 | [FEAT] Dockerfile 시드 교체 + 사이드카 compose | C | feature | 컨테이너 데모 |
| 9 | [FEAT] 이전 Production Archived + (선택) 회귀 테스트 | C | feature | — |
| 10 | [FEAT] 폭염 시나리오 · 토마토 기준 가격 | D | feature | 라이브 데모 |
| 11 | [FEAT] 승격 시 판정 윈도우 초기화 | D | feature | — |
| 12 | [FEAT] LLM 관측 · 지연 · 에러율 경고 | D | feature | — |
| 13 | [CHORE] 발표 ④⑤⑥ 캡처 · API 명세 | 전원 | chore | 발표 |

번호는 실제 생성 순서에 따라 달라질 수 있습니다.

---

## 5-1. 선택 과제 — 품목 확장 (맨 마지막)

> ⚠️ **완전 확장이라 우선순위 꼴찌.** 토마토 파이프라인(데이터 → 게이트 → 서빙 → 드리프트 → 재학습 → 보고서)과 발표 ④⑤⑥ 이 **전부 끝난 뒤**에만 손댄다.
> B · C 파일을 여러 개 건드리므로, 시간이 없으면 하지 않고 발표에서 설계로만 말한다.

| 제목 | 내용 | 건드리는 파일 |
|---|---|---|
| [FEAT] (선택) `ITEM` 환경변수로 품목 전환 | `ITEM=tomato` → 모델 이름 `<품목>_Price_Predictor` · 스케일러 `scaler_<품목>.pkl` · 데이터 `data/<품목>_prices.csv` 를 품목별로 분리 → `ITEM=cherry_tomato` 로 같은 파이프라인 데모 | `train_and_register.py` · `train_baseline_v1.py` · `model_loader.py` · `retrain_trigger.py` · `Dockerfile` · `index.html` (B · C · D 리뷰) |

- 사계절용 품목만 대상. 계절용(대저 · 딸기)은 휴장 공백을 철 단위로 끊는 처리가 따로 필요해서 이 과제로도 안 된다 → 설계로만
- 지금 코드로도 CSV 를 바꿔 다시 학습하면 다른 사계절용 품목이 **한 번에 하나씩**은 돈다. 이 과제는 "동시에 · 이름으로 구분해서"를 위한 것

---

## 6. 일정

| 언제 | 할 일 | 산출물 |
|---|---|---|
| 10/1(목) 오후 | 이슈 생성 · HAIC 로 각자 기능 · A 는 CSV 와 임계값 먼저 | 이슈 2 · 3 · 4 머지 |
| 10/1(목) 저녁 | 공유 상수 PR 머지 → 토마토로 전환 | `docker compose up --build` 가 토마토로 뜸 |
| 10/2(금) 오전 | 사이드카 · 데모 시나리오 · ⑥ 화면 캡처 · 리허설 (강사 1:1 지원) | 기획서 ④⑤⑥ |
| (시간 남으면) | 5-1 선택 과제 — 품목 확장. 마감 30분 전까지 안 끝나면 버린다 | — |
| 10/2(금) 점심 | **개발 마감** — 이후 main 은 발표용 fix 만 | — |
| 10/2(금) 14:00 | 조별 발표 · 리뷰 (20분 이내) | — |

---

## 7. 팀 결정이 필요한 것

- [ ] RMSE 임계값 (원/kg) — 단순 예측 612원/kg 미만에서
- [ ] 열 이름 치환 수준 — 수준 1(그대로) / 수준 2(이름까지, 권장이지만 6개 파일)
- [ ] LLM 호출 방식 — 외부 API(키 · 비용 담당자) / 로컬 ollama(이미지 커짐)
- [ ] 승격 시 판정 윈도우 초기화 · 이전 버전 Archived · 회귀 테스트 구현 범위
- [ ] 반입량 API 승인 안 되면 소매가격으로 갈지, 시점은 언제 결정할지
- [ ] 계절용 그룹을 발표에서 어떻게 설명할지 — 철 시작 후 25거래일 전에는 예측을 안 함 / 지난 철 마지막 가중치에서 시작 (설계만, 구현 X)
