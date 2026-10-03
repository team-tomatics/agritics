# 🌾 agritics — 농수산물 시세 예보 AIOps

가락시장 품목의 최근 25거래일 도매가격 · 반입량으로 **다음날 도매가격(원/kg)** 을 예측하고,
폭염 · 장마로 예측이 틀리기 시작하면 **감지 → 알림 → 자동 재학습 → 게이트 재검증**을 사람 개입 없이 돌린다.
옆에 붙인 **LLM 사이드카**가 예측 이력과 운영 로그를 읽어 점주 · 본사가 바로 읽는 일일 보고서를 만든다 — 예측형 AI + 생성형 AI.

> **서비스 범위 = 농수산물 시세 예보 · 구현 · 발표 품목 = 🍅 토마토** (발표 제목: 토마토 시세 예보 AIOps)

## 품목 그룹

품목은 출하 패턴에 따라 두 그룹으로 나눈다 (설계표: `config/items.yaml`).

| 그룹 | 기준 | 품목 | 이번 구현 |
|---|---|---|---|
| **사계절용** | 연중 출하 — 25거래일(가락시장 한 달) 연속 시퀀스가 끊기지 않는다 | **토마토** · 토마토 완숙 · 방울토마토 (3년 908거래일 모두 있음) | **토마토 1개 모델** |
| **계절용** | 출하 철에만 거래 — 철마다 새로 시작하는 짧은 시계열 | 토마토 대저 (3년 중 288일) · 딸기 (여름 공백) | 설계 · 발표로만 |

**보고서 시각 = 가락시장 경매 시각 기준.** 토마토(과일류)는 02:00 경매 → 경락가 확정 · 데이터 반영 → **06:00 보고서** → 07시 전 점주 · 본사가 읽음. 품목마다 경매 시각이 달라(사과 · 배 08:00, 채소류 18:30–23:00) 확장하면 보고서 시각도 품목별로 정해진다 ([기획서](docs/기획서.md) 「경매 시각과 보고서 시각」).

품목을 늘리는 경로 (기획 · 발표용 설계 — 이번에 구현하지 않음):

| | 규칙 | 토마토 (구현) |
|---|---|---|
| 데이터 | `data/<품목 키>_prices.csv` (`Date,Close,Volume`) | `data/tomato_prices.csv` |
| 모델 이름 (MLflow) | `<품목>_Price_Predictor` | `Tomato_Price_Predictor` |
| 설계표 | `config/items.yaml` 의 그룹 · 품목 · 경매 시각 | `default_item: tomato` · `auction_time: "02:00"` → 보고서 06:00 |

**기획은 범용**(농수산물 전체 · 그룹핑), **구현은 토마토만**이다. 그룹은 "어떤 품목을 같은 방식으로 돌릴 수 있는가"를 나누는 기준이고, 발표에서 설계로 보여 준다 (기획서 「품목 그룹」 · 교수님 피드백 10/1).

SKALA 모델 서빙 및 AIOps 조별 미니프로젝트 (10반). 베이스는 교수님 실습 스켈레톤(HAIC LSTM 서빙 → MLOps → AIOps).

| 문서 | 내용 |
|---|---|
| [docs/기획서.md](docs/기획서.md) | ① 이해관계자 가치 · ② 운영 목표 · ③ 게이트 · 모니터링 · 드리프트 대응 (`기획서.docx` 와 같은 내용) |
| [docs/ROLES.md](docs/ROLES.md) | 역할 분담 · 공유 상수 위치 · 의존 그래프 · 초기 이슈 · 일정 |
| [docs/API.md](docs/API.md) | ⑤ API 명세 · Lazy/Eager 응답 시간 측정 |
| [docs/실험_반입량.md](docs/실험_반입량.md) | 반입량이 줄면 도매가가 오르나 — 데이터(+8.1%) · 모델 what-if(+13.2%) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 브랜치 · 커밋 · PR 규칙 |
| [docs/skeleton_README.md](docs/skeleton_README.md) | 교수님 스켈레톤 원본 설명 |

## 구조

```
serving_app/            FastAPI 예측 서버 (8077 로컬 · 8099 컨테이너)
├── routers/            predict · health · data · logs · report(신규)
├── monitoring/         drift_detector · retrain_trigger
├── history_log.py      이력 로그 JSONL (신규)
├── train_and_register.py  학습 · 게이트 · MLflow 등록
└── static/index.html   대시보드
report_sidecar/         LLM 일일 보고서 (신규) — 예측 API 와 분리
data/                   품목별 CSV (<품목 키>_prices.csv) · 피처 · 업로드
scripts/                baseline · 드리프트 시뮬레이션 · 공유 상수 검사
config/items.yaml       품목 그룹 설계표 (사계절용 / 계절용) — 코드가 읽지 않음
```

## 빠른 실행

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose -f serving_app/docker-compose.yml up --build     # → http://localhost:8099/
```

로컬 실행 순서는 [docs/ROLES.md](docs/ROLES.md) 0장.

판정 기준: **배포 게이트** 검증 RMSE ≤ 612원/kg · 회귀 테스트 RMSE ≤ 현재 Production / **드리프트** 최근 15건 WAPE(오차 합 ÷ 가격 합) > 18% → 재학습 (기획서 3-1 · 3-2)

## 시연 순서 (재현)

컨테이너를 새로 띄운 뒤 그대로 따라 하면 10/3 main 실측과 같은 결과가 나온다 (`simulate_drift.py` 는 시드 고정).

```bash
docker compose -f serving_app/docker-compose.yml down -v          # 이전 로그 · 업로드 비우기
docker compose -f serving_app/docker-compose.yml up -d --build    # 빌드 때 토마토 903행 학습 → v1 (RMSE 540.04)
curl -s localhost:8099/health                                     # model_loaded: true 가 될 때까지 대기

# ① 전체 데이터 상태 — 드리프트 → 재학습 → 게이트 탈락, v1 유지
.venv/bin/python scripts/simulate_drift.py --target container
#   폭염 배치 wape=20.81 → [FAIL] new_rmse=1258.41 - gate/regression failed, keep current Production

# ② 2026-07-30 까지 자른 CSV 업로드 → 같은 시뮬레이션 — 재학습 통과, v2 승격
awk -F, 'NR==1 || $1<="2026-07-30"' data/tomato_prices.csv > /tmp/tomato_until_0730.csv
curl -s -F "file=@/tmp/tomato_until_0730.csv" localhost:8099/data/upload
.venv/bin/python scripts/simulate_drift.py --target container
#   폭염 배치 wape=20.91 → [OK] new_rmse=490.56 - production promoted: Tomato_Price_Predictor v2
```

③ 확인: 대시보드 `http://localhost:8099/` (파이프라인 · 승격 이력 · 알림) · `POST /predict` 의 `model_version` 이 `production-v2` · `GET /report` (보고서는 사이드카가 시작할 때와 매일 06:00 에 만든다 — 바로 다시 만들려면 `docker compose -f serving_app/docker-compose.yml exec report-sidecar python -m report_sidecar.generator`, OpenAI 키가 없으면 템플릿 보고서).

재학습은 **가장 최근 업로드 CSV 의 마지막 50행**으로 한다. ①은 그 50행이 9월 급등기라 게이트(RMSE ≤ 612)를 못 넘고, ②는 급등 전 구간이라 통과한다. 응답 예시는 [docs/API.md](docs/API.md).

## 팀

| 역할 | 이름 |
|---|---|
| A 데이터 오너 | 심준용 |
| B 서빙 오너 · Git | 박유진 |
| C MLOps 오너 | 황재원 |
| D 모니터링 오너 | 민영은 |
