# 🌾 agritics — 농수산물 시세 예보 AIOps

가락시장 품목의 최근 20거래일 도매가격 · 반입량으로 **다음날 도매가격(원/kg)** 을 예측하고,
폭염 · 장마로 예측이 틀리기 시작하면 **감지 → 알림 → 자동 재학습 → 게이트 재검증**을 사람 개입 없이 돌린다.
옆에 붙인 **LLM 사이드카**가 예측 이력과 운영 로그를 읽어 점주 · 본사가 바로 읽는 일일 보고서를 만든다 — 예측형 AI + 생성형 AI.

> **서비스 범위 = 농수산물 시세 예보 · 구현 · 발표 품목 = 🍅 토마토** (발표 제목: 토마토 시세 예보 AIOps)

## 품목 그룹

품목은 출하 패턴에 따라 두 그룹으로 나눈다. 품목은 코드가 아니라 설정값(`config/items.yaml`)이다.

| 그룹 | 기준 | 품목 | 이번 구현 |
|---|---|---|---|
| **사계절용** | 연중 출하 — 20거래일 연속 시퀀스가 끊기지 않는다 | **토마토** · 토마토 완숙 · 방울토마토 (3년 908거래일 모두 있음) | **토마토 1개 모델** |
| **계절용** | 출하 철에만 거래 — 철마다 새로 시작하는 짧은 시계열 | 토마토 대저 (3년 중 288일) · 딸기 (여름 공백) | 설계 · 발표로만 |

품목을 늘리는 경로 — 품목 하나 = 아래 세 가지 하나씩. 코드는 그대로 두고 설정만 늘린다.

| | 규칙 | 토마토 |
|---|---|---|
| 데이터 | `data/<품목 키>_prices.csv` (`Date,Close,Volume`) | `data/tomato_prices.csv` |
| 모델 이름 (MLflow) | `<품목>_Price_Predictor` | `Tomato_Price_Predictor` |
| 설정 | `config/items.yaml` 의 그룹 · 키 · 이름 | `default_item: tomato` |

예측은 품목마다 LSTM 1개(시계열만)이고, 그룹은 "어떤 품목을 같은 방식으로 돌릴 수 있는가"를 나누는 기준이다 (기획서 「품목 그룹」 · 교수님 피드백 10/1).

SKALA 모델 서빙 및 AIOps 조별 미니프로젝트 (10반). 베이스는 교수님 실습 스켈레톤(HAIC LSTM 서빙 → MLOps → AIOps).

| 문서 | 내용 |
|---|---|
| [docs/기획서.md](docs/기획서.md) | ① 이해관계자 가치 · ② 운영 목표 · ③ 게이트 · 모니터링 · 드리프트 대응 |
| [docs/ROLES.md](docs/ROLES.md) | 역할 분담 · 공유 상수 위치 · 의존 그래프 · 초기 이슈 · 일정 |
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
config/items.yaml       품목 그룹 (사계절용 / 계절용) · 품목별 데이터 · 모델 이름
```

## 빠른 실행

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose -f serving_app/docker-compose.yml up --build     # → http://localhost:8099/
```

로컬 실행 순서는 [docs/ROLES.md](docs/ROLES.md) 0장.

## 팀

| 역할 | 이름 |
|---|---|
| A 데이터 오너 | 심준용 |
| B 서빙 오너 · Git | 박유진 |
| C MLOps 오너 | 황재원 |
| D 모니터링 오너 | 민영은 |
