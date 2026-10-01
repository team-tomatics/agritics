# 🍅 토마토 시세 예보 AIOps

가락시장 토마토의 최근 20거래일 도매가격 · 반입량으로 **다음날 도매가격(원/kg)** 을 예측하고,
폭염 · 장마로 예측이 틀리기 시작하면 **감지 → 알림 → 자동 재학습 → 게이트 재검증**을 사람 개입 없이 돌린다.
옆에 붙인 **LLM 사이드카**가 예측 이력과 운영 로그를 읽어 점주 · 본사가 바로 읽는 일일 보고서를 만든다 — 예측형 AI + 생성형 AI.

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
data/                   CSV · 피처 · 업로드
scripts/                baseline · 드리프트 시뮬레이션 · 공유 상수 검사
config/items.yaml       품목 그룹 (사계절용 / 계절용)
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
