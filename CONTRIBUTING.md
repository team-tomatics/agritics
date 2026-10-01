# 협업 규칙

> 이슈 → 브랜치 → PR → 리뷰 → 머지. 예외 없이 이 순서로 갑니다.
> `main` 에 직접 push 금지.

---

## 0. 이 저장소의 특수 사정

개발 시간이 **10/1 오후 ~ 10/2 점심**뿐이고, 4명이 **하나의 파이프라인**(업로드 → 학습 · 게이트 → 서빙 → 드리프트 → 재학습 → 보고서)을 나눠 고칩니다.
그래서 아래 두 가지가 다른 어떤 규칙보다 중요합니다.

1. **공유 상수는 한 사람이 정해 공지하고, 모든 위치를 한 PR 에서 같이 바꾼다** — 입력 길이 · 판정 윈도우 · RMSE 임계값 · 기준 가격 · 모델 이름 · 열 이름. 위치 목록은 [docs/ROLES.md](docs/ROLES.md) 3장. 대시보드(`index.html`)와 시뮬레이터는 서버 값을 **복사**해 쓰기 때문에 한 곳만 고치면 조용히 어긋납니다. CI 가 `scripts/check_constants.py` 로 막습니다
2. **자기 파일만 고친다** — 소유는 [docs/ROLES.md](docs/ROLES.md) 1장. 남의 파트는 이슈로 넘긴다

충돌이 난다면 거의 `index.html`(A · D 상수 + D 화면) · `main.py`(라우터 · 미들웨어 등록) · `Dockerfile` · `docker-compose.yml` 입니다.

### 1회 설정 — 각자 자기 PC 에서 한 번만

```bash
git clone https://github.com/dolmaroyujinpark/agritics.git
cd agritics
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt     # tensorflow 포함이라 몇 분
cp .env.example .env                          # 키 채우기 (커밋 금지)
```

- zsh 에 `pip` alias 가 걸려 있으면 venv 를 켜도 전역에 깔립니다. **`.venv/bin/pip` 경로로 설치**하세요
- 명령은 전부 **프로젝트 루트**에서 (`serving_app/` 안에서 돌리면 `data` 모듈을 못 찾습니다)
- 패키지 추가는 C(MLOps)에게 이슈 → `requirements.txt` 는 C 가 고칩니다 (컨테이너 이미지와 같이 움직임)

---

## 1. 파트 소유

| | 이름 | 역할 | 소유 파일 |
| --- | --- | --- | --- |
| **A** | 심준용 | 데이터 오너 | `data/` · `serving_app/routers/data.py` · `scripts/train_baseline_v1.py` · `scripts/fetch_*.py` · `config/items.yaml` · `report_sidecar/prompts/` |
| **B** | 박유진 | 서빙 오너 + Git 저장소 | `serving_app/schemas.py` `model_loader.py` `main.py` `history_log.py` · `routers/predict.py` `health.py` `report.py` · `report_sidecar/generator.py` |
| **C** | 황재원 | MLOps 오너 | `serving_app/train_and_register.py` `lstm_model.py` `Dockerfile` `docker-compose.yml` · `requirements.txt` · `report_sidecar/Dockerfile` |
| **D** | 민영은 | 모니터링 오너 | `serving_app/monitoring/` · `scripts/simulate_drift.py` · `serving_app/static/index.html` · `routers/logs.py` · `report_sidecar/observe.py` |

`.github/` · `.env.example` · `docs/` · `README.md` · `scripts/check_constants.py` 는 공용입니다. 바꾸기 전에 슬랙에 공유하세요.
누가 무엇을 기다리는지는 [docs/ROLES.md](docs/ROLES.md) 의존 그래프 참고.

**남의 파트에 변경이 필요하면 직접 고치지 말고 이슈를 만들어 소유자에게 넘깁니다.**
예외: 공유 상수 변경 PR 은 여러 파일을 건드릴 수 있습니다 — 해당 파일 소유자 전원을 리뷰어로 겁니다.

---

## 2. 브랜치 전략

개발이 하루라 `develop` 없이 **main + 작업 브랜치** 2단계로만 갑니다.

```
main ────●────────●────────●────────●──▶   (항상 docker compose up --build 가 도는 상태)
          \      /  \     /  \     /
           feat/3   feat/7   fix/11        (이슈 1개 = 브랜치 1개)
```

### 네이밍 규칙

```
<type>/<이슈번호>-<slug>
```

| type | 언제 | 예시 |
| --- | --- | --- |
| `feat` | 새 기능 (API, 미들웨어, 사이드카, 시나리오) | `feat/4-history-log` |
| `fix` | 버그 수정 | `fix/11-cache-not-cleared` |
| `chore` | 설정, CI, 의존성, 공유 상수 | `chore/5-tomato-threshold` |
| `docs` | 문서·기획서·README 만 수정 | `docs/8-api-spec` |
| `refactor` | 동작 변화 없는 정리 | `refactor/14-rename-close-to-price` |
| `data` | 데이터 수집 · CSV 교체 | `data/2-kamis-tomato-csv` |

- slug 는 **영문 소문자 + 하이픈**. 한글·공백·대문자 금지.
- 이슈 번호는 필수. 번호 없는 브랜치는 리뷰하지 않습니다.
- 한 브랜치에 한 가지 일만. 커지면 이슈를 쪼개세요.

> 이슈 페이지 오른쪽 **Development → Create a branch** 를 쓰면 브랜치가 이슈에 자동 연결되고,
> PR 머지 시 이슈가 자동으로 닫힙니다. GitHub 이 제안하는 이름(`4-feat-...`)은
> 그 팝업에서 `feat/4-history-log` 로 **직접 고쳐서** 만드세요.

---

## 3. 작업 순서

```bash
# 1) 이슈 생성 (GitHub 웹에서 템플릿 선택) → 이슈 번호 확인 (#4)

# 2) 최신 main 받기
git switch main
git pull origin main

# 3) 브랜치 생성
git switch -c feat/4-history-log

# 4) 작업 → 로컬 확인 → 커밋
python3 scripts/check_constants.py
.venv/bin/uvicorn serving_app.main:app --port 8077     # /docs 에서 확인
git add <내 파일>
git commit -m "feat(serving): 이력 로그 미들웨어 — /predict 지연 · 모델 버전 기록"

# 5) 푸시
git push -u origin feat/4-history-log

# 6) GitHub에서 PR 생성 (템플릿 자동 삽입됨) → 리뷰 요청

# 7) 승인 후 Squash and merge → 브랜치 삭제
```

`git add .` 대신 **내 파일만** add 하세요. `mlflow.db` · `mlruns/` · `logs/` 는 gitignore 돼 있지만, 실수로 남의 파일을 같이 올리는 일이 잦습니다.

---

## 4. 커밋 컨벤션

```
<type>(<scope>): <한글 요약>
```

scope 는 파트: `data` `serving` `mlops` `monitor` `sidecar` `docs` `ci`

| type | 의미 |
| --- | --- |
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `chore` | 설정, 빌드, 패키지, 공유 상수 |
| `docs` | 문서 |
| `refactor` | 리팩터링 |
| `test` | 테스트 · 측정 스크립트 |
| `data` | 데이터 수집 · CSV |

```bash
git commit -m "data(data): KAMIS 가락시장 토마토 908거래일 CSV"
git commit -m "chore(mlops): 게이트 · 드리프트 · 대시보드 임계값 600원/kg 으로 통일"
git commit -m "fix(monitor): 승격 직후 판정 윈도우 초기화"
git commit -m "feat(sidecar): 일일 보고서 생성기 — 타임아웃 30초"
```

**scope 를 붙이는 이유** — 파트별로 파일이 갈려 있어서 scope 만 보면 누구 작업인지 압니다.

### gitmoji는 **선택**입니다

쓰고 싶으면 type 앞에 붙이세요. 안 써도 리뷰에서 지적하지 않습니다.

| gitmoji | type | | gitmoji | type |
| --- | --- | --- | --- | --- |
| ✨ | feat | | ♻️ | refactor |
| 🐛 | fix | | ✅ | test |
| 📝 | docs | | 🔧 | chore |
| 🗃️ | data | | | |

> 딱 하나만 맞춰주세요: **`<type>:` 은 반드시 있어야 합니다.**

---

## 5. PR 규칙

- 제목: `[FEAT] 이력 로그 미들웨어 — /predict 지연 · 모델 버전 기록` (이슈 제목과 맞추면 편합니다)
- 본문의 `closes #4` 를 **반드시** 채우기 → 머지 시 이슈 자동 종료
- 리뷰어 **최소 1명** 승인 후 머지
- 머지 방식: **Squash and merge**
- 머지 후 원격 브랜치 삭제
- **스스로 머지하지 않기.** 리뷰가 급하면 슬랙에서 부르기
- 발표에 쓸 캡처 · 로그는 PR 본문 "발표 증빙" 칸에 붙이기 — 발표 자료를 PR 에서 모읍니다

### PR을 잘게 쪼개주세요

**"API 하나 = PR 하나", "공유 상수 변경 = PR 하나"** 수준으로 올려야
리뷰가 밀리지 않고 마지막에 충돌이 안 납니다.

### CI 를 통과해야 머지할 수 있습니다

PR 을 올리면 `검사` 가 자동으로 돕니다.

| 검사 | 실패하면 |
| --- | --- |
| API 키 하드코딩 (`sk-`, `sk-ant-`, `..._KEY = "..."`) | ❌ 실패 — `.env` 로 옮기기 |
| 커밋 금지 파일 (`.env`, `mlflow.db`, `mlruns/`, `*.keras`, `scaler.pkl`, `data/uploads/*.csv`, `logs/*`, `reference/`) | ❌ 실패 — `git rm --cached` |
| `ruff check` (문법 오류 · 정의 안 된 이름만) | ❌ 실패 |
| 공유 상수 일치 (`scripts/check_constants.py`) | ❌ 실패 — ROLES.md 3장 위치 전부 고치기 |

---

## 6. 🚨 하지 말아야 할 것

| 대상 | 이유 |
| --- | --- |
| **공유 상수를 한 곳만 바꾸기** | 서버는 새 값, 대시보드 · 시뮬레이터는 옛 값으로 돌아 데모에서 깨집니다 |
| **남의 파일 수정** | 소유자에게 이슈로 넘기세요 |
| **LLM 이 예측값을 만들거나 고치게 하기** | 기획서 ② "LLM 은 숫자를 만들지 않는다". 숫자는 로그 값 그대로 |
| **LLM 호출을 `/predict` 경로 안에 넣기** | LLM 이 죽으면 예측도 죽습니다. 사이드카로 분리한 이유 (기획서 ① 표) |
| **샘플 CSV 를 HAIC 로 둔 채 컨테이너 빌드** | Dockerfile 이 빌드 때 학습합니다 — 컨테이너 안 모델이 HAIC 가 됩니다 |
| **`mlflow.db` · 모델 파일 커밋** | 각자 PC 에서 재생성. 커밋하면 버전이 꼬입니다 |
| **API 키 코드 직접 입력** | `.env`. CI 가 막습니다 |
| **`main` 직접 push** | 아래 ⚠️ 참조 |

> ⚠️ **`main` 보호는 GitHub 이 강제하지 못할 수 있습니다.** 무료 플랜의 private 저장소에는
> branch protection·ruleset 을 걸 수 없습니다 (public 전환 시 가능).
> 서로 지키는 것으로 합니다.

---

## 7. 기획서와 코드가 어긋나면

기획서(`docs/기획서.md`)는 10/1 확정본입니다. 개발하다 바꿔야 할 게 생기면:

1. `[CHORE] 기획 변경` 이슈로 올리고 조원 합의
2. 코드 PR 에서 "기획서 반영" 칸에 절 번호 적기
3. 노란 칸(팀 결정)을 정했으면 `docs/기획서.md` 도 같은 PR 에서 채우기

발표 ④⑤⑥ 은 구현 결과로 채우므로, 기획서 ②③ 의 숫자와 실제 코드 값이 같아야 합니다.

---

## 8. 충돌이 났다면

```bash
git fetch origin && git rebase origin/main    # 내 브랜치를 최신 main 위로
```

`index.html` 상수 블록 충돌이면 **main 쪽 값**(먼저 머지된 공유 상수 PR)을 살리고 내 화면 변경만 다시 얹습니다.
`main.py` 충돌이면 라우터 · 미들웨어 등록 줄을 양쪽 다 살립니다.
`requirements.txt` 충돌이면 양쪽 패키지를 다 살리고 C 에게 알립니다.

---

## 9. 부록 — 이슈 템플릿 YAML 읽는 법

`.github/ISSUE_TEMPLATE/*.yml` 은 GitHub 의 **Issue Forms** 형식입니다.

```yaml
name: "✨ Feature"          # 이슈 만들기 화면의 카드 제목
description: "새 기능 추가"   # 카드 부제
title: "[FEAT] "           # 이슈 제목 기본값
labels: ["feature"]        # 자동으로 붙는 라벨

body:                      # 위에서부터 순서대로 렌더링
  - type: input            # 한 줄 입력
  - type: textarea         # 여러 줄 입력
  - type: dropdown         # 선택지 (options 필요)
  - type: checkboxes       # 체크박스 묶음
  - type: markdown         # 안내문 (제출 내용에 안 들어감)
```

| 자주 틀리는 것 | 결과 |
| --- | --- |
| 들여쓰기에 **탭** 사용 | 파싱 실패 → 템플릿이 아예 안 보임 |
| `label` 없이 `description` 만 씀 | `label` 은 필수 |
| `placeholder` 를 기본값으로 착각 | 실제로 채우려면 `value` |
| `config.yml` 을 템플릿으로 착각 | 템플릿이 아니라 **선택 화면 설정** |

> **템플릿은 `main` 에 있어야 보입니다.** PR 브랜치에만 있으면 이슈 생성 화면에 안 뜹니다.
> **PR 템플릿은 저장소 화면 어디에도 따로 안 보입니다.** 브랜치를 푸시하고 `Compare & pull request` 를 눌렀을 때 PR 본문에 자동으로 채워지는 게 전부입니다.
> `…/issues/new/choose` 에서 확인하세요.
