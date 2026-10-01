<!-- 제목: [FEAT] 이력 로그 미들웨어 — /predict 지연 · 모델 버전 기록  (이슈 제목과 맞추기) -->

## 관련 이슈
- closes #

> 예시: closes #4

---

## 변경 사항
이번 PR에서 어떤 작업을 했는지 간단하게 작성해주세요.

- 
- 
- 

---

## 건드린 파일
수정한 곳에 체크해주세요. **자기 파트만 체크되어야 정상입니다.**

- [ ] `data/` &nbsp;/&nbsp; `serving_app/routers/data.py` &nbsp;/&nbsp; `scripts/train_baseline_v1.py` `fetch_*.py` &nbsp;/&nbsp; `config/items.yaml` &nbsp;/&nbsp; `report_sidecar/prompts/` — **A** 심준용
- [ ] `serving_app/schemas.py` `model_loader.py` `main.py` `history_log.py` &nbsp;/&nbsp; `routers/predict.py` `health.py` `report.py` &nbsp;/&nbsp; `report_sidecar/generator.py` — **B** 박유진
- [ ] `serving_app/train_and_register.py` `lstm_model.py` `Dockerfile` `docker-compose.yml` &nbsp;/&nbsp; `requirements.txt` &nbsp;/&nbsp; `report_sidecar/Dockerfile` — **C** 황재원
- [ ] `serving_app/monitoring/` &nbsp;/&nbsp; `scripts/simulate_drift.py` &nbsp;/&nbsp; `serving_app/static/index.html` &nbsp;/&nbsp; `routers/logs.py` &nbsp;/&nbsp; `report_sidecar/observe.py` — **D** 민영은
- [ ] `config/items.yaml` (품목 그룹 설계표 — 기획 · 발표용)
- [ ] **공유 상수** (입력 길이 · 윈도우 · 임계값 · 기준 가격 · 모델 이름 · 열 이름) ← 체크되면 **ROLES.md 3장 위치를 전부 고쳤는지 확인하고 슬랙에 알리세요**
- [ ] `logs/history.jsonl` 한 줄 형식 ← 체크되면 B · D 둘 다 리뷰
- [ ] `.github/` &nbsp;/&nbsp; `.env.example` &nbsp;/&nbsp; `docs/` &nbsp;/&nbsp; `README.md` (공용)
- [ ] 문서만 수정

---

## 동작 확인
- [ ] 로컬 `uvicorn serving_app.main:app --port 8077` 기동 → `/health` `model_loaded`
- [ ] `/predict` 응답: <!-- predicted_close · model_version -->
- [ ] 컨테이너 `docker compose -f serving_app/docker-compose.yml up --build` → `localhost:8099/docs`
- [ ] 드리프트 경로 영향 시 `python scripts/simulate_drift.py --target container` → `[WARN]` → `[OK]`
- [ ] 끝까지 안 됨 → 어디까지 되는지:

---

## 기획서 반영
- [ ] `docs/기획서.md` 수정이 필요 없습니다
- [ ] 필요합니다 → 어느 절인지: <!-- 예: ③ 3-1 성능 임계값 — 600원/kg 으로 확정 -->

---

## 발표 증빙 (선택)
⑥ 동작 화면 · ⑤ API 명세에 쓸 캡처 · 로그 · 측정값을 붙여주세요. (발표 자료에 그대로 씁니다)

```
```

---

## 리뷰어가 볼 것
<!-- 리뷰어가 30초 안에 판단할 수 있게 — "임계값 세 곳이 같은지만 봐줘" / "캐시 비우는 위치가 승격 직후인지" -->
- 

---

## 체크리스트
- [ ] 브랜치 네이밍 규칙을 준수했습니다. (`feat/이슈번호-slug`)
- [ ] 관련 이슈를 연결했습니다. (`closes #이슈번호`)
- [ ] 커밋 메시지가 `<type>(<scope>): 요약` 형식입니다.
- [ ] API 키를 코드에 직접 적지 않았습니다. (`.env`)
- [ ] `mlflow.db`, `mlruns/`, `*.keras`, `scaler.pkl`, `data/uploads/`, `logs/`, `.env` 를 커밋하지 않았습니다.
- [ ] LLM 이 예측값을 만들거나 고치지 않습니다. (숫자는 로그 값 그대로)
- [ ] 남의 파일을 고치지 않았습니다.
