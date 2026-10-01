# report_sidecar — LLM 일일 보고서 (생성형 사이드카)

예측 컨테이너 옆에서 따로 돈다. `logs/history.jsonl`(이력 로그) · `logs/aiops.log`(드리프트 · 재학습)를
읽어 점주 · 본사가 읽을 일일 보고서를 `outputs/reports/YYYY-MM-DD.md` 로 쓴다.
**LLM 은 숫자를 만들지 않는다** — 예측값 · RMSE · 버전은 로그에 있는 값을 그대로 옮기고 문장만 만든다.

| 파일 | 소유 | 할 일 |
|---|---|---|
| `prompts/report.md` | A 데이터 · 심준용 | 점주 · 본사가 읽을 문장 — 품목 · 시세 지식, "신뢰도 낮음" 문구 규칙 |
| `generator.py` | B 서빙 · 박유진 | 로그 읽기 → 프롬프트 채우기 → LLM 호출 → md 저장 |
| `observe.py` | D 모니터링 · 민영은 | 생성 실패 · 30초 초과 · 드리프트인데 "신뢰도 낮음" 누락 → `aiops.log` 경고 |
| `Dockerfile` · compose 서비스 | C MLOps · 황재원 | 사이드카 컨테이너, API 키는 `.env` 로 주입, `logs/` · `outputs/` 볼륨 공유 |

팀 결정 필요: LLM 호출 방식 — 외부 API(키 · 비용 담당) / 로컬 ollama(이미지 커짐). 정해지면 `.env.example` 갱신.
