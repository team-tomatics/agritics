# report_sidecar — LLM 일일 보고서 (생성형 사이드카)

예측 컨테이너 옆에서 따로 돈다. `logs/history.jsonl`(이력 로그) · `logs/aiops.log`(드리프트 · 재학습)를
읽어 점주 · 본사가 읽을 일일 보고서를 `outputs/reports/YYYY-MM-DD.md` 로 쓴다.
**LLM 은 숫자를 만들지 않는다** — 예측값 · RMSE · 버전은 로그에 있는 값을 그대로 옮기고 문장만 만든다.
구현 · 발표 품목은 토마토 하나.

| 파일 | 소유 | 할 일 |
|---|---|---|
| `prompts/report.md` | A 데이터 · 심준용 | 점주 · 본사가 읽을 문장 — 품목 · 시세 지식, "신뢰도 낮음" 문구 규칙 |
| `generator.py` | B 서빙 · 박유진 | 로그 읽기 → 프롬프트 채우기 → LLM 호출 → md 저장 |
| `observe.py` | D 모니터링 · 민영은 | 생성 실패 · 30초 초과 · 드리프트인데 "신뢰도 낮음" 누락 → `aiops.log` 경고 |
| `Dockerfile` · compose 서비스 | C MLOps · 황재원 | 사이드카 컨테이너, API 키는 `.env` 로 주입, `logs/` · `outputs/` 볼륨 공유 |

**결정 (10/1, 이슈 #14)**: LLM = OpenAI (`OPENAI_API_KEY` · `LLM_MODEL`) · 실패(키 없음 · 30초 초과 · 오류)하면 같은 숫자로 **템플릿 보고서**를 만들고 `source: "template"` 로 표시.

```bash
python -m report_sidecar.generator              # 1회 → outputs/reports/YYYY-MM-DD.md + .json
python -m report_sidecar.generator --every 3600 # 사이드카 컨테이너용 반복
curl localhost:8077/report                      # 최신 보고서 (없으면 404)
```

- 숫자는 `collect_facts()` 가 로그에서 계산한다. LLM 프롬프트와 템플릿이 같은 dict 를 쓴다
- `prompts/report.md` 에 `TODO` 가 남아 있는 동안은 `generator.py` 의 기본 프롬프트를 쓴다. A 가 채울 때 쓸 수 있는 키: `item_name` `generated_at` `predicted_price` `last_price` `change_pct` `model_version` `drift_detected` `drift_count` `promotions` `request_count` `avg_latency_ms` `error_rate` `low_confidence_rule`
- C: compose 사이드카는 위 `--every` 명령 + `env_file: .env` + `logs/` · `outputs/` 볼륨 공유 (`serving_app/history_log.py` 를 import 하므로 같은 코드 · 의존성 이미지를 써도 됨)
- D: 생성 직후 `observe.check_report(markdown, drift_detected, elapsed_s)` 를 부른다 (지금은 NotImplementedError 무시). 메타 `.json` 의 `source` · `error` · `elapsed_s` 도 쓸 수 있다
