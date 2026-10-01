# LLM 일일 보고서 사이드카 — 예측 컨테이너와 분리 (LLM 이 죽어도 /predict 는 산다, 기획서 ①)
# 빌드 · 실행은 serving_app/docker-compose.yml 의 report-sidecar 서비스로.
#
# LLM 은 외부 API (팀 결정). 키는 이미지에 넣지 않고 compose 의 env_file(.env) 로 주입한다.
# tensorflow · mlflow 는 필요 없다 — 로그(logs/)를 읽고 보고서(outputs/reports/)만 쓴다.
FROM python:3.11-slim

WORKDIR /app

# 외부 API 는 HTTP 호출이라 requests 하나로 충분. 공급사 SDK 를 쓰게 되면 C 에게 알려 주세요.
RUN pip install --no-cache-dir requests==2.33.1

COPY report_sidecar/ report_sidecar/

# ponytail: 기동 시 1번 + 24시간마다 1번. "매일 07시 정각"이 필요해지면 cron 으로 교체.
# 데모에서 바로 만들려면: docker compose -f serving_app/docker-compose.yml exec report-sidecar python -m report_sidecar.generator
CMD ["sh", "-c", "while true; do python -m report_sidecar.generator; sleep 86400; done"]
