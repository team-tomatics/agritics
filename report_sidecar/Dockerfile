# LLM 일일 보고서 사이드카 — 예측 컨테이너와 분리 (LLM 이 죽어도 /predict 는 산다, 기획서 ①)
# 빌드 · 실행은 serving_app/docker-compose.yml 의 report-sidecar 서비스로.
#
# LLM 은 OpenAI 외부 API (#14). 키(OPENAI_API_KEY)는 이미지에 넣지 않고 compose 의 env_file(.env) 로 주입한다.
# 키가 없으면 generator 가 템플릿 보고서를 만든다. tensorflow · mlflow 는 필요 없다.
FROM python:3.11-slim

WORKDIR /app

# requests: OpenAI HTTP 호출 · fastapi: serving_app/history_log.py 가 import (집계 함수 재사용)
RUN pip install --no-cache-dir requests==2.33.1 fastapi==0.141.1

COPY serving_app/__init__.py serving_app/history_log.py serving_app/
COPY report_sidecar/ report_sidecar/

# print 가 버퍼에 묶이지 않고 docker logs 에 바로 보이게 (#28)
ENV PYTHONUNBUFFERED=1

# 시작 시 1회 + 매일 06:00 KST (토마토 경매 02:00 기준, #27). 데모에서 바로 만들려면:
#   docker compose -f serving_app/docker-compose.yml exec report-sidecar python -m report_sidecar.generator
CMD ["python", "-m", "report_sidecar.generator", "--at", "06:00"]
