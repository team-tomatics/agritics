"""
[소유: B 서빙 · 박유진]  일일 보고서 생성 (기획서 ② GET /report · 운영 목표 "매일 07시 전 생성")

TODO
- [ ] logs/history.jsonl · logs/aiops.log 에서 최근 24시간 읽기
- [ ] prompts/report.md 에 값 채우기 (내일 가격 · 신뢰도 · 드리프트 · 재학습 이력)
- [ ] LLM 호출 (키는 .env — 코드에 직접 쓰지 않음) · 타임아웃 30초
- [ ] outputs/reports/YYYY-MM-DD.md 저장
"""


def generate() -> str:
    raise NotImplementedError


if __name__ == "__main__":
    print(generate())
