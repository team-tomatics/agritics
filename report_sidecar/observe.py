"""
[소유: D 모니터링 · 민영은]  LLM 관측 (기획서 ③ 3-2 "LLM 보고서" 행)

생성형 출력도 에러 없이 틀릴 수 있다 — 예측 모델과 같은 방식으로 감시한다.

TODO
- [ ] 생성 실패 → aiops.log [WARN]
- [ ] 생성 30초 초과 → aiops.log [WARN]
- [ ] 드리프트 상태인데 보고서에 "신뢰도 낮음" 누락 → aiops.log [WARN]
"""


def check_report(report_text: str, drift_detected: bool, elapsed_s: float) -> list[str]:
    raise NotImplementedError
