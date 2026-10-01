"""
[소유: B 서빙 · 박유진]  GET /report — 일일 보고서 (기획서 ② 핵심 기능)

report_sidecar 가 만든 최신 보고서(outputs/reports/*.md)를 돌려준다.
LLM 이 실패해도 /predict 는 영향이 없어야 한다 (사이드카로 분리한 이유).

TODO
- [ ] 최신 보고서 파일 읽어 반환 (없으면 404)
- [ ] 응답에 생성 시각 · 모델 버전 · 드리프트 상태 포함
"""
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/report")
def get_report():
    raise HTTPException(status_code=501, detail="아직 구현 전 (report_sidecar 참고)")
