"""
GET /report — 일일 보고서 (기획서 ② 핵심 기능)

작성자: 박유진
버전: v1.0.0 (2026-10-01)
변경 이력:
  v1.0.0  #16  최신 보고서 반환 · 없으면 404

report_sidecar(generator.py) 가 만든 최신 보고서(outputs/reports/YYYY-MM-DD.md + .json)를 돌려준다.
서빙 프로세스는 LLM 을 부르지 않는다 — LLM 이 느리거나 죽어도 /predict 는 영향이 없다 (사이드카로 분리한 이유).
"""
import glob
import json
import os

from fastapi import APIRouter, HTTPException

router = APIRouter()

REPORT_DIR = "outputs/reports"


@router.get("/report")
def get_report():
    """
    최신 일일 보고서.
    - source: "llm" (생성형 문장) / "template" (LLM 실패 → 정해진 양식, error 에 사유)
    - drift_detected 가 true 면 본문 첫 줄에 "가격 급변 감지 — 오늘 예측 신뢰도 낮음"
    - 보고서가 아직 없으면 404 — `python -m report_sidecar.generator` 로 생성
    """
    metas = sorted(glob.glob(os.path.join(REPORT_DIR, "*.json")))
    if not metas:
        raise HTTPException(404, "아직 생성된 보고서가 없습니다 (python -m report_sidecar.generator)")
    with open(metas[-1], encoding="utf-8") as f:
        meta = json.load(f)
    md_path = metas[-1][: -len(".json")] + ".md"
    with open(md_path, encoding="utf-8") as f:
        markdown = f.read()
    facts = meta.get("facts", {})
    return {
        "date": meta["date"],
        "generated_at": meta["generated_at"],
        "source": meta["source"],
        "error": meta.get("error"),
        "elapsed_s": meta.get("elapsed_s"),
        "item_name": facts.get("item_name"),
        "model_version": facts.get("model_version"),
        "drift_detected": facts.get("drift_detected"),
        "markdown": markdown,
    }
