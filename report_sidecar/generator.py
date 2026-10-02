"""
일일 보고서 생성 (기획서 ② GET /report · 운영 목표 "매일 07시 전 생성")

작성자: 박유진
버전: v1.2.0 (2026-10-01)
변경 이력:
  v1.0.0  #16  로그 집계 → OpenAI 요약 · 실패 시 템플릿 보고서
  v1.1.0  #26  서비스 상태를 예측 요청 기준으로
  v1.2.0  #29  매일 06:00 KST 생성 (토마토 02:00 경매 기준)

예측형 AI(LSTM) 의 이력을 생성형 AI(LLM) 가 사람이 읽는 문장으로 바꾼다.
  1) logs/history.jsonl · logs/aiops.log 에서 숫자를 **코드로** 계산한다 (collect_facts)
  2) LLM 은 그 숫자로 문장만 만든다 — 숫자를 만들거나 고치지 않는다
  3) LLM 이 실패하면(키 없음 · 30초 초과 · 오류) 같은 숫자로 템플릿 보고서를 만든다
     → 점주가 아침 구매 판단 시간을 놓치지 않게 (10/1 결정, 이슈 #14)
  4) outputs/reports/YYYY-MM-DD.md (본문) + .json (메타: source · 소요 시간 · 실패 사유 · 숫자)

실행 (프로젝트 루트에서)
  python -m report_sidecar.generator              # 1회 생성
  python -m report_sidecar.generator --at 06:00   # 사이드카 컨테이너: 시작 시 1회 + 매일 06:00 (한국 시간)
  python -m report_sidecar.generator --every 3600 # (테스트용) N초마다

보고서 시각 = 가락시장 경매 시각 기준 (#27). 토마토(과일류) 02:00 경매 → 경락가 확정 · 데이터 반영 → 06:00 보고서
→ 점주 · 본사가 07시 전에 읽는다 (기획서 ② 보고서). 품목을 늘리면 품목별 경매 시각으로 --at 이 달라진다.

환경변수 (.env)
  OPENAI_API_KEY  없으면 템플릿 보고서
  LLM_MODEL       기본 gpt-4o-mini
  LLM_TIMEOUT     기본 30 (초, 기획서 3-2 "30초 초과" 경고 기준)
  ITEM_NAME       기본 토마토
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, time as dtime, timedelta

import requests

from serving_app import history_log

AIOPS_LOG = "logs/aiops.log"
REPORT_DIR = "outputs/reports"
PROMPT_PATH = "report_sidecar/prompts/report.md"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
LOW_CONFIDENCE = "가격 급변 감지 — 오늘 예측 신뢰도 낮음"  # 기획서 3-3 알림 문구 (D 의 observe 가 이 문장을 검사)

# A(심준용) 가 prompts/report.md 를 채우기 전까지 쓰는 기본 프롬프트. {키} 는 collect_facts() 의 키
DEFAULT_PROMPT = """너는 외식 프랜차이즈 가맹점주와 본사 구매 담당자에게 {item_name} 도매 시세 일일 보고서를 쓰는 비서다.
아래 [데이터]의 숫자만 그대로 쓴다. 새로운 숫자 · 날짜 · 기간 · 수량을 만들지 않는다. 데이터가 "없음"이면 없다고 쓴다.
{low_confidence_rule}
형식: 마크다운, 5~8줄. 1) 내일 가격 한 줄 2) 오늘 대비 변화와 구매 판단에 참고할 점 3) 드리프트 · 재학습 상황 4) 서비스 상태.
구매를 지시하지 말고 "검토"로 쓴다.

[데이터]
- 품목: {item_name} (가락시장 도매가, 원/kg)
- 보고 기준 시각: {generated_at}
- 내일 예측가: {predicted_price}
- 오늘(입력 마지막 날) 가격: {last_price}
- 변화율: {change_pct}
- 예측 모델 버전: {model_version}
- 최근 24시간 드리프트 감지: {drift_count}회 · 재학습 승격: {promotions}
- 최근 24시간 예측 요청 {request_count}건 · 예측 평균 응답 {avg_latency_ms} · 서버 에러율 {error_rate}
"""


def _fmt_won(v):
    return f"{v:,.0f}원/kg" if isinstance(v, (int, float)) else "없음"


def _read_aiops(hours: float, now: datetime) -> tuple[int, list[str]]:
    """aiops.log 최근 hours 시간의 드리프트 감지 횟수 · 승격 이력. (retrain_trigger.py 가 남기는 형식)"""
    if not os.path.isfile(AIOPS_LOG):
        return 0, []
    since = now - timedelta(hours=hours)
    drift, promos = 0, []
    with open(AIOPS_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts < since:
                continue
            if "[WARN] drift detected" in line:
                drift += 1
            m = re.search(r"\[OK\] new_rmse=([\d.]+) - production promoted: (\S+) v(\d+)", line)
            if m:
                promos.append(f"v{m.group(3)} (RMSE {m.group(1)})")
    return drift, promos


def load_dotenv(path: str = ".env") -> None:
    """.env 를 읽어 비어 있는 환경변수만 채운다 (python-dotenv 없이). 컨테이너는 compose env_file 로 들어온다."""
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip() and not os.getenv(k.strip()):
                os.environ[k.strip()] = v.strip()


def collect_facts(hours: float = 24, item_name: str | None = None) -> dict:
    """보고서에 들어갈 숫자를 전부 여기서 계산한다. LLM · 템플릿 둘 다 이 dict 만 쓴다."""
    now = datetime.now(history_log.KST)
    entries = history_log.read_entries(minutes=hours * 60)
    preds = [e for e in entries if e.get("path") == "/predict" and e.get("status") == 200 and "predicted" in e]
    last = preds[-1] if preds else {}
    predicted = last.get("predicted")
    last_price = (last.get("input_last") or {}).get("close")
    change = (predicted / last_price - 1) * 100 if predicted and last_price else None
    summary = history_log.summarize(minutes=hours * 60)
    # aiops.log 는 logging 의 로컬 시각(컨테이너면 UTC)이라 같은 기계의 로컬 시각과 비교한다
    drift_count, promotions = _read_aiops(hours, datetime.now())
    return {
        "item_name": item_name or os.getenv("ITEM_NAME", "토마토"),
        "generated_at": now.isoformat(timespec="seconds"),
        "predicted_price": _fmt_won(predicted),
        "last_price": _fmt_won(last_price),
        "change_pct": f"{change:+.1f}%" if change is not None else "없음",
        "model_version": last.get("model_version", "없음"),
        "drift_detected": drift_count > 0,
        "drift_count": drift_count,
        "promotions": ", ".join(promotions) if promotions else "없음",
        "request_count": summary["count"],
        "avg_latency_ms": f"{summary['avg_latency_ms']}ms" if summary["avg_latency_ms"] is not None else "없음",
        "error_rate": f"{summary['error_rate'] * 100:.1f}%" if summary["error_rate"] is not None else "없음",
    }


def build_prompt(facts: dict) -> str:
    template = DEFAULT_PROMPT
    if os.path.isfile(PROMPT_PATH):
        text = open(PROMPT_PATH, encoding="utf-8").read()
        if "TODO" not in text:  # A 가 채운 뒤부터 그 프롬프트를 쓴다
            template = text
    rule = (f'드리프트가 감지됐으므로 첫 줄은 반드시 "{LOW_CONFIDENCE}" 로 시작한다.'
            if facts["drift_detected"] else "")
    try:
        return template.format(low_confidence_rule=rule, **facts)
    except (KeyError, IndexError, ValueError):  # 프롬프트에 모르는 {키} 가 있으면 기본 프롬프트로
        return DEFAULT_PROMPT.format(low_confidence_rule=rule, **facts)


def call_llm(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 없음")
    r = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("LLM_MODEL", "gpt-4o-mini"), "temperature": 0.2,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _reason(e: Exception) -> str:
    """점주가 읽는 실패 사유 (개발자용 상세는 .json 의 error)."""
    if isinstance(e, requests.Timeout):
        return f"응답 {os.getenv('LLM_TIMEOUT', '30')}초 초과"
    if isinstance(e, requests.HTTPError):
        return "요약 서비스 오류"
    if isinstance(e, RuntimeError):
        return "요약 서비스 미설정"
    return "요약 서비스 연결 실패"


def template_report(facts: dict, reason: str) -> str:
    """LLM 실패 시 대체 보고서 — 숫자는 facts 그대로, 문장은 고정."""
    lines = []
    if facts["drift_detected"]:
        lines.append(f"**{LOW_CONFIDENCE}**\n")
    lines += [
        f"## {facts['item_name']} 시세 일일 보고서",
        f"- 내일 예측가: **{facts['predicted_price']}** (오늘 {facts['last_price']}, {facts['change_pct']})",
        f"- 예측 모델: {facts['model_version']}",
        f"- 최근 24시간 드리프트 감지 {facts['drift_count']}회 · 재학습 승격: {facts['promotions']}",
        f"- 서비스 상태: 예측 요청 {facts['request_count']}건 · 평균 응답 {facts['avg_latency_ms']} · 서버 에러율 {facts['error_rate']}",
        "",
        f"> 자동 문장 생성에 실패해 정해진 양식으로 만든 보고서입니다 ({reason}). 숫자는 같은 기록에서 나왔습니다.",
    ]
    return "\n".join(lines)


def generate() -> dict:
    facts = collect_facts()
    start = time.perf_counter()
    source, error = "llm", None
    try:
        markdown = call_llm(build_prompt(facts))
    except Exception as e:  # 키 없음 · 타임아웃 · HTTP 오류 · 응답 형식 — 전부 템플릿으로
        source, error = "template", f"{type(e).__name__}: {e}"[:200]
        markdown = template_report(facts, _reason(e))
    elapsed = round(time.perf_counter() - start, 2)

    meta = {"date": facts["generated_at"][:10], "generated_at": facts["generated_at"], "source": source,
            "error": error, "elapsed_s": elapsed, "facts": facts}
    os.makedirs(REPORT_DIR, exist_ok=True)
    base = os.path.join(REPORT_DIR, meta["date"])
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # D(민영은) 의 LLM 관측 — 구현되면 여기서 경고를 남긴다 (생성 실패 · 30초 초과 · 신뢰도 문구 누락)
    try:
        from report_sidecar.observe import check_report
        check_report(markdown, facts["drift_detected"], elapsed, source=source, error=error)
    except Exception:
        # 관측 실패가 보고서 생성 자체를 막지 않도록 사이드카 본문은 그대로 제공한다.
        pass
    return meta


def seconds_until(at: str, now: datetime | None = None) -> float:
    """다음 at(HH:MM, 한국 시간)까지 남은 초. 이미 지났으면 다음 날."""
    now = now or datetime.now(history_log.KST)
    h, m = map(int, at.split(":"))
    target = datetime.combine(now.date(), dtime(h, m), tzinfo=history_log.KST)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _run_once() -> None:
    m = generate()
    print(f"[report] {m['date']} source={m['source']} elapsed={m['elapsed_s']}s"
          + (f" error={m['error']}" if m["error"] else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="매일 이 시각(HH:MM, 한국 시간)에 생성. 시작할 때도 1회 생성")
    ap.add_argument("--every", type=float, default=0, help="(테스트용) 초 단위 반복")
    args = ap.parse_args()
    load_dotenv()
    _run_once()  # 시작 시 1회 — /report 가 404 로 비어 있지 않게
    while args.at or args.every > 0:
        wait = seconds_until(args.at) if args.at else args.every
        if args.at:
            print(f"[report] 다음 생성 {args.at} KST 까지 {wait / 3600:.1f}시간", flush=True)
        time.sleep(wait)
        _run_once()
