"""
반입량 민감도 — "반입량이 줄면 도매가가 오른다"를 데이터 · 모델로 확인 (#68)

작성자: 박유진
버전: v1.0.0 (2026-10-02)
변경 이력:
  v1.0.0  #68  요일 보정 데이터 분석 + /predict what-if

프로젝트 루트에서 (표준 라이브러리만 사용)
    python3 scripts/volume_sensitivity.py analyze                       # 1) 데이터: 반입량이 줄었던 날 → 다음날 가격
    python3 scripts/volume_sensitivity.py whatif --url http://localhost:8077  # 2) 모델: 반입량만 줄여 /predict

1) analyze — 가락시장 반입량은 요일 차이가 크다 (월요일 최다 · 목요일 최소). 그래서 "같은 요일 직전 평균"과 비교한다.
   요일을 무시하면 월요일 반입량 효과가 섞여 결론이 틀어진다.
2) whatif — 실제 25거래일 입력을 그대로 두고 반입량만 바꿔 예측가가 얼마나 움직이는지 본다.
   모델이 반입량을 실제로 쓰는지(피처 2개를 둔 이유) 확인하는 용도. 서버에 토마토 모델이 떠 있어야 한다.
"""
import argparse
import csv
import json
import statistics as st
import sys
import urllib.request
from datetime import date

CSV = "data/tomato_prices.csv"
SEQ_LEN = 25
WEEKDAY = "월화수목금토일"


def load(path=CSV):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return ([date.fromisoformat(r["Date"]) for r in rows], [float(r["Close"]) for r in rows],
            [float(r["Volume"]) for r in rows])


def analyze(drop=0.3, lookback=30):
    d, p, v = load()
    n = len(p)
    nxt = [(p[i + 1] / p[i] - 1) * 100 for i in range(n - 1)]
    base = nxt[lookback:]
    print(f"데이터 {d[0]} ~ {d[-1]} · {n}거래일")
    print(f"[전체] 다음날 가격 변화 평균 {st.mean(base):+.2f}% · 중앙 {st.median(base):+.2f}% · 오른 날 {sum(x > 0 for x in base) / len(base) * 100:.0f}%")
    print("[요일별 평균 반입량] " + " · ".join(
        f"{WEEKDAY[w]} {st.mean(v[i] for i in range(n) if d[i].weekday() == w):,.0f}kg" for w in range(6)))

    def same_weekday_mean(i):
        same = [v[j] for j in range(i - lookback, i) if d[j].weekday() == d[i].weekday()]
        return st.mean(same) if same else None

    for thr in (0.2, drop, 0.4):
        idx = [i for i in range(lookback, n - 1) if (m := same_weekday_mean(i)) and v[i] <= (1 - thr) * m]
        ch = [nxt[i] for i in idx]
        print(f"[같은 요일 평균보다 반입량 {thr * 100:.0f}%↓] {len(idx)}일 → 다음날 평균 {st.mean(ch):+.2f}% · "
              f"중앙 {st.median(ch):+.2f}% · 오른 날 {sum(x > 0 for x in ch) / len(ch) * 100:.0f}%")
    idx = [i for i in range(lookback, n - 1) if (m := same_weekday_mean(i)) and v[i] >= 1.3 * m]
    ch = [nxt[i] for i in idx]
    print(f"[참고: 같은 요일 평균보다 반입량 30%↑] {len(idx)}일 → 다음날 평균 {st.mean(ch):+.2f}% · "
          f"오른 날 {sum(x > 0 for x in ch) / len(ch) * 100:.0f}%")


def _predict(url, seq):
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=json.dumps({"sequence": seq}).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["predicted_close"]


def whatif(url, factor=0.7, last_days=3, windows=40):
    d, p, v = load()
    n = len(p)
    starts = list(range(n - SEQ_LEN - windows * 5, n - SEQ_LEN + 1, 5))[-windows:]  # 최근 구간에서 5일 간격
    effects, full = [], []
    for s in starts:
        seq = [{"close": p[i], "volume": int(v[i])} for i in range(s, s + SEQ_LEN)]
        base = _predict(url, seq)
        cut = [dict(x, volume=int(x["volume"] * factor)) if k >= SEQ_LEN - last_days else x for k, x in enumerate(seq)]
        allcut = [dict(x, volume=int(x["volume"] * factor)) for x in seq]
        effects.append((_predict(url, cut) / base - 1) * 100)
        full.append((_predict(url, allcut) / base - 1) * 100)
    change = f"{abs(1 - factor) * 100:.0f}%{'↓' if factor < 1 else '↑'}"
    print(f"모델 what-if — {len(starts)}개 입력 창 ({d[starts[0]]} ~ {d[starts[-1] + SEQ_LEN - 1]} 끝), 반입량 × {factor}")
    print(f"[최근 {last_days}일만 반입량 {change}] 예측가 변화 평균 {st.mean(effects):+.2f}% · "
          f"범위 {min(effects):+.2f} ~ {max(effects):+.2f}% · 오른 창 {sum(e > 0 for e in effects)}/{len(effects)}")
    print(f"[25일 전부 반입량 {change}] 예측가 변화 평균 {st.mean(full):+.2f}% · "
          f"범위 {min(full):+.2f} ~ {max(full):+.2f}% · 오른 창 {sum(e > 0 for e in full)}/{len(full)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analyze")
    w = sub.add_parser("whatif")
    w.add_argument("--url", default="http://localhost:8077")
    w.add_argument("--factor", type=float, default=0.7)
    w.add_argument("--last-days", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze()
    else:
        try:
            whatif(a.url, a.factor, a.last_days)
        except OSError as e:
            sys.exit(f"서버에 연결할 수 없습니다 ({a.url}): {e} — 토마토 모델로 서버를 먼저 띄우세요")
