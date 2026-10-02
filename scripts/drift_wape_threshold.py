"""
드리프트 WAPE 임계값 실험 — 최근 15건 RMSE > 612 를 WAPE > X% 로 바꿀 때 X 근거 (#102)

작성자: 박유진
버전: v1.0.0 (2026-10-02)
변경 이력:
  v1.0.0  #102  급등 탐지 · 평시 오탐 (기획서 3-6 방식) · batch-test 시나리오 4개의 15건 WAPE

교수님 안내: 드리프트 탐지 · 재학습 트리거 → WAPE (물량 변동과 무관한 비율 기준). 배포 판정은 RMSE 유지.

프로젝트 루트에서 (mlflow.db 에 Production 이 있어야 함 — train_and_register 실행 후)
    python scripts/drift_wape_threshold.py

보여 주는 것
  1) 3년 전체를 하루씩 밀며 Production 으로 예측 → 날마다 "최근 15건" RMSE · WAPE
     급등 = 가격 > 직전 60거래일 중앙값 × 1.5, 20거래일 안이면 같은 급등 (기획서 3-6 과 같은 정의)
     탐지 = 급등 첫날 ~ 14거래일 뒤 사이에 임계값을 넘은 날이 있음 · 오탐 = 급등과 그 뒤 15일을 뺀 평시 날 중 넘은 비율
     ⚠ 앞 80% 는 학습 구간이라 오차가 작게 나온다 — 평시 오탐은 시험 구간(뒤 20%) 값도 같이 본다
  2) batch-test 와 똑같이(반입량 = 중앙값, 가격 40개 → 15건) 시나리오 4개의 15건 RMSE · WAPE
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data.features import SEQ_LEN, PriceVolumeScaler, load_rows  # noqa: E402
from scripts.gate_metrics import rmse, wape  # noqa: E402

WINDOW = 15
SCALER_PATH = "serving_app/models/scaler.pkl"
WAPE_CANDIDATES = [10, 12, 15, 18, 20, 25, 30]
RMSE_NOW = 612.0


def _predict(model, scaler, seqs):
    """seqs = [[{"Close", "Volume"} x 25], ...] → 원/kg 예측 목록 (model_loader.predict_one 과 같은 변환을 한 번에)."""
    X = np.array([[scaler.transform_point(r["Close"], r["Volume"]) for r in s] for s in seqs], dtype="float32")
    return [scaler.inverse_close(v) for v in model.predict(X, verbose=0).flatten()]


def _spikes(closes):
    flags = [i >= 60 and closes[i] > 1.5 * statistics.median(closes[i - 60:i]) for i in range(len(closes))]
    episodes = []
    for i, f in enumerate(flags):
        if f:
            if episodes and i - episodes[-1][1] <= 20:
                episodes[-1][1] = i
            else:
                episodes.append([i, i])
    return episodes


def main():
    import mlflow
    import mlflow.tensorflow

    from serving_app.train_and_register import MODEL_NAME

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    scaler = PriceVolumeScaler.load(SCALER_PATH)
    rows = load_rows("data/tomato_prices.csv")
    closes = [r["Close"] for r in rows]
    n = len(rows)

    # 1) 날마다 예측 (인덱스 t = 예측 대상 날)
    days = list(range(SEQ_LEN, n))
    pred = dict(zip(days, _predict(model, scaler, [rows[t - SEQ_LEN:t] for t in days])))
    naive = {t: closes[t - 1] for t in days}
    test_start = SEQ_LEN + int((n - SEQ_LEN) * 0.8)

    def window_metrics(p):
        out = {}
        for t in days[WINDOW - 1:]:
            ts = range(t - WINDOW + 1, t + 1)
            y, q = [closes[i] for i in ts], [p[i] for i in ts]
            out[t] = (rmse(y, q), wape(y, q))
        return out

    episodes = _spikes(closes)
    busy = set()
    for s, e in episodes:
        busy.update(range(s, e + WINDOW + 1))
    print(f"데이터 {n}행 ({rows[0]['Date']} ~ {rows[-1]['Date']}) · 급등 {len(episodes)}건 · 평시 판정일 기준 15건 창\n")
    for s, e in episodes:
        print(f"  급등 {rows[s]['Date']} ~ {rows[e]['Date']} · 최고 {max(closes[s:e + 1]):,.0f}원/kg")

    for label, p in [("Production LSTM", pred), ("단순 예측 (내일 = 오늘)", naive)]:
        wm = window_metrics(p)
        calm = [t for t in wm if t not in busy]
        calm_test = [t for t in calm if t >= test_start]
        print(f"\n[1] {label}")
        print(f"  평시 15건 WAPE 중앙값 {statistics.median(wm[t][1] for t in calm):.1f}% · "
              f"상위 5% {np.percentile([wm[t][1] for t in calm], 95):.1f}% "
              f"(시험 구간만 중앙값 {statistics.median(wm[t][1] for t in calm_test):.1f}%)")
        print(f"  {'기준':<16} 급등 탐지  평시 오탐(전체)  평시 오탐(시험 구간)")
        crits = [(f"RMSE > {RMSE_NOW:.0f} (지금)", 0, RMSE_NOW)] + [(f"WAPE > {x}%", 1, x) for x in WAPE_CANDIDATES]
        for name, k, thr in crits:
            hit = sum(any(wm.get(t, (0, 0))[k] > thr for t in range(s, s + 15)) for s, _ in episodes)
            fa = 100 * sum(wm[t][k] > thr for t in calm) / len(calm)
            fa_t = 100 * sum(wm[t][k] > thr for t in calm_test) / max(len(calm_test), 1)
            print(f"  {name:<16} {hit:>2}/{len(episodes)}건    {fa:5.1f}%          {fa_t:5.1f}%")

    # 2) batch-test 시나리오 — routers/predict.py batch_test 와 같은 계산
    vol = statistics.median(r["Volume"] for r in rows)
    from scripts.simulate_drift import DEFAULT_BASE_PRICE, generate_drift_batch, generate_normal_batch

    def by_date(start, count=SEQ_LEN + WINDOW):
        i = next(k for k, r in enumerate(rows) if r["Date"] >= start)
        return closes[i:i + count]

    scenarios = [
        ("평온 실제 2025-03-28~", by_date("2025-03-28")),
        ("추석 급등 실제 2026-08-13~", by_date("2026-08-13")),
        ("시뮬레이터 정상", list(generate_normal_batch(base=DEFAULT_BASE_PRICE))),
        ("시뮬레이터 폭염 (σ×4)", list(generate_drift_batch(base=DEFAULT_BASE_PRICE))),
    ]
    print(f"\n[2] batch-test 시나리오 (가격 40개 → 15건, 반입량 중앙값 {vol:,.0f}kg)")
    for label, prices in scenarios:
        seqs = [[{"Close": c, "Volume": vol} for c in prices[i:i + SEQ_LEN]] for i in range(len(prices) - SEQ_LEN)]
        q = _predict(model, scaler, seqs)
        y = prices[SEQ_LEN:]
        print(f"  {label:<24} RMSE {rmse(y, q):7.1f}원/kg · WAPE {wape(y, q):5.1f}%")


if __name__ == "__main__":
    main()
