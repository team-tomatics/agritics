"""
드리프트 정책 백테스트 — RMSE 612 vs WAPE X% 중 어느 쪽이 실제 서비스 예측을 더 정확하게 만드나 (#102)

작성자: 박유진
버전: v1.0.0 (2026-10-02)
변경 이력:
  v1.0.0  #102  운영 재현(walk-forward): 예측 → 드리프트 판정 → fine-tune → 게이트 → 승격, 정책별 서비스 오차

프로젝트 루트에서 (scaler.pkl 필요). 정책 6개 × 시드 3개, 10분 남짓
    python scripts/drift_backtest.py

어떻게 돌리나 (지금 코드와 같은 규칙)
  - 시작 모델: 시작일 전 데이터로만 스크래치 학습 (BASE_EPOCHS, 시드 3개) — 평가 구간은 모델이 처음 보는 데이터
  - 매일: 현재 모델로 다음날 예측 → 실제와 비교해 최근 WINDOW_SIZE(15)건에 쌓음 → 정책으로 드리프트 판정
  - 드리프트: 그날까지의 최근 RETRAIN_DAYS + SEQ_LEN(50)행으로 fine-tune (FINE_TUNE_EPOCHS · FINE_TUNE_LR)
    → 게이트 RMSE ≤ RMSE_GATE · 회귀 RMSE ≤ 현재 모델 (같은 채점 구간) → 통과하면 교체 + 판정 윈도우 비움
  - 재학습이 탈락하면 5거래일 뒤에 다시 판정 (실서비스는 batch-test 호출마다 — 계산량 때문에 둔 가정, 정책 모두 같음)
결과: 정책별로 실제 서비스된 예측의 RMSE · MAE · WAPE · Bias (전체 · 명절 · 평시) · 재학습 시도 · 승격 횟수
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data.features import SEQ_LEN, PriceVolumeScaler, load_rows  # noqa: E402
from scripts.gate_metrics import bias, is_holiday, mae, rmse, wape  # noqa: E402

START = "2025-03-04"  # 평가 시작 — 앞 약 1년 5개월로 학습, 뒤 약 1년 7개월(추석 2번 · 설 1번 · 2025 · 2026 급등) 평가
SEEDS = [42, 1, 7]
RETRY_AFTER = 5
POLICIES = [
    ("재학습 안 함", None, None),
    ("RMSE > 612 (지금)", "rmse", 612.0),
    ("WAPE > 15%", "wape", 15.0),
    ("WAPE > 18%", "wape", 18.0),
    ("WAPE > 20%", "wape", 20.0),
    ("WAPE > 25%", "wape", 25.0),
]


def base_model(rows, scaler, start, seed):
    """시작일 전 데이터로 스크래치 학습 — train_and_register 와 같은 분할 · 에포크."""
    from tensorflow import keras

    import serving_app.train_and_register as tr
    from serving_app.lstm_model import build_model

    keras.utils.set_random_seed(seed)
    Xtr, ytr, _, _ = tr._prepare(rows[:start], scaler)
    model = build_model()
    model.fit(Xtr, ytr, epochs=tr.BASE_EPOCHS, verbose=0)
    return model


def _copy(model):
    from tensorflow import keras

    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    return m


def run(rows, scaler, start, seed, base, kind, thr):
    from tensorflow import keras

    import serving_app.train_and_register as tr
    from serving_app.monitoring.drift_detector import WINDOW_SIZE
    from serving_app.monitoring.retrain_trigger import RETRAIN_DAYS

    model = _copy(base)

    def predict_day(m, t):
        x = np.array([[scaler.transform_point(r["Close"], r["Volume"]) for r in rows[t - SEQ_LEN:t]]], dtype="float32")
        return scaler.inverse_close(m.predict(x, verbose=0)[0][0])

    served, recent = [], []
    attempts = promoted = 0
    next_check = start
    for t in range(start, len(rows)):
        p, a = predict_day(model, t), rows[t]["Close"]
        served.append((rows[t]["Date"], a, p))
        recent = (recent + [(a, p)])[-WINDOW_SIZE:]
        if kind is None or len(recent) < WINDOW_SIZE or t < next_check:
            continue
        y, q = [r[0] for r in recent], [r[1] for r in recent]
        if (rmse(y, q) if kind == "rmse" else wape(y, q)) <= thr:
            continue
        attempts += 1
        window_rows = rows[t + 1 - (RETRAIN_DAYS + SEQ_LEN):t + 1]  # 그날 실제 가격까지만
        Xf, yf, Xte, yte = tr._prepare(window_rows, scaler)
        prod_rmse = rmse(yte, [scaler.inverse_close(v) for v in model.predict(Xte, verbose=0).flatten()])
        cand = _copy(model)
        cand.compile(optimizer=keras.optimizers.Adam(learning_rate=tr.FINE_TUNE_LR), loss="mse")
        keras.utils.set_random_seed(seed)
        cand.fit(Xf, yf, epochs=tr.FINE_TUNE_EPOCHS, verbose=0)
        new_rmse = rmse(yte, [scaler.inverse_close(v) for v in cand.predict(Xte, verbose=0).flatten()])
        if new_rmse <= tr.RMSE_GATE and new_rmse <= prod_rmse:
            model, recent = cand, []
            promoted += 1
        else:
            next_check = t + RETRY_AFTER
    return served, attempts, promoted


def summary(served):
    def m(sel):
        y, p = [s[1] for s in sel], [s[2] for s in sel]
        return rmse(y, p), mae(y, p), wape(y, p), bias(y, p)

    hol = [s for s in served if is_holiday(s[0])]
    calm = [s for s in served if not is_holiday(s[0])]
    return m(served), m(hol), m(calm), len(hol)


def main():
    import serving_app.train_and_register as tr

    rows = load_rows("data/tomato_prices.csv")
    scaler = PriceVolumeScaler.load(tr.SCALER_PATH)
    start = next(i for i, r in enumerate(rows) if r["Date"] >= START)
    print(f"평가 {rows[start]['Date']} ~ {rows[-1]['Date']} ({len(rows) - start}거래일) · 시작 모델은 그 전 {start}행으로 학습 · 시드 {SEEDS}\n")

    table = {}
    bases = {seed: base_model(rows, scaler, start, seed) for seed in SEEDS}
    for name, kind, thr in POLICIES:
        for seed in SEEDS:
            served, att, pro = run(rows, scaler, start, seed, bases[seed], kind, thr)
            (r, a, w, b), (_, ha, hw, _), (_, ca, cw, _), nh = summary(served)
            table.setdefault(name, []).append((r, a, w, b, ha, hw, ca, cw, att, pro))
            print(f"  {name:<18} seed {seed:<3} RMSE {r:6.1f} · MAE {a:6.1f} · WAPE {w:5.2f}% · Bias {b:+5.1f}% · "
                  f"명절({nh}일) MAE {ha:6.1f} · 평시 MAE {ca:6.1f} · 재학습 {att}회 · 승격 {pro}회", flush=True)

    print("\n[평균 — 시드 3개]  서비스된 예측 기준 (낮을수록 좋음)")
    print(f"  {'정책':<18} {'RMSE':>7} {'MAE':>7} {'WAPE':>7} {'Bias':>7} {'명절 MAE':>9} {'평시 MAE':>9} {'재학습':>6} {'승격':>5}")
    for name, rs in table.items():
        avg = np.mean(rs, axis=0)
        print(f"  {name:<18} {avg[0]:7.1f} {avg[1]:7.1f} {avg[2]:6.2f}% {avg[3]:+6.1f}% {avg[4]:9.1f} {avg[6]:9.1f} {avg[8]:6.1f} {avg[9]:5.1f}")


if __name__ == "__main__":
    main()
