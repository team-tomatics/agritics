"""
게이트 지표 비교 — RMSE · MAE · MAPE · WAPE · Bias 를 모델 · 단순 예측 · 재학습 모델로 한 번에 (#102 검증용)

작성자: 박유진
버전: v1.1.0 (2026-10-02)
변경 이력:
  v1.0.0  #102  RMSE · MAPE 비교 (게이트 시험 구간 · 재학습 채점 구간)
  v1.1.0  #102  교수님 안내(케이스별 권장 지표)대로 — 배포 판정 · 회귀 테스트 = RMSE (현행) ·
                 명절 구간 포함 평가 = MAE + WAPE (명절 구간 오차 비중) · 품절 · 재고 방향 = Bias
                 (드리프트 = WAPE 는 scripts/drift_wape_threshold.py)

프로젝트 루트에서 (mlflow.db · scaler.pkl 이 있어야 함 — train_baseline_v1 · train_and_register 실행 후)
    python scripts/gate_metrics.py                         # data/tomato_prices.csv 기준
    python scripts/gate_metrics.py --csv data/uploads/xxx.csv

보여 주는 것
  1) 3년 전체 단순 예측("내일 = 오늘")      — 평시 기준선 (RMSE 612 의 근거)
  2) 게이트 시험 구간 (train_and_register 와 같은 80/20 분할) — 첫 배포 모델이 받는 점수
  3) 재학습 채점 구간 (마지막 RETRAIN_DAYS + SEQ_LEN 행의 시험 20%) — fine_tune 이 받는 점수
     · 현재 Production · 가장 최근 fine-tune 실행 모델(탈락했어도 MLflow 에 남아 있음) · 단순 예측
  명절 비중 = 설 · 추석 14일 전 ~ 7일 후가 제곱 오차(RMSE) · 절대 오차(MAE · WAPE) 에서 차지하는 몫
  Bias = Σ(예측 − 실제) / Σ실제 — 음수 과소예측(값이 실제보다 낮음 → 구매 부족) · 양수 과대예측
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import SEQ_LEN, PriceVolumeScaler, build_sequences, load_rows, train_test_split  # noqa: E402

SCALER_PATH = "serving_app/models/scaler.pkl"


def rmse(y, p):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))


def mape(y, p):
    return 100 * sum(abs(a - b) / a for a, b in zip(y, p)) / len(y)


def mae(y, p):
    return sum(abs(a - b) for a, b in zip(y, p)) / len(y)


def wape(y, p):
    """% — 오차 합 / 실제 가격 합. 날마다 나누는 MAPE 와 달리 싼 날 · 비싼 날이 금액만큼만 기여한다."""
    return 100 * sum(abs(a - b) for a, b in zip(y, p)) / sum(y)


def bias(y, p):
    """% — Σ(예측 − 실제) / Σ실제. 음수 = 과소예측, 양수 = 과대예측."""
    return 100 * sum(b - a for a, b in zip(y, p)) / sum(y)


# 명절 물량 변동 구간 = 설 · 추석 14일 전 ~ 7일 후 (도매가는 명절 전 반입 · 수요가 몰릴 때 뛴다)
HOLIDAYS = ["2023-09-29", "2024-02-10", "2024-09-17", "2025-01-29", "2025-10-06", "2026-02-17", "2026-09-25"]


def is_holiday(day: str) -> bool:
    from datetime import date

    d = date.fromisoformat(day)
    return any(-7 <= (date.fromisoformat(h) - d).days <= 14 for h in HOLIDAYS)


def _share(dates, y, p):
    """명절 구간이 오차에서 차지하는 비율 — RMSE(제곱 오차) vs MAE(절대 오차)."""
    hol = [is_holiday(d) for d in dates]
    sq = [(a - b) ** 2 for a, b in zip(y, p)]
    ab = [abs(a - b) for a, b in zip(y, p)]
    n = sum(hol)
    return (f"  명절 {n}/{len(y)}일({100 * n / len(y):.0f}%)이 차지하는 오차 — "
            f"제곱 오차(RMSE) {100 * sum(v for v, h in zip(sq, hol) if h) / sum(sq):.0f}% · "
            f"절대 오차(MAE·WAPE) {100 * sum(v for v, h in zip(ab, hol) if h) / sum(ab):.0f}%")


def _window(rows, scaler, model):
    """train_and_register._prepare 와 같은 분할로 시험 구간의 (날짜, 실제, 모델 예측, 단순 예측)."""
    import numpy as np

    X, y = build_sequences(rows, scaler)
    _, _, X_test, y_test = train_test_split(X, y)
    idx = list(range(len(rows) - SEQ_LEN))
    _, _, test_idx, _ = train_test_split(idx, idx)
    naive = [rows[i + SEQ_LEN - 1]["Close"] for i in test_idx]
    pred = None
    if model is not None:
        pred = [scaler.inverse_close(v) for v in model.predict(np.array(X_test, dtype="float32"), verbose=0).flatten()]
    span = f"{rows[test_idx[0] + SEQ_LEN]['Date']} ~ {rows[test_idx[-1] + SEQ_LEN]['Date']} ({len(test_idx)}일)"
    dates = [rows[i + SEQ_LEN]["Date"] for i in test_idx]
    return span, y_test, pred, naive, dates


def _line(label, y, p, gate=None):
    """gate = RMSE 게이트(원/kg) 면 판정을 붙인다 (회귀 테스트는 따로 — [3] 에서 Production 과 RMSE 비교)."""
    r = rmse(y, p)
    line = (f"  {label:<24} RMSE {r:7.1f} · MAE {mae(y, p):7.1f}원/kg · MAPE {mape(y, p):5.1f}% · "
            f"WAPE {wape(y, p):5.1f}% · Bias {bias(y, p):+5.1f}%")
    if gate:
        line += "  게이트 " + ("통과" if r <= gate else "탈락")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/tomato_prices.csv")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    a = ap.parse_args()

    import mlflow
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    from serving_app.monitoring.retrain_trigger import RETRAIN_DAYS
    import serving_app.train_and_register as tr

    mlflow.set_tracking_uri(a.tracking_uri)
    rows = load_rows(a.csv)
    scaler = PriceVolumeScaler.load(SCALER_PATH)
    closes = [r["Close"] for r in rows]
    gate = getattr(tr, "RMSE_GATE", None)
    print(f"데이터 {a.csv} · {len(rows)}행 ({rows[0]['Date']} ~ {rows[-1]['Date']})")
    print(f"게이트 RMSE ≤ {gate:.0f}원/kg" if gate else "게이트 판정 생략 — train_and_register 에 RMSE_GATE 없음",
          "· 재학습은 회귀 테스트(RMSE ≤ 현재 Production)도 — MAE · WAPE · Bias 는 평가용\n")

    print("[1] 전체 기간 단순 예측 (평시 기준선)")
    print(_line("단순 예측 내일 = 오늘", closes[1:], closes[:-1]))
    print(_share([r["Date"] for r in rows[1:]], closes[1:], closes[:-1]))

    prod = mlflow.tensorflow.load_model(f"models:/{tr.MODEL_NAME}/Production")
    span, y, pred, naive, dates = _window(rows, scaler, prod)
    print(f"\n[2] 게이트 시험 구간 {span}")
    print(_line("Production", y, pred, gate))
    print(_share(dates, y, pred))
    print(_line("단순 예측", y, naive, gate))

    recent = rows[-(RETRAIN_DAYS + SEQ_LEN):]
    span, y, pred, naive, _ = _window(recent, scaler, prod)
    print(f"\n[3] 재학습 채점 구간 {span}  (마지막 {len(recent)}행)")
    print(_line("현재 Production", y, pred))
    client = MlflowClient()
    runs = client.search_runs([e.experiment_id for e in client.search_experiments()],
                              "tags.mlflow.runName = 'fine-tune'", order_by=["attributes.start_time DESC"], max_results=1)
    if runs:
        ft = mlflow.tensorflow.load_model(f"runs:/{runs[0].info.run_id}/model")
        _, _, ft_pred, _, _ = _window(recent, scaler, ft)
        print(_line("최근 fine-tune 모델", y, ft_pred, gate))
        print(f"  회귀 테스트 RMSE {rmse(y, ft_pred):.1f} vs Production {rmse(y, pred):.1f} → "
              f"{'통과' if rmse(y, ft_pred) <= rmse(y, pred) else '탈락'}"
              f"  (명절 구간 평가 MAE {mae(y, ft_pred):.1f} vs {mae(y, pred):.1f} — "
              f"{'나아짐' if mae(y, ft_pred) <= mae(y, pred) else '나빠짐'})")
    else:
        print("  (fine-tune 실행 기록 없음 — 드리프트 재학습을 한 번 돌린 뒤 다시 실행)")
    print(_line("단순 예측", y, naive, gate))


if __name__ == "__main__":
    main()
