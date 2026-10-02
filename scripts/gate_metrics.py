"""
게이트 지표 비교 — RMSE · MAPE 를 모델 · 단순 예측 · 재학습 모델로 한 번에 (#102 MAPE 게이트 결정 · 검증용)

작성자: 박유진
버전: v1.0.0 (2026-10-02)
변경 이력:
  v1.0.0  #102  RMSE · MAPE 비교 (게이트 시험 구간 · 재학습 채점 구간)

프로젝트 루트에서 (mlflow.db · scaler.pkl 이 있어야 함 — train_baseline_v1 · train_and_register 실행 후)
    python scripts/gate_metrics.py                         # data/tomato_prices.csv 기준
    python scripts/gate_metrics.py --csv data/uploads/xxx.csv

보여 주는 것
  1) 3년 전체 단순 예측("내일 = 오늘")      — 평시 기준선 (RMSE 612 의 근거)
  2) 게이트 시험 구간 (train_and_register 와 같은 80/20 분할) — 첫 배포 모델이 받는 점수
  3) 재학습 채점 구간 (마지막 RETRAIN_DAYS + SEQ_LEN 행의 시험 20%) — fine_tune 이 받는 점수
     · 현재 Production · 가장 최근 fine-tune 실행 모델(탈락했어도 MLflow 에 남아 있음) · 단순 예측
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
    return span, y_test, pred, naive


def _line(label, y, p):
    return f"  {label:<28} RMSE {rmse(y, p):8.1f}원/kg · MAPE {mape(y, p):5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/tomato_prices.csv")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    a = ap.parse_args()

    import mlflow
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    from serving_app.monitoring.retrain_trigger import RETRAIN_DAYS
    from serving_app.train_and_register import MODEL_NAME

    mlflow.set_tracking_uri(a.tracking_uri)
    rows = load_rows(a.csv)
    scaler = PriceVolumeScaler.load(SCALER_PATH)
    closes = [r["Close"] for r in rows]
    print(f"데이터 {a.csv} · {len(rows)}행 ({rows[0]['Date']} ~ {rows[-1]['Date']})\n")

    print("[1] 전체 기간 단순 예측 (평시 기준선)")
    print(_line("단순 예측 내일 = 오늘", closes[1:], closes[:-1]))

    prod = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    span, y, pred, naive = _window(rows, scaler, prod)
    print(f"\n[2] 게이트 시험 구간 {span}")
    print(_line("Production", y, pred))
    print(_line("단순 예측", y, naive))

    recent = rows[-(RETRAIN_DAYS + SEQ_LEN):]
    span, y, pred, naive = _window(recent, scaler, prod)
    print(f"\n[3] 재학습 채점 구간 {span}  (마지막 {len(recent)}행)")
    print(_line("현재 Production", y, pred))
    client = MlflowClient()
    runs = client.search_runs([e.experiment_id for e in client.search_experiments()],
                              "tags.mlflow.runName = 'fine-tune'", order_by=["attributes.start_time DESC"], max_results=1)
    if runs:
        ft = mlflow.tensorflow.load_model(f"runs:/{runs[0].info.run_id}/model")
        _, _, ft_pred, _ = _window(recent, scaler, ft)
        print(_line("최근 fine-tune 모델", y, ft_pred))
    else:
        print("  (fine-tune 실행 기록 없음 — 드리프트 재학습을 한 번 돌린 뒤 다시 실행)")
    print(_line("단순 예측", y, naive))


if __name__ == "__main__":
    main()
