"""
시드 5개 실험 — 첫 배포 · 추석 fine-tune 의 RMSE · MAE · MAPE · WAPE (#102 게이트 값 근거, MLflow 등록 없음)

작성자: 박유진
버전: v1.0.0 (2026-10-02)
변경 이력:
  v1.0.0  #102  시드 42 · 1 · 7 · 123 · 2026 — 게이트 시험 구간 · 명절 비중 · 재학습 회귀 테스트(RMSE vs MAE)

프로젝트 루트에서 (scaler.pkl 필요 — train_baseline_v1 실행 후). 시드당 1분 남짓
    python scripts/seed_metrics.py
train_and_register 와 같은 분할 · 에포크 · fine-tune 설정이라 시드 42 는 실제 파이프라인 값(540.0 · 1258.4)과 같다.
시드는 고르지 않는다 — 5개 전부의 분포로 게이트 값을 정한다 (#45 합의).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

from data.features import SEQ_LEN, PriceVolumeScaler, load_rows  # noqa: E402
from scripts.gate_metrics import _share, mae, mape, rmse, wape  # noqa: E402
from serving_app.lstm_model import build_model  # noqa: E402
from serving_app.monitoring.retrain_trigger import RETRAIN_DAYS  # noqa: E402
import serving_app.train_and_register as tr  # noqa: E402

rows = load_rows("data/tomato_prices.csv")
scaler = PriceVolumeScaler.load(tr.SCALER_PATH)
recent = rows[-(RETRAIN_DAYS + SEQ_LEN):]
n_all = len(rows) - SEQ_LEN
dates = [rows[i + SEQ_LEN]["Date"] for i in range(n_all)][int(n_all * 0.8):]


def m4(y, p):
    return f"RMSE {rmse(y, p):7.1f} · MAE {mae(y, p):7.1f} · MAPE {mape(y, p):5.1f}% · WAPE {wape(y, p):5.1f}%"


def pred(model, X):
    return [scaler.inverse_close(v) for v in model.predict(np.array(X, dtype="float32"), verbose=0).flatten()]


for seed in [42, 1, 7, 123, 2026]:
    keras.utils.set_random_seed(seed)
    Xtr, ytr, Xte, yte = tr._prepare(rows, scaler)
    model = build_model()
    model.fit(Xtr, ytr, epochs=tr.BASE_EPOCHS, verbose=0)
    p = pred(model, Xte)
    print(f"seed {seed:<4} 첫 배포  {m4(yte, p)}")
    print(f"           {_share(dates, yte, p).strip()}")

    Xtr2, ytr2, Xte2, yte2 = tr._prepare(recent, scaler)
    before = pred(model, Xte2)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=tr.FINE_TUNE_LR), loss="mse")
    model.fit(Xtr2, ytr2, epochs=tr.FINE_TUNE_EPOCHS, verbose=0)
    after = pred(model, Xte2)
    print(f"           추석 기존  {m4(yte2, before)}")
    print(f"           추석 재학습 {m4(yte2, after)}")
    print(f"           회귀 RMSE {'통과' if rmse(yte2, after) <= rmse(yte2, before) else '탈락'} · "
          f"MAE {'통과' if mae(yte2, after) <= mae(yte2, before) else '탈락'}", flush=True)
