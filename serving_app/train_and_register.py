"""
학습 · 게이트 · MLflow 등록 · fine-tune 재학습

작성자: 황재원 (원본: 교수님 스켈레톤)
버전: v0.9.0 (2026-10-02)
변경 이력:
  v0.1.0  —    교수님 스켈레톤 원본 (빈칸 채운 배포본)
  v0.2.0  #11  모델 이름 Tomato_Price_Predictor
  v0.3.0  #15  승격 시 이전 Production Archived + 회귀 테스트
  v0.4.0  #24  Production 채점을 학습 뒤로 — SEED 재현성
  v0.5.0  #48  게이트 탈락 시 exit 1 — Docker 빌드에서 바로 실패
  v0.6.0  #57  게이트 612원/kg (박유진)
  v0.7.0  #75  PriceVolumeScaler · 토마토 문구 (박유진)
  v0.8.0  #86  최초 Production 승격을 aiops.log 에 기록 (민영은)
  v0.9.0  #102 게이트에 MAPE 병행 — RMSE ≤ 612 OR MAPE ≤ 28% (회귀 테스트는 RMSE 유지)

Day2: MLflow로 토마토 시세 LSTM 모델을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

실습 시나리오 (94번 슬라이드를 LSTM 버전으로 재구성):
    1) 토마토 가격 · 반입량 데이터로 base 모델 학습 -> RMSE 확인 (게이트 미달 가능)
    2) 게이트(612원/kg) 통과 시 Production으로 승격
    3) (Day3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 1개월 데이터로
       10 epoch만 fine-tuning (처음부터 다시 학습하지 않음 - 21거래일로는 스크래치 학습이 불안정)

실행:
    (대시보드에서 시세 CSV를 먼저 업로드하세요 - data/tomato_prices.csv)
    python scripts/train_baseline_v1.py     # 최초 1회 (scaler.pkl 생성)
    python serving_app/train_and_register.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import load_rows, build_sequences, train_test_split, PriceVolumeScaler
from data.storage import latest_upload
from serving_app.lstm_model import build_model
from serving_app.monitoring.promotion_log import log_promotion

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 크게 흔들려
# (토마토 실측: 시드 42 · 1 · 7 · 123 · 2026 → 540.0 · 487.6 · 510.4 · 488.7 · 499.6원/kg, 기획서 3-6)
# 게이트(612원/kg)는 모두 통과하지만 단순 예측(529.9)보다 나은지는 시드에 좌우됩니다. numpy/tensorflow/python
# random을 한 번에 고정해 재현 가능한 학습 결과를 보장합니다.
SEED = 42
keras.utils.set_random_seed(SEED)

RMSE_GATE = 612.0  # 원/kg — 토마토 3년 단순 예측("내일 = 오늘") RMSE. 기획서 3-1 기본값 · 실험 후 확정 (#45)
# % — 가격이 뛴 구간용 (#102). 평시 Production MAPE 14.0% 의 2배. RMSE 와 OR 라 평시에도 느슨해지지만
# 재학습은 회귀 테스트(RMSE, 현재 Production 보다 나아야 함)가 막아 준다
MAPE_GATE = 28.0
MODEL_NAME = "Tomato_Price_Predictor"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 3층 LSTM + 3년치 데이터 기준, RMSE가 안정적으로 게이트 아래로 수렴하는 지점
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4  # base 학습(1e-3)보다 낮은 학습률로 살짝만 갱신


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    """% — scripts/gate_metrics.py 와 같은 정의."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)


def _prepare(rows: list[dict], scaler: PriceVolumeScaler):
    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_close(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _score(model, X_test, y_test, scaler: PriceVolumeScaler) -> tuple[float, float]:
    """(RMSE 원/kg, MAPE %)"""
    preds = [scaler.inverse_close(p) for p in model.predict(X_test, verbose=0).flatten()]
    return rmse(y_test, preds), mape(y_test, preds)


def _production_score(X_test, y_test, scaler: PriceVolumeScaler) -> tuple[float, float] | None:
    """회귀 테스트 기준: 현재 Production 을 새 모델과 같은 test 셋으로 채점. 첫 등록이면 None."""
    try:
        prod = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    except MlflowException:
        return None
    return _score(prod, X_test, y_test, scaler)


def _register_if_gate_passed(model, run_id: str, score: float, mape_score: float, prod_score: float | None) -> dict:
    """게이트 = RMSE ≤ RMSE_GATE OR MAPE ≤ MAPE_GATE (#102) · 회귀 테스트 = RMSE 가 현재 Production 이하"""
    result = {"run_id": run_id, "rmse": score, "mape": mape_score, "promoted": False}
    if score > RMSE_GATE and mape_score > MAPE_GATE:
        print(f"[GATE FAILED] rmse={score:.2f} > {RMSE_GATE} and mape={mape_score:.1f}% > {MAPE_GATE}% -> 배포 차단, 기존 Production 유지")
    elif prod_score is not None and score > prod_score:
        print(f"[GATE FAILED] rmse={score:.2f} > production rmse={prod_score:.2f} (회귀) -> 배포 차단, 기존 Production 유지")
    else:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        # 이전 Production 은 Archived 로 — Production 은 항상 1개, 롤백 대상은 가장 최근 Archived
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Production", archive_existing_versions=True
        )
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.2f} mape={mape_score:.1f}% -> {MODEL_NAME} v{v.version} promoted to Production")
    return result


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용합니다.

    csv_path를 지정하지 않으면 data/uploads/에 가장 최근 업로드된 CSV를 사용합니다
    (data/storage.py의 latest_upload() - 대시보드에서 업로드한 파일).
    """
    if rows is None:
        rows = load_rows(csv_path or latest_upload())
    scaler = PriceVolumeScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

        score, mape_score = _score(model, X_test, y_test, scaler)
        # Production 로드는 난수를 소비한다 — 학습 뒤에 해야 SEED 재현성이 Production 유무와 무관해진다 (#23)
        prod = _production_score(X_test, y_test, scaler)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("mape", mape_score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        prod_score = None
        if prod is not None:
            prod_score, prod_mape = prod
            mlflow.log_metric("prod_rmse", prod_score)
            mlflow.log_metric("prod_mape", prod_mape)
        result = _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, mape_score, prod_score)
        if result["promoted"]:
            log_promotion(result["rmse"], MODEL_NAME, result["version"])
        return result


def fine_tune(rows: list[dict]) -> dict:
    """
    Day3: 현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 rows(최근 데이터)로
    짧게 fine-tuning합니다. rows가 적을 때(예: 최근 1개월)도 스크래치 학습보다 훨씬 안정적입니다.
    """
    scaler = PriceVolumeScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    prod = _score(model, X_test, y_test, scaler)  # fine-tune 전 = 현재 Production

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)

        score, mape_score = _score(model, X_test, y_test, scaler)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("mape", mape_score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        prod_score = None
        if prod is not None:
            prod_score, prod_mape = prod
            mlflow.log_metric("prod_rmse", prod_score)
            mlflow.log_metric("prod_mape", prod_mape)
        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, mape_score, prod_score)


if __name__ == "__main__":
    # 게이트 탈락 = exit 1 -> Dockerfile 빌드가 여기서 실패한다 (모델 없는 컨테이너를 만들지 않음, #47)
    sys.exit(0 if train_and_register()["promoted"] else 1)
