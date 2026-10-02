"""토마토 가격·반입량 CSV로 로컬 baseline LSTM과 스케일러를 만든다.

기본 동작은 대시보드에서 가장 최근에 업로드한 CSV를 사용한다. 업로드 없이
정식 학습 데이터를 직접 검증하려면 다음처럼 경로를 지정한다.

    python scripts/train_baseline_v1.py --csv data/tomato_prices.csv

입력은 ``Date,Close,Volume`` 형식이며, 최근 25거래일의 가격(원/kg)과
반입량(kg)으로 다음 거래일 가격을 예측한다. 현재 생성되는 ``haic_v1.keras``는
기존 로컬 모델 로더가 참조하는 임시 레거시 파일명이다. 토마토 모델명으로의 동시
교체는 B 소유 로더와 함께 #56에서 처리한다. 정식 배포 게이트 검증과 MLflow 등록은
``serving_app/train_and_register.py``가 담당한다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import SEQ_LEN, HAICScaler, build_sequences, load_rows, train_test_split
from data.storage import latest_upload

MODEL_PATH = "serving_app/models/haic_v1.keras"  # TODO(#56): B 로더와 함께 tomato_v1로 변경
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 전체 문제를 100번 반복해서 학습
SEED = 42
MIN_BASELINE_ROWS = SEQ_LEN + 2  # train/test 시퀀스를 각각 최소 1개 생성


def rmse(y_true, y_pred) -> float:
    """예측 가격이 실제 가격보다 평균적으로 얼마나 다른지 원/kg로 계산한다."""
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def main(csv_path: str | None = None) -> dict:
    import numpy as np
    from tensorflow import keras

    from serving_app.lstm_model import build_model

    keras.utils.set_random_seed(SEED)

    # STEP 1. 명시한 CSV 또는 가장 최근 업로드 CSV를 읽는다.
    source = csv_path or latest_upload()
    rows = load_rows(source)
    if len(rows) < MIN_BASELINE_ROWS:
        raise ValueError(
            f"baseline 학습에는 최소 {MIN_BASELINE_ROWS}행이 필요합니다. 현재 {len(rows)}행입니다."
        )
    print(f"data source: {source} ({len(rows)}행, seed={SEED})")

    # STEP 2. 스케일러 만들고 저장하기
    #   가격(원/kg)과 반입량(kg)의 크기가 다르므로 각각 0~1로 맞춘다.
    #   서버·MLflow 학습·재학습은 여기서 저장한 같은 스케일러를 사용한다.
    scaler = HAICScaler().fit(rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(rows)}행 -> {SCALER_PATH}")

    # STEP 3. 문제(X)와 정답(y) 만들기
    #   최근 25거래일의 (가격, 반입량)으로 다음 거래일 가격을 맞히는 문제를 만든다.
    #   X는 0~1로 변환되고 y는 원/kg 실값을 유지한다.
    X, y = build_sequences(rows, scaler)

    # STEP 4. 공부용 / 시험용 나누기
    #   주가는 시간 순서가 중요하므로 섞지 않고 "앞 80% 공부 / 뒤 20% 시험"으로 나눕니다.
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")

    # STEP 5. 학습용 정답도 입력과 같은 가격 스케일의 0~1 범위로 바꾼다.
    y_train_scaled = np.array([scaler.scale_close(v) for v in y_train], dtype="float32")

    # STEP 6. 시험 구간을 섞지 않고 학습 구간만 사용한다.
    model = build_model()
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    # STEP 7. 모델 출력을 원/kg로 복원해 시간순 마지막 20% 구간을 평가한다.
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = [scaler.inverse_close(p) for p in preds_scaled]
    score = rmse(y_test, preds)
    print(f"baseline v1 RMSE = {score:.2f}원/kg")

    # STEP 8. 모델 저장
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")
    return {"source": str(source), "rows": len(rows), "rmse": score, "seed": SEED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        help="학습할 Date,Close,Volume CSV. 생략하면 data/uploads의 최신 파일을 사용합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.csv)
