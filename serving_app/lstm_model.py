"""
LSTM 아키텍처

작성자: 황재원 (원본: 교수님 스켈레톤)
버전: v0.2.0 (2026-10-02)
변경 이력:
  v0.1.0  —    교수님 스켈레톤 원본
  v0.2.0  #75  토마토 903거래일 · 입력 25 설명 (박유진)

토마토 다음 거래일 도매가격 예측용 LSTM 아키텍처 (Day1 baseline과 Day2 MLflow 학습이 공유).

토마토 3년치(903거래일) 데이터 + SEQ_LEN(25)을 적용하면 시퀀스가 878개(학습 80% 약 700개)라,
6개월/1층 구성 대비 파라미터 대비 샘플 비율이 충분합니다. 그래서 LSTM 3층
(32 -> 32 -> 16, 앞 두 층은 return_sequences=True로 다음 LSTM에 전체 시퀀스를 넘김) +
Dense 1층 구조를 택했습니다 - 이 정도 크기(파라미터 약 1.6만 개)는 CPU로 50 epoch을
학습해도 수십 초~1분 내외면 끝납니다.
"""
from tensorflow import keras

from data.features import SEQ_LEN

N_FEATURES = 2  # (close, volume)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
