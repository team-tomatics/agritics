"""
[Day1 → Day3] 예측 API  —  serving_app/routers/predict.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   외부 요청을 받아 모델에게 전달하고, 결과를 돌려주는 "창구"입니다.
   계산은 직접 하지 않고, 모델(model_loader)과 감시 도구(retrain_trigger)에게 맡깁니다.

■ 엔드포인트
   [Day1] POST /predict             : 25거래일 데이터 → 다음 거래일 도매가격 1개
   [Day3] POST /predict/batch-test  : 긴 가격 목록 → 여러 번 예측 → 드리프트 검사

■ 이 파일의 빈칸 : [빈칸 6]  (batch_test 의 슬라이딩 윈도우)
"""
from fastapi import APIRouter, Request

from data.features import SEQ_LEN  # = 25
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

# (Day3) 최근 예측 기록을 모아 두는 목록.  예: [{"predicted": 161.2, "actual": 163.0}, ...]
#        드리프트 판단은 "최근 WINDOW_SIZE(15)건"(drift_detector.py)만 보므로 그만큼만 유지합니다.
recent_predictions: list[dict] = []

# (Day3) 시뮬레이션은 가격만 보내므로 반입량은 고정값으로 채운다.
# 팀: HAIC 거래량 1,200,000 은 토마토 반입량(최대 약 36만 kg)의 3배라 스케일러 범위 밖 → 예측이 크게 틀어져
#     가짜 드리프트가 났다 (#49). 최근 업로드 CSV 의 반입량 중앙값을 쓴다 — 품목 · 데이터가 바뀌어도 학습 범위 안.
DEFAULT_SIMULATED_VOLUME = 136_449  # 토마토 903행 반입량 중앙값(kg) — 업로드가 없을 때만


def simulated_volume() -> float:
    try:
        import statistics

        from data.features import load_rows
        from data.storage import latest_upload

        return statistics.median(r["Volume"] for r in load_rows(latest_upload()))
    except Exception:  # noqa: BLE001 — 업로드 없음 · 읽기 실패여도 시뮬레이션은 돈다
        return DEFAULT_SIMULATED_VOLUME


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    """
    [Day1] 다음 거래일 도매가격 예측 (원/kg)
    받는 것  : {"sequence": [{"close": 4599.0, "volume": 158977}, ... 25개]}   (가격 원/kg · 반입량 kg, 오래된 날 → 최근 날)
               25개가 아니거나 가격 ≤ 0 · 반입량 < 0 이면 schemas.py 가 422 에러를 돌려줍니다.
    돌려줄 것: {"predicted_close": 3925.86, "model_version": "production-v1"}   (실제 2026-08-29 ~ 09-29 입력)

    흐름: 모델 가져오기(get_model) → dict 목록으로 변환 → predict_one → 응답 포장
    핵심 계산은 모두 model_loader.predict_one() 안에 있습니다. ([빈칸 2], [빈칸 3])
    """
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted_close = round(model.predict_one(sequence), 2)
    # 이력 로그 (history_log.py 미들웨어가 이 값을 한 줄에 합쳐 기록)
    request.state.history = {"model_version": model.version, "predicted": predicted_close, "input_last": sequence[-1]}
    return PredictResponse(predicted_close=predicted_close, model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest, request: Request):
    """
    [Day3] 드리프트 시뮬레이션
    받는 것  : {"prices": [2833.0, 2850.0, ... 40개]}   (원/kg, scripts/simulate_drift.py 가 보냄)
    돌려줄 것: {"predictions": [예측값 15개], "drift_check": {"status": "ok"} 또는 재학습 결과}
               반입량은 최근 업로드 CSV 의 중앙값으로 채운다 (#49)

    ■ 핵심 아이디어: 슬라이딩 윈도우 (25칸짜리 창문을 한 칸씩 밀기)
      가격 40개가 들어오면, 25개씩 잘라 "그다음 날"을 예측하고 실제 값과 비교합니다.

        i=0 : [p0  ~ p24] → 예측   vs  실제 p25
        i=1 : [p1  ~ p25] → 예측   vs  실제 p26
        ...
        i=14: [p14 ~ p38] → 예측   vs  실제 p39
        → 총 40 - 25 = 15번 예측 = 드리프트 판단에 필요한 15건이 딱 채워집니다 (612원/kg 기준).

    확인 방법
      python scripts/simulate_drift.py
        [normal]          drift_check = {'status': 'ok'}
        [drift_injection] drift_check = {'status': 'retrain_triggered', 'promoted': True 또는 False, 'rmse': ...}
      /docs 에서 직접 호출할 때는 predictions 가 (가격 개수 - 25)개인지 확인하세요.
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    prices = req.prices
    volume = simulated_volume()
    for i in range(len(prices) - SEQ_LEN):
        # ════════════════════════════ [빈칸 6] ════════════════════════════
        # i번째 창문(window)의 시작·끝 위치와, 그 창문 바로 다음 날(actual)의 위치를 채우세요. (i 와 SEQ_LEN 으로)
        #   (위 docstring 의 그림에서 i=0 일 때 무엇이 창문이고 무엇이 실제 값인지 먼저 확인)
        #
        #   생각해 볼 질문
        #     · 파이썬 슬라이싱 prices[a:b] 는 b 를 포함하나요?
        #     · 실제 값을 한 칸 앞(창문의 마지막 날)으로 잡으면, 모델은 무엇을 "맞힌" 셈이 될까요?
        #     · 반대로 창문을 한 칸 더 길게 잡아서 실제 값이 창문 안에 들어가면 RMSE는 어떻게 될까요?
        window = prices[i : i + SEQ_LEN]
        sequence = [{"close": p, "volume": volume} for p in window]
        pred = model.predict_one(sequence)
        actual = prices[i + SEQ_LEN]
        predictions.append(round(pred, 2))  # /predict 와 같은 소수 둘째 자리 (드리프트 RMSE 는 원값 recent_predictions 로 계산)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 WINDOW_SIZE(15)건만 남기기 — 오래된 기록까지 섞이면 "지금" 상태를 판단할 수 없습니다.
    # (recent_predictions = ... 로 쓰면 함수 안의 새 변수가 되므로, [:] 로 목록 내용을 바꿉니다)
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]

    # 드리프트 판단·재학습은 retrain_trigger.py 가 합니다. 여기서는 넘겨주기만!
    drift_check = check_and_trigger(recent_predictions)
    request.state.history = {"model_version": model.version, "predicted": round(predictions[-1], 2) if predictions else None,
                             "n_predictions": len(predictions), "drift_status": drift_check.get("status")}
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
