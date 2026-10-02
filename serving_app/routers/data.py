"""토마토 가격·반입량 CSV 업로드와 최신 데이터 상태 조회.

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import io
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from data.summary import summarize_price_volume_rows
from data.validation import DataValidationError, parse_price_volume_csv
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    try:
        rows = parse_price_volume_csv(io.StringIO(text), min_rows=MIN_ROWS)
    except DataValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"tomato_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    return {
        "exists": True,
        "filename": os.path.basename(path),
        **summarize_price_volume_rows(rows),
    }
