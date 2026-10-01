"""가격·반입량 CSV의 공용 검증 유틸리티."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from typing import TextIO


REQUIRED_COLUMNS = {"Date", "Close", "Volume"}


class DataValidationError(ValueError):
    """CSV가 학습 파이프라인의 입력 계약을 만족하지 않을 때 발생한다."""


def _parse_number(raw: str | None, *, field: str, row_number: int) -> float:
    value = (raw or "").strip().replace(",", "")
    if not value:
        raise DataValidationError(f"{row_number}행의 {field} 값이 비어 있습니다.")
    try:
        number = float(value)
    except ValueError as exc:
        raise DataValidationError(
            f"{row_number}행의 {field} 값이 숫자가 아닙니다: {raw!r}"
        ) from exc
    if not math.isfinite(number):
        raise DataValidationError(f"{row_number}행의 {field} 값은 유한한 숫자여야 합니다.")
    return number


def parse_price_volume_csv(source: TextIO, *, min_rows: int = 1) -> list[dict]:
    """열·날짜·값·정렬을 검증하고 정규화된 행을 반환한다."""

    reader = csv.DictReader(source)
    columns = set(reader.fieldnames or [])
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        raise DataValidationError(
            f"CSV에 {sorted(missing_columns)} 컬럼이 없습니다. "
            f"필수 컬럼: {sorted(REQUIRED_COLUMNS)}"
        )

    rows: list[dict] = []
    previous_date: date | None = None
    seen_dates: set[date] = set()

    for row_number, raw in enumerate(reader, start=2):
        raw_date = (raw.get("Date") or "").strip()
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise DataValidationError(
                f"{row_number}행의 Date는 YYYY-MM-DD 형식이어야 합니다: {raw_date!r}"
            ) from exc

        if parsed_date in seen_dates:
            raise DataValidationError(f"중복 날짜가 있습니다: {parsed_date.isoformat()}")
        if previous_date is not None and parsed_date <= previous_date:
            raise DataValidationError(
                f"Date는 오름차순이어야 합니다: {previous_date} 다음에 {parsed_date}"
            )

        close = _parse_number(raw.get("Close"), field="Close", row_number=row_number)
        volume = _parse_number(raw.get("Volume"), field="Volume", row_number=row_number)
        if close <= 0:
            raise DataValidationError(f"{row_number}행의 Close는 0보다 커야 합니다.")
        if volume < 0:
            raise DataValidationError(f"{row_number}행의 Volume은 0 이상이어야 합니다.")

        rows.append(
            {
                "Date": parsed_date.isoformat(),
                "Close": close,
                "Volume": volume,
            }
        )
        seen_dates.add(parsed_date)
        previous_date = parsed_date

    if len(rows) < min_rows:
        raise DataValidationError(
            f"최소 {min_rows}행 이상의 데이터가 필요합니다. 현재 {len(rows)}행입니다."
        )
    return rows


def load_price_volume_csv(csv_path: str | Path, *, min_rows: int = 1) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as source:
        return parse_price_volume_csv(source, min_rows=min_rows)
