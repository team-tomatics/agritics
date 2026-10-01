"""가격·반입량 행을 API 응답용 통계로 요약한다."""


def summarize_price_volume_rows(rows: list[dict]) -> dict:
    """검증을 마친 시간순 행의 기간·가격·반입량 범위를 반환한다."""
    if not rows:
        raise ValueError("요약할 데이터가 없습니다.")

    closes = [row["Close"] for row in rows]
    volumes = [row["Volume"] for row in rows]
    return {
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_close": min(closes),
        "max_close": max(closes),
        "price_unit": "원/kg",
        "min_volume": min(volumes),
        "max_volume": max(volumes),
        "avg_volume": sum(volumes) / len(volumes),
        "volume_unit": "kg",
    }
