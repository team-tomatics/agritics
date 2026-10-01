"""KAMIS 토마토 가격을 점검하고 일별 반입량과 결합한다.

API 승인 전:
    python scripts/fetch_kamis.py analyze --prices data/tomato_prices.csv

반입량 CSV 확보 후:
    python scripts/fetch_kamis.py merge \
      --prices data/tomato_prices.csv \
      --volumes /path/to/volume.csv \
      --volume-date-column Date \
      --volume-column Volume \
      --output data/tomato_prices.csv

실제 OpenAPI 호출은 승인 후 응답 필드와 단위를 확인한 다음 추가한다. 승인 전에는
임의의 Volume 값을 본 데이터에 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validation import DataValidationError, load_price_volume_csv


DEFAULT_PRICE_CSV = Path("data/tomato_prices.csv")


def parse_date(raw: str, *, row_number: int, field: str) -> date:
    value = raw.strip()
    for parser in (date.fromisoformat, lambda text: date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")):
        try:
            return parser(value)
        except (ValueError, IndexError):
            continue
    raise DataValidationError(
        f"{row_number}행의 {field} 날짜 형식을 읽을 수 없습니다: {raw!r} "
        "(YYYY-MM-DD 또는 YYYYMMDD 필요)"
    )


def parse_number(raw: str | None, *, row_number: int, field: str) -> float:
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


def read_daily_series(
    csv_path: Path,
    *,
    date_column: str,
    value_column: str,
    aggregate: str = "error",
) -> dict[date, float]:
    """CSV의 날짜별 값을 읽는다. 반입량 원천의 여러 행은 sum으로 합칠 수 있다."""

    values: dict[date, list[float]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing = {date_column, value_column} - columns
        if missing:
            raise DataValidationError(
                f"{csv_path}에 {sorted(missing)} 컬럼이 없습니다. 현재 컬럼: {sorted(columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            parsed_date = parse_date(
                row.get(date_column, ""), row_number=row_number, field=date_column
            )
            number = parse_number(
                row.get(value_column), row_number=row_number, field=value_column
            )
            if number < 0:
                raise DataValidationError(
                    f"{row_number}행의 {value_column} 값은 0 이상이어야 합니다."
                )
            values[parsed_date].append(number)

    if not values:
        raise DataValidationError(f"{csv_path}에 데이터 행이 없습니다.")

    duplicates = [day for day, day_values in values.items() if len(day_values) > 1]
    if duplicates and aggregate == "error":
        preview = ", ".join(day.isoformat() for day in sorted(duplicates)[:5])
        raise DataValidationError(
            f"{csv_path}에 같은 날짜가 여러 번 있습니다: {preview}. "
            "일별 세부 행을 합쳐야 한다면 --volume-aggregate sum을 사용하세요."
        )

    return {
        day: sum(day_values) if aggregate == "sum" else day_values[0]
        for day, day_values in values.items()
    }


def close_statistics(prices: dict[date, float]) -> dict[str, float | int | str]:
    ordered = sorted(prices.items())
    closes = [value for _, value in ordered]
    gaps = [(ordered[index][0] - ordered[index - 1][0]).days for index in range(1, len(ordered))]
    changes = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
    squared_errors = [(closes[index] - closes[index - 1]) ** 2 for index in range(1, len(closes))]

    return {
        "rows": len(ordered),
        "start": ordered[0][0].isoformat(),
        "end": ordered[-1][0].isoformat(),
        "mean_close": statistics.fmean(closes),
        "min_close": min(closes),
        "max_close": max(closes),
        "max_calendar_gap_days": max(gaps, default=0),
        "naive_rmse": math.sqrt(statistics.fmean(squared_errors)) if squared_errors else 0.0,
        "mean_abs_daily_change_pct": statistics.fmean(abs(value) for value in changes) * 100
        if changes
        else 0.0,
        "daily_change_stddev_pct": statistics.pstdev(changes) * 100 if changes else 0.0,
    }


def print_statistics(stats: dict[str, float | int | str]) -> None:
    print(f"rows={stats['rows']}  period={stats['start']}..{stats['end']}")
    print(
        "close: "
        f"mean={stats['mean_close']:.2f}  min={stats['min_close']:.2f}  "
        f"max={stats['max_close']:.2f}"
    )
    print(f"max_calendar_gap_days={stats['max_calendar_gap_days']}")
    print(f"naive_rmse={stats['naive_rmse']:.2f}")
    print(f"mean_abs_daily_change_pct={stats['mean_abs_daily_change_pct']:.4f}")
    print(f"daily_change_stddev_pct={stats['daily_change_stddev_pct']:.4f}")


def merge_price_and_volume(
    prices: dict[date, float], volumes: dict[date, float]
) -> list[dict[str, str | float]]:
    missing = sorted(set(prices) - set(volumes))
    if missing:
        preview = ", ".join(day.isoformat() for day in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise DataValidationError(
            f"가격 거래일 중 반입량이 없는 날짜가 {len(missing)}개입니다: {preview}{suffix}"
        )

    return [
        {"Date": day.isoformat(), "Close": prices[day], "Volume": volumes[day]}
        for day in sorted(prices)
    ]


def write_rows_atomic(rows: list[dict[str, str | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=["Date", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(destination.name)

    try:
        load_price_volume_csv(temporary_path, min_rows=1)
        temporary_path.replace(output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def add_price_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_CSV)
    parser.add_argument("--price-date-column", default="Date")
    parser.add_argument("--price-column", default="Close")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="가격 CSV를 검증하고 기초 통계를 출력")
    add_price_arguments(analyze)

    merge = subparsers.add_parser("merge", help="가격과 반입량 CSV를 날짜 기준으로 결합")
    add_price_arguments(merge)
    merge.add_argument("--volumes", type=Path, required=True)
    merge.add_argument("--volume-date-column", default="Date")
    merge.add_argument("--volume-column", default="Volume")
    merge.add_argument(
        "--volume-aggregate",
        choices=("error", "sum"),
        default="error",
        help="반입량 원천에 날짜별 여러 행이 있을 때 합산할지 여부",
    )
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        prices = read_daily_series(
            args.prices,
            date_column=args.price_date_column,
            value_column=args.price_column,
        )
        print_statistics(close_statistics(prices))

        if args.command == "analyze":
            if args.prices == DEFAULT_PRICE_CSV:
                print("status=price-ready, volume-pending (원본 CSV는 변경하지 않음)")
            return 0

        volumes = read_daily_series(
            args.volumes,
            date_column=args.volume_date_column,
            value_column=args.volume_column,
            aggregate=args.volume_aggregate,
        )
        rows = merge_price_and_volume(prices, volumes)
        write_rows_atomic(rows, args.output)
        print(f"saved={args.output} rows={len(rows)}")
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
