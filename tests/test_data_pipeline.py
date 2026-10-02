import csv
import tempfile
import unittest
from datetime import date
from io import StringIO
from pathlib import Path

from data.summary import summarize_price_volume_rows
from data.validation import DataValidationError, parse_price_volume_csv
from scripts.fetch_kamis import (
    close_statistics,
    embedded_volume_status,
    merge_price_and_volume,
    read_daily_series,
    write_rows_atomic,
)


class DataValidationTests(unittest.TestCase):
    def test_valid_csv_is_normalized(self):
        rows = parse_price_volume_csv(
            StringIO("Date,Close,Volume\n2026-09-29,3,10\n2026-09-30,4,20\n"),
            min_rows=2,
        )
        self.assertEqual(rows[0], {"Date": "2026-09-29", "Close": 3.0, "Volume": 10.0})

    def test_duplicate_date_is_rejected(self):
        with self.assertRaisesRegex(DataValidationError, "중복 날짜"):
            parse_price_volume_csv(
                StringIO("Date,Close,Volume\n2026-09-29,3,10\n2026-09-29,4,20\n")
            )

    def test_negative_volume_is_rejected(self):
        with self.assertRaisesRegex(DataValidationError, "Volume은 0 이상"):
            parse_price_volume_csv(StringIO("Date,Close,Volume\n2026-09-29,3,-1\n"))

    def test_summary_includes_price_and_volume_units(self):
        summary = summarize_price_volume_rows(
            [
                {"Date": "2026-09-29", "Close": 3000.0, "Volume": 100.0},
                {"Date": "2026-09-30", "Close": 4000.0, "Volume": 300.0},
            ]
        )

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["start_date"], "2026-09-29")
        self.assertEqual(summary["end_date"], "2026-09-30")
        self.assertEqual(summary["min_close"], 3000.0)
        self.assertEqual(summary["max_close"], 4000.0)
        self.assertEqual(summary["price_unit"], "원/kg")
        self.assertEqual(summary["min_volume"], 100.0)
        self.assertEqual(summary["max_volume"], 300.0)
        self.assertEqual(summary["avg_volume"], 200.0)
        self.assertEqual(summary["volume_unit"], "kg")


class FetchKamisTests(unittest.TestCase):
    def test_merge_requires_volume_for_every_price_day(self):
        prices = {
            date(2026, 9, 29): 3.0,
            date(2026, 9, 30): 4.0,
        }
        volumes = {date(2026, 9, 29): 10.0}
        with self.assertRaisesRegex(DataValidationError, "반입량이 없는 날짜가 1개"):
            merge_price_and_volume(prices, volumes)

    def test_volume_rows_can_be_summed_by_date(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "volume.csv"
            source.write_text(
                "day,amount\n20260929,10\n20260929,15\n20260930,20\n", encoding="utf-8"
            )
            result = read_daily_series(
                source, date_column="day", value_column="amount", aggregate="sum"
            )
            self.assertEqual(list(result.values()), [25.0, 20.0])

    def test_atomic_output_has_expected_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tomato.csv"
            rows = [
                {"Date": "2026-09-29", "Close": 3.0, "Volume": 10.0},
                {"Date": "2026-09-30", "Close": 4.0, "Volume": 20.0},
            ]
            write_rows_atomic(rows, output)
            with output.open(encoding="utf-8") as source:
                self.assertEqual(list(csv.DictReader(source)), [
                    {"Date": "2026-09-29", "Close": "3.0", "Volume": "10.0"},
                    {"Date": "2026-09-30", "Close": "4.0", "Volume": "20.0"},
                ])

    def test_current_price_statistics_match_known_values(self):
        prices = read_daily_series(
            Path("data/tomato_prices.csv"), date_column="Date", value_column="Close"
        )
        stats = close_statistics(prices)
        self.assertEqual(stats["rows"], 903)
        self.assertEqual(stats["max_calendar_gap_days"], 5)
        self.assertAlmostEqual(stats["mean_close"], 3722.8704318936875)
        self.assertAlmostEqual(stats["naive_rmse"], 614.0589279232946)

    def test_current_volume_is_populated(self):
        status, zero_count = embedded_volume_status(
            Path("data/tomato_prices.csv"), date_column="Date"
        )
        self.assertEqual(status, "populated")
        self.assertEqual(zero_count, 0)


if __name__ == "__main__":
    unittest.main()
