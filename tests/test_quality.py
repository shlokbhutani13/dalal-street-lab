"""Candle rules and calendar coverage, tested on Thanksgiving week 2026.

That week has a holiday (Thursday) AND a half-day (Friday): the tricky cases.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from dalal_street_lab.history import parse_bars, save_candles
from dalal_street_lab.models import Instrument
from dalal_street_lab.quality import (
    Coverage,
    big_daily_moves,
    daily_coverage,
    load_candles,
    minute_coverage,
    rule_failures,
    trading_sessions,
    verdict,
)


def candles(times: list[str], **columns: list[float]) -> pd.DataFrame:
    """A small, valid set of candles; pass columns to override values."""
    n = len(times)
    data: dict[str, object] = {
        "symbol": ["INDA"] * n,
        "time": pd.to_datetime(times, utc=True).as_unit("us"),
        "open": [50.0] * n,
        "high": [51.0] * n,
        "low": [49.0] * n,
        "close": [50.5] * n,
        "volume": [1000] * n,
        "trade_count": [12.0] * n,
        "vwap": [50.2] * n,
    }
    data.update(columns)
    return pd.DataFrame(data)


def minutes(day: str, start_utc: str, count: int) -> list[str]:
    stamps = pd.date_range(f"{day} {start_utc}", periods=count, freq="1min", tz="UTC")
    return [t.isoformat() for t in stamps]


# ---------- Rules ----------
def test_valid_candles_pass() -> None:
    assert rule_failures(candles(["2026-11-23T14:30Z", "2026-11-23T14:31Z"])).empty


@pytest.mark.parametrize(
    ("override", "broken_check"),
    [
        ({"high": [48.0]}, "high_is_the_highest"),  # high below low/open/close
        ({"low": [52.0]}, "low_is_the_lowest"),  # low above high/open/close
        ({"vwap": [60.0]}, "vwap_is_inside_the_range"),  # average price outside the range
        ({"close": [-1.0]}, "greater_than(0)"),  # impossible price
        ({"trade_count": [0.0], "vwap": [0.0]}, "greater_than(0)"),  # a "candle" with no trades
    ],
)
def test_impossible_candles_are_named(override: dict[str, list[float]], broken_check: str) -> None:
    failures = rule_failures(candles(["2026-11-23T14:30Z"], **override))
    assert broken_check in set(failures["check"])


def test_duplicate_candles_are_caught() -> None:
    failures = rule_failures(candles(["2026-11-23T14:30Z", "2026-11-23T14:30Z"]))
    assert set(failures["check"]) == {"multiple_fields_uniqueness"}


def test_big_daily_moves_are_flagged() -> None:
    df = candles(["2026-11-23", "2026-11-24", "2026-11-25"], close=[50.0, 51.0, 38.0])
    moves = big_daily_moves(df)
    assert list(moves["close"]) == [38.0]  # -25.5%, the only big one


# ---------- Calendar ----------
def test_calendar_knows_holidays_and_half_days() -> None:
    sessions = trading_sessions(date(2026, 11, 23), date(2026, 11, 27))
    days = [d.date() for d in sessions.index]
    assert date(2026, 11, 26) not in days  # Thanksgiving: closed
    friday = sessions.loc["2026-11-27"]
    assert (friday["market_close"] - friday["market_open"]).seconds == 210 * 60  # closes 1 PM


def test_daily_coverage_counts_missing_trading_days() -> None:
    # Mon, Wed, Fri present; Tue missing; Thu is Thanksgiving (not expected at all).
    df = candles(["2026-11-23T05:00Z", "2026-11-25T05:00Z", "2026-11-27T05:00Z"])
    coverage = daily_coverage(df)
    assert (coverage.expected, coverage.present) == (4, 3)


def test_minute_coverage_uses_real_session_lengths_and_spots_extended_hours() -> None:
    times = (
        minutes("2026-11-23", "12:00", 5)  # 7:00 AM New York: pre-market
        + minutes("2026-11-23", "14:30", 390)  # the full Monday session
        + minutes("2026-11-27", "14:30", 1)  # one bar on the Friday half-day
    )
    coverage = minute_coverage(candles(times))
    # Expected: Mon-Wed full (3 x 390) + Fri half-day (210) = 1380 minutes.
    assert coverage.expected == 1380
    assert coverage.present == 391
    assert coverage.extended_hours == 5


@pytest.mark.parametrize(
    ("interval", "present", "expected_verdict"),
    [
        ("1day", 99, "OK"),
        ("1day", 90, "GAPS"),
        ("1minute", 85, "OK"),
        ("1minute", 50, "THIN"),
        ("1minute", 5, "TOO THIN"),
    ],
)
def test_verdicts(interval: str, present: int, expected_verdict: str) -> None:
    assert verdict(interval, Coverage("X", 100, present, 0)) == expected_verdict


# ---------- From the real database ----------
def test_candles_from_the_database_pass_the_rules(session: Session) -> None:
    instrument = Instrument(
        instrument_key="uuid-INDA",
        symbol="INDA",
        name="iShares MSCI India ETF",
        exchange="BATS",
        segment="us_equity",
        instrument_type="ETF",
        tick_size=Decimal("0.01"),
    )
    session.add(instrument)
    session.flush()
    bar = {
        "t": "2026-11-23T14:30:00Z",
        "o": 50,
        "h": 51,
        "l": 49,
        "c": 50.5,
        "v": 9,
        "n": 2,
        "vw": 50.1,
    }
    save_candles(session, parse_bars([bar], instrument.id, "1minute"))

    df = load_candles(session, "1minute")
    assert len(df) == 1
    assert rule_failures(df).empty
