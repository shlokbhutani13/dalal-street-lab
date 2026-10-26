"""Check the candles: are they valid, and how complete are they?

Run it with:  uv run dalal-quality

Two kinds of checks:
  1. RULES (pandera): every single candle must make sense. Any failure = exit code 1.
  2. COVERAGE (market calendar): compared with the real US trading calendar, how many
     days/minutes are actually there? Low coverage isn't an error, it's a warning:
     it tells later steps which symbols are too thin to model minute by minute.
"""

import sys
import warnings
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal
import pandera.pandas as pa
import structlog
from pandera.typing import Series
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Candle, Instrument

log = structlog.get_logger()

# pandas_market_calendars still uses a pandas feature that pandas will remove one day.
# It's their code, not ours, and it works today, so we hide just that one message.
warnings.filterwarnings("ignore", message="Series.dt.weekday is deprecated")

NEW_YORK = "America/New_York"
TOLERANCE = 1e-6  # tiny allowance for rounding when comparing prices
BIG_DAILY_MOVE = 0.20  # a 20%+ change in one day is rare enough to look at by hand

# Coverage thresholds (share of expected trading days / minutes actually present)
DAILY_OK = 0.98
MINUTE_OK = 0.80
MINUTE_THIN = 0.40


# ---------- 1. RULES: what every single candle must satisfy ----------
class CandleSchema(pa.DataFrameModel):
    symbol: Series[str]
    time: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "us", "tz": "UTC"})
    open: Series[float] = pa.Field(gt=0)
    high: Series[float] = pa.Field(gt=0)
    low: Series[float] = pa.Field(gt=0)
    close: Series[float] = pa.Field(gt=0)
    volume: Series[int] = pa.Field(ge=0)
    trade_count: Series[float] = pa.Field(gt=0, nullable=True)  # a candle means trades happened
    vwap: Series[float] = pa.Field(gt=0, nullable=True)

    @pa.dataframe_check
    @classmethod
    def high_is_the_highest(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        return df["high"] + TOLERANCE >= df[["open", "close", "low"]].max(axis=1)

    @pa.dataframe_check
    @classmethod
    def low_is_the_lowest(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        return df["low"] - TOLERANCE <= df[["open", "close", "high"]].min(axis=1)

    @pa.dataframe_check
    @classmethod
    def vwap_is_inside_the_range(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        # The average price paid can't be outside the lowest and highest prices paid.
        inside = (df["vwap"] >= df["low"] - TOLERANCE) & (df["vwap"] <= df["high"] + TOLERANCE)
        return inside | df["vwap"].isna()

    class Config:
        unique = ["symbol", "time"]  # one candle per symbol per moment
        strict = True  # no unexpected columns


def load_candles(session: Session, interval: str) -> pd.DataFrame:
    """Read candles into a DataFrame. Prices become floats here, for analysis only:
    the database keeps the exact decimals."""
    query = (
        select(
            Instrument.symbol,
            Candle.time,
            Candle.open,
            Candle.high,
            Candle.low,
            Candle.close,
            Candle.volume,
            Candle.trade_count,
            Candle.vwap,
        )
        .join(Instrument, Instrument.id == Candle.instrument_id)
        .where(Candle.interval == interval, Instrument.is_active)
        .order_by(Instrument.symbol, Candle.time)
    )
    df = pd.DataFrame(session.execute(query).all(), columns=list(CandleSchema.to_schema().columns))
    for col in ["open", "high", "low", "close", "trade_count", "vwap"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype("int64")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.as_unit("us")
    return df


def rule_failures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (broken rule, bad candle). lazy=True collects ALL failures, not just the
    first; pandera reports a row once per column it looked at, so we de-duplicate."""
    try:
        CandleSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as errors:
        cases: pd.DataFrame = errors.failure_cases
        cases = cases.rename(columns={"index": "row"})[["check", "row"]]
        return cases.drop_duplicates().sort_values(["row", "check"]).reset_index(drop=True)
    return pd.DataFrame(columns=["check", "row"])


def big_daily_moves(df: pd.DataFrame) -> pd.DataFrame:
    """Days where the close moved 20%+ from the day before: real crashes, or bad data?"""
    df = df.sort_values(["symbol", "time"]).copy()
    df["change"] = df.groupby("symbol")["close"].pct_change()
    return df.loc[df["change"].abs() >= BIG_DAILY_MOVE, ["symbol", "time", "close", "change"]]


# ---------- 2. COVERAGE: compared with the real trading calendar ----------
def trading_sessions(start: date, end: date) -> pd.DataFrame:
    """Every NYSE trading day between two dates, with its open and close time (UTC).
    Knows holidays and half-days (like the day after Thanksgiving)."""
    schedule: pd.DataFrame = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return schedule


def ny_date(times: "pd.Series[pd.Timestamp]") -> "pd.Series[date]":
    return times.dt.tz_convert(NEW_YORK).dt.date


@dataclass
class Coverage:
    symbol: str
    expected: int  # trading days (daily) or regular-session minutes (minute)
    present: int  # how many we actually have inside regular hours
    extended_hours: int  # minute bars outside 9:30-16:00 (pre-market / after-hours)

    @property
    def share(self) -> float:
        return self.present / self.expected if self.expected else 0.0


def daily_coverage(group: pd.DataFrame) -> Coverage:
    days = set(ny_date(group["time"]))
    sessions = trading_sessions(min(days), max(days))
    expected = {d.date() for d in sessions.index}
    return Coverage(str(group["symbol"].iloc[0]), len(expected), len(days & expected), 0)


def minute_coverage(group: pd.DataFrame) -> Coverage:
    first, last = ny_date(group["time"]).min(), ny_date(group["time"]).max()
    sessions = trading_sessions(first, last)
    expected = int(
        ((sessions["market_close"] - sessions["market_open"]).dt.total_seconds() // 60).sum()
    )

    # A bar is "regular hours" if its date is a session and it starts in [open, close).
    bars = group[["time"]].copy()
    bars["day"] = pd.to_datetime(ny_date(bars["time"]))
    bars = bars.merge(sessions, left_on="day", right_index=True, how="left")
    regular = (bars["time"] >= bars["market_open"]) & (bars["time"] < bars["market_close"])
    return Coverage(
        str(group["symbol"].iloc[0]), expected, int(regular.sum()), int((~regular).sum())
    )


def verdict(interval: str, coverage: Coverage) -> str:
    if interval == "1day":
        return "OK" if coverage.share >= DAILY_OK else "GAPS"
    if coverage.share >= MINUTE_OK:
        return "OK"
    return "THIN" if coverage.share >= MINUTE_THIN else "TOO THIN"


def coverage_report(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    measure = daily_coverage if interval == "1day" else minute_coverage
    rows = []
    for _, group in df.groupby("symbol"):
        c = measure(group)
        rows.append(
            {
                "symbol": c.symbol,
                "interval": interval,
                "expected": c.expected,
                "present": c.present,
                "coverage": round(c.share, 3),
                "extended_hours": c.extended_hours,
                "verdict": verdict(interval, c),
            }
        )
    return pd.DataFrame(rows).sort_values("coverage", ascending=False)


def main() -> None:
    configure_logging()
    broken_rules = 0
    with new_session() as session:
        for interval in ["1day", "1minute"]:
            df = load_candles(session, interval)
            if df.empty:
                log.warning("no_candles", interval=interval)
                continue

            failures = rule_failures(df)
            broken_rules += len(failures)
            if not failures.empty:
                log.error("rules_broken", interval=interval, count=len(failures))
                print(failures["check"].value_counts().to_string())
                bad_rows = failures["row"].dropna().astype(int).unique()[:10]
                print(df.loc[df.index.intersection(list(bad_rows))].to_string())

            print(f"\n=== {interval}: {len(df):,} candles ===")
            print(coverage_report(df, interval).to_string(index=False))

            if interval == "1day":
                moves = big_daily_moves(df)
                if not moves.empty:
                    print(f"\nDaily moves of {BIG_DAILY_MOVE:.0%}+ (check these by hand):")
                    print(moves.to_string(index=False))

    log.info("quality_checked", broken_rules=broken_rules)
    sys.exit(1 if broken_rules else 0)  # broken rules stop pipelines; low coverage doesn't


if __name__ == "__main__":
    main()
