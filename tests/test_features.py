"""Features: correct maths, gaps respected, and above all NO peeking at the future."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalal_street_lab.features import (
    FEATURES,
    FeatureSchema,
    build_features,
    closed_sessions,
    news_day,
    save_features,
)
from dalal_street_lab.models import FeatureRow, Instrument

FIRST = date(2020, 7, 27)


def new_york(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


def fake_prices(days: list[date], symbols: tuple[str, ...] = ("INDA", "INFY")) -> pd.DataFrame:
    """A random walk per symbol: realistic enough to exercise every feature."""
    rng = np.random.default_rng(7)
    rows = []
    for symbol in symbols:
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        for day, c in zip(days, close, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "day": day,
                    "open": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": int(rng.integers(1_000, 5_000)),
                }
            )
    return pd.DataFrame(rows)


def fake_news(days: list[date]) -> pd.DataFrame:
    """Two INFY articles every third day: one at lunchtime, one after the closing bell."""
    rows = []
    for day in days[::3]:
        rows.append({"symbol": "INFY", "published_at": new_york(f"{day} 12:00"), "sentiment": 0.6})
        rows.append({"symbol": "INFY", "published_at": new_york(f"{day} 17:00"), "sentiment": -0.4})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def days() -> list[date]:
    sessions = closed_sessions(FIRST, datetime(2020, 11, 1, tzinfo=UTC))
    return [d for d in sessions["day"] if d >= FIRST]


def close_of(day: date) -> datetime:
    sessions = closed_sessions(
        day, datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC)
    )
    return pd.Timestamp(sessions["market_close"].iloc[-1]).to_pydatetime()


# ---------- The rules that prevent look-ahead ----------
def test_news_counts_for_the_next_closing_bell() -> None:
    sessions = closed_sessions(date(2026, 11, 20), datetime(2026, 12, 1, tzinfo=UTC))
    news = pd.DataFrame(
        {
            "symbol": "INFY",
            "published_at": [
                new_york("2026-11-23 15:59"),  # Monday, before the bell: Monday
                new_york("2026-11-23 16:01"),  # Monday, after the bell: Tuesday
                new_york("2026-11-26 10:00"),  # Thanksgiving, market shut: Friday
                new_york("2026-11-27 13:30"),  # Friday half-day, closed at 1 PM: next Monday
                new_york("2026-11-28 09:00"),  # Saturday: Monday
            ],
            "sentiment": 0.5,
        }
    )
    assigned = news_day(news, sessions)["day"].tolist()
    assert assigned == [
        date(2026, 11, 23),
        date(2026, 11, 24),
        date(2026, 11, 27),
        date(2026, 11, 30),
        date(2026, 11, 30),
    ]


def test_todays_unfinished_bar_is_left_out() -> None:
    lunchtime = new_york("2026-10-26 12:00").to_pydatetime()
    evening = new_york("2026-10-26 18:00").to_pydatetime()
    assert closed_sessions(date(2026, 10, 19), lunchtime)["day"].iloc[-1] == date(2026, 10, 23)
    assert closed_sessions(date(2026, 10, 19), evening)["day"].iloc[-1] == date(2026, 10, 26)


def test_no_feature_ever_peeks_at_the_future(days: list[date]) -> None:
    """THE most important test. For several days D: work out the features again using ONLY
    data up to D's closing bell. If any feature for D changes, it was using the future."""
    prices, news = fake_prices(days), fake_news(days)
    everything = build_features(prices, news, now=close_of(days[-1]))

    for day in days[25::7]:
        known_then = build_features(
            prices[prices["day"] <= day],
            news[news["published_at"] <= pd.Timestamp(close_of(day))],
            now=close_of(day),
        )
        later = everything[everything["day"] == day].set_index("symbol")[FEATURES]
        then = known_then[known_then["day"] == day].set_index("symbol")[FEATURES]
        pd.testing.assert_frame_equal(later, then)

        # And on day D itself, tomorrow's answer is not known yet.
        assert known_then.loc[known_then["day"] == day, "target_ret_1d"].isna().all()


# ---------- The maths ----------
def test_returns_and_the_target(days: list[date]) -> None:
    prices = fake_prices(days)
    df = build_features(prices, fake_news(days), now=close_of(days[-1]))
    infy = df[df["symbol"] == "INFY"].set_index("day")
    close = prices[prices["symbol"] == "INFY"].set_index("day")["close"]

    d = infy.index[10]
    i = days.index(d)
    assert infy.loc[d, "ret_1d"] == pytest.approx(np.log(close[days[i]] / close[days[i - 1]]))
    assert infy.loc[d, "ret_5d"] == pytest.approx(np.log(close[days[i]] / close[days[i - 5]]))
    assert infy.loc[d, "target_ret_1d"] == pytest.approx(
        np.log(close[days[i + 1]] / close[days[i]])
    )
    assert infy.loc[d, "target_up"] == (close[days[i + 1]] > close[days[i]])


def test_rows_start_after_the_20_day_warm_up(days: list[date]) -> None:
    df = build_features(fake_prices(days), fake_news(days), now=close_of(days[-1]))
    assert df["day"].min() == days[20]  # ret_20d needs 20 earlier days
    assert df.groupby("symbol")["day"].max().tolist() == [days[-1], days[-1]]


def test_a_missing_day_is_a_gap_not_glued_over(days: list[date]) -> None:
    prices = fake_prices(days)
    missing = days[40]
    prices = prices[~((prices["symbol"] == "INFY") & (prices["day"] == missing))]
    df = build_features(prices, fake_news(days), now=close_of(days[-1]))
    infy_days = set(df.loc[df["symbol"] == "INFY", "day"])

    # Without that day, the 20-day features can't be trusted for the next 20 days.
    assert not infy_days & set(days[40:61])
    assert days[61] in infy_days


def test_news_mood_uses_the_safe_day(days: list[date]) -> None:
    df = build_features(fake_prices(days), fake_news(days), now=close_of(days[-1]))
    infy = df[df["symbol"] == "INFY"].set_index("day")
    d = next(day for day in days[::3] if day in infy.index)
    nxt = days[days.index(d) + 1]
    today, tomorrow = infy.loc[d], infy.loc[nxt]
    # Day D: only the lunchtime article (+0.6). Its evening article came after the bell...
    assert today["news_count_1d"] == 1
    assert today["news_mood_1d"] == pytest.approx(0.6)
    # ...so it counts for the next trading day instead.
    assert tomorrow["news_count_1d"] == 1
    assert tomorrow["news_mood_1d"] == pytest.approx(-0.4)


# ---------- Checking and saving ----------
def test_the_schema_accepts_real_output_and_rejects_nonsense(days: list[date]) -> None:
    df = build_features(fake_prices(days), fake_news(days), now=close_of(days[-1]))
    FeatureSchema.validate(df, lazy=True)

    broken = df.copy()
    broken.loc[0, "rsi_14"] = 140  # RSI can't go above 100
    with pytest.raises(pa.errors.SchemaErrors):
        FeatureSchema.validate(broken, lazy=True)


def test_saving_twice_replaces_instead_of_duplicating(session: Session, days: list[date]) -> None:
    for symbol in ["INDA", "INFY"]:
        session.add(
            Instrument(
                instrument_key=f"uuid-{symbol}",
                symbol=symbol,
                name=symbol,
                exchange="NYSE",
                segment="us_equity",
                instrument_type="ETF",
                tick_size=Decimal("0.01"),
            )
        )
    session.flush()
    df = build_features(fake_prices(days), fake_news(days), now=close_of(days[-1]))

    assert save_features(session, df) == len(df)
    assert save_features(session, df) == len(df)
    assert session.scalar(select(func.count()).select_from(FeatureRow)) == len(df)
    newest = session.scalars(select(FeatureRow).order_by(FeatureRow.day.desc())).first()
    assert newest is not None
    assert newest.target_up is None  # the last day's answer isn't known yet
