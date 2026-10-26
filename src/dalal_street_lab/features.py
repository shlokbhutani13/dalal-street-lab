"""Turn daily prices and news moods into features: the numbers a model learns from.

Run it with:  uv run dalal-features

One row per symbol per trading day. Every feature on day D uses ONLY what was known at
the closing bell of day D. The target (what the model must guess) is the NEXT day's move.
Peeking at the future by accident is called "look-ahead" or "leakage", and it's the most
common way trading models look brilliant in testing and then lose money for real.
"""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pandera.pandas as pa
import structlog
from pandera.typing import Series
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Article, ArticleSymbol, FeatureRow, Instrument
from dalal_street_lab.quality import load_candles, ny_date, trading_sessions

log = structlog.get_logger()

FEATURE_VERSION = "v1"  # bump this whenever a feature's meaning changes
FIRST_DAY = date(2020, 7, 27)  # free IEX history starts here (Step 5)
MARKET_SYMBOL = "INDA"  # the broad India ETF stands in for "how India did today"

FEATURES = [
    "ret_1d",  # today's move (log return)
    "ret_5d",  # this week's move
    "ret_20d",  # this month's move
    "volatility_20d",  # how jumpy the last 20 days were
    "volume_ratio_20d",  # today's volume vs the usual (log): 0 = normal, +0.7 = double
    "rsi_14",  # 0-100: share of the last 14 days' movement that was up
    "sma_gap_20d",  # how far above (+) or below (-) the 20-day average price
    "range_1d",  # today's high-low spread, as a share of the price
    "market_ret_1d",  # how the India market (INDA) moved today
    "news_count_1d",  # articles about this symbol today
    "news_mood_1d",  # their average FinBERT mood (0 when there was no news)
    "news_mood_5d",  # average mood over the last 5 trading days
]
TARGETS = ["target_ret_1d", "target_up"]


# ---------- Reading the inputs ----------
def daily_prices(session: Session) -> pd.DataFrame:
    df = load_candles(session, "1day")
    df["day"] = ny_date(df["time"])  # a daily bar is stamped midnight New York: its date
    return df[["symbol", "day", "open", "high", "low", "close", "volume"]]


def news_moods(session: Session) -> pd.DataFrame:
    rows = session.execute(
        select(Instrument.symbol, Article.published_at, Article.sentiment)
        .join(ArticleSymbol, ArticleSymbol.instrument_id == Instrument.id)
        .join(Article, Article.id == ArticleSymbol.news_id)
        .where(Instrument.is_active, Article.sentiment.is_not(None))
    ).all()
    df = pd.DataFrame(rows, columns=["symbol", "published_at", "sentiment"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.as_unit("us")
    df["sentiment"] = df["sentiment"].astype(float)
    return df


def closed_sessions(start: date, now: datetime) -> pd.DataFrame:
    """Trading days whose closing bell has already rung: ["day", "market_close"].

    Today only counts once the market has closed: before that, today's bar is unfinished.
    """
    s = trading_sessions(start, now.date())
    sessions = pd.DataFrame(
        {"day": pd.DatetimeIndex(s.index).date, "market_close": s["market_close"].to_numpy()}
    )
    return sessions[sessions["market_close"] <= pd.Timestamp(now)].reset_index(drop=True)


# ---------- The rules that prevent look-ahead ----------
def news_day(news: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Give each article to the first trading day whose closing bell comes AFTER it.

    News at 3:59 PM New York counts for today. News at 4:01 PM, on a weekend or on a
    holiday arrives too late to trade on today, so it counts for the next trading day.
    """
    if news.empty:
        return news.assign(day=pd.Series(dtype=object))
    news = news.sort_values("published_at")
    closes = sessions.sort_values("market_close")
    matched: pd.DataFrame = pd.merge_asof(
        news,
        closes,
        left_on="published_at",
        right_on="market_close",
        direction="forward",  # the next close at or after publication
    )
    matched = matched.dropna(subset=["day"])
    # The first day on the list has no "previous close" to start counting from, so older
    # articles would all pile onto it. It gets no news instead.
    matched = matched[matched["day"] != closes["day"].iloc[0]]
    return matched.drop(columns="market_close")


def symbol_features(prices: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """Price features for ONE symbol. Every calculation looks backwards only."""
    # Put every trading day in, even ones with no bar, so a missing day shows as a gap
    # (NaN) instead of silently gluing Monday to Wednesday.
    df = prices.set_index("day").reindex(days)
    df.index.name = "day"
    log_close = np.log(df["close"])

    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = log_close.diff(1)
    out["ret_5d"] = log_close.diff(5)
    out["ret_20d"] = log_close.diff(20)
    # min_periods = window: one missing day inside the window makes the feature NaN.
    out["volatility_20d"] = out["ret_1d"].rolling(20, min_periods=20).std()
    average_volume = df["volume"].rolling(20, min_periods=20).mean()
    out["volume_ratio_20d"] = np.log((df["volume"] + 1) / (average_volume + 1))
    ups = out["ret_1d"].clip(lower=0).rolling(14, min_periods=14).sum()
    downs = (-out["ret_1d"]).clip(lower=0).rolling(14, min_periods=14).sum()
    out["rsi_14"] = (100 * ups / (ups + downs)).where(ups + downs > 0, 50.0)
    out.loc[ups.isna(), "rsi_14"] = np.nan
    out["sma_gap_20d"] = df["close"] / df["close"].rolling(20, min_periods=20).mean() - 1
    out["range_1d"] = (df["high"] - df["low"]) / df["close"]

    # The TARGET is the only thing allowed to look forward: tomorrow's move.
    out["target_ret_1d"] = out["ret_1d"].shift(-1)
    return out.reset_index()


def build_features(prices: pd.DataFrame, news: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """All features for all symbols, as they were known at `now`. A pure function: same
    inputs, same output, no database. That's what makes it easy to test."""
    # A few extra days before FIRST_DAY, so news from the weekend before has a home.
    sessions = closed_sessions(FIRST_DAY - timedelta(days=10), now)
    days = [d for d in sessions["day"] if d >= FIRST_DAY]

    parts = []
    for symbol, group in prices.groupby("symbol"):
        part = symbol_features(group, days)
        part["symbol"] = symbol
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    # The market's move today, next to every symbol's own move.
    market = df.loc[df["symbol"] == MARKET_SYMBOL, ["day", "ret_1d"]]
    df = df.merge(market.rename(columns={"ret_1d": "market_ret_1d"}), on="day", how="left")

    # News: count and average mood per symbol per (look-ahead-safe) day.
    daily_news = (
        news_day(news, sessions)
        .groupby(["symbol", "day"])["sentiment"]
        .agg(news_count_1d="count", news_mood_sum="sum")
        .reset_index()
    )
    df = df.merge(daily_news, on=["symbol", "day"], how="left")
    df[["news_count_1d", "news_mood_sum"]] = df[["news_count_1d", "news_mood_sum"]].fillna(0)
    df["news_mood_1d"] = (df["news_mood_sum"] / df["news_count_1d"]).where(
        df["news_count_1d"] > 0, 0.0
    )
    df = df.sort_values(["symbol", "day"])
    by_symbol = df.groupby("symbol")
    count_5d = by_symbol["news_count_1d"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    mood_5d = by_symbol["news_mood_sum"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    df["news_mood_5d"] = (mood_5d / count_5d).where(count_5d > 0, 0.0)

    df["target_up"] = (df["target_ret_1d"] > 0).astype("boolean")
    df.loc[df["target_ret_1d"].isna(), "target_up"] = pd.NA

    # Keep only rows where every feature could be worked out (no warm-up, no gaps).
    ready = df.dropna(subset=FEATURES)
    return ready[["symbol", "day", *FEATURES, *TARGETS]].reset_index(drop=True)


# ---------- Checking the output ----------
class FeatureSchema(pa.DataFrameModel):
    symbol: Series[str]
    day: Series[object]  # datetime.date
    ret_1d: Series[float] = pa.Field(ge=-1.5, le=1.5)  # a log return beyond +-1.5 is a 4x move
    ret_5d: Series[float]
    ret_20d: Series[float]
    volatility_20d: Series[float] = pa.Field(ge=0)
    volume_ratio_20d: Series[float]
    rsi_14: Series[float] = pa.Field(ge=0, le=100)
    sma_gap_20d: Series[float] = pa.Field(gt=-1)
    range_1d: Series[float] = pa.Field(ge=0)
    market_ret_1d: Series[float]
    news_count_1d: Series[float] = pa.Field(ge=0)
    news_mood_1d: Series[float] = pa.Field(ge=-1, le=1)
    news_mood_5d: Series[float] = pa.Field(ge=-1, le=1)
    target_ret_1d: Series[float] = pa.Field(nullable=True)  # unknown for the latest day
    target_up: Series[pd.BooleanDtype] = pa.Field(nullable=True)

    @pa.dataframe_check
    @classmethod
    def target_up_matches_the_return(cls, df: pd.DataFrame) -> "pd.Series[bool]":
        known = df["target_ret_1d"].notna()
        return ~known | (df["target_up"].fillna(False) == (df["target_ret_1d"] > 0))

    class Config:
        unique = ["symbol", "day"]
        strict = True


# ---------- Saving ----------
def save_features(session: Session, df: pd.DataFrame) -> int:
    """Features are calculated, not collected, so we rebuild them all instead of patching:
    delete this version's rows and insert the fresh ones, in one transaction."""
    ids = {symbol: i for symbol, i in session.execute(select(Instrument.symbol, Instrument.id))}
    records = [
        {
            "instrument_id": ids[row["symbol"]],
            "day": row["day"],
            "feature_version": FEATURE_VERSION,
            **{col: float(row[col]) for col in FEATURES},
            "target_ret_1d": None if pd.isna(row["target_ret_1d"]) else float(row["target_ret_1d"]),
            "target_up": None if pd.isna(row["target_up"]) else bool(row["target_up"]),
        }
        for row in df.to_dict("records")
    ]
    session.execute(delete(FeatureRow).where(FeatureRow.feature_version == FEATURE_VERSION))
    for start in range(0, len(records), 2000):
        session.execute(insert(FeatureRow).values(records[start : start + 2000]))
    session.commit()
    return len(records)


def main() -> None:
    configure_logging()
    with new_session() as session:
        prices = daily_prices(session)
        if prices.empty:
            raise SystemExit("No daily candles. Run: uv run dalal-backfill")
        news = news_moods(session)
        features = build_features(prices, news, now=datetime.now(UTC))
        FeatureSchema.validate(features, lazy=True)
        saved = save_features(session, features)

    log.info("features_saved", rows=saved, version=FEATURE_VERSION)
    summary = features.groupby("symbol").agg(
        rows=("day", "size"),
        first=("day", "min"),
        days_with_news=("news_count_1d", lambda s: int((s > 0).sum())),
        up_days=("target_up", "mean"),
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
