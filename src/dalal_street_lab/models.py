"""The database tables, written as Python classes (SQLAlchemy ORM).

Each class is one table, and each attribute is one column. Alembic reads these
classes to write the migrations that actually create the tables.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Money is stored as an exact decimal, never a float: 0.1 + 0.2 must equal 0.3.
PRICE = Numeric(14, 4)
SCORE = Numeric(5, 4)

# Predictable names for indexes and constraints, so migrations can find them later.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Instrument(Base):
    """Anything we watch or trade: a stock, an index, or a commodity future."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_key: Mapped[str] = mapped_column(String(64), unique=True)  # "NSE_EQ|INE002A01018"
    symbol: Mapped[str] = mapped_column(String(32))  # "RELIANCE", "NIFTY 50", "GOLD"
    name: Mapped[str] = mapped_column(String(128))
    exchange: Mapped[str] = mapped_column(String(8))  # NSE, BSE, MCX
    segment: Mapped[str] = mapped_column(String(16))  # NSE_EQ, NSE_INDEX, MCX_FO
    instrument_type: Mapped[str] = mapped_column(String(16))  # EQ, INDEX, FUT
    lot_size: Mapped[int] = mapped_column(default=1)
    tick_size: Mapped[Decimal] = mapped_column(PRICE)  # smallest price step, e.g. 0.05
    expiry: Mapped[date | None]  # futures expire; stocks and indices don't
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Candle(Base):
    """One bar of a candlestick chart: open, high, low, close and volume for a time window.

    This is a TimescaleDB hypertable: split into chunks by time, so queries like
    "the last 30 days of Reliance" only read the chunks they need.
    """

    __tablename__ = "candles"

    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    interval: Mapped[str] = mapped_column(String(16), primary_key=True)  # "1minute", "1day"
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[Decimal] = mapped_column(PRICE)
    high: Mapped[Decimal] = mapped_column(PRICE)
    low: Mapped[Decimal] = mapped_column(PRICE)
    close: Mapped[Decimal] = mapped_column(PRICE)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    open_interest: Mapped[int | None] = mapped_column(BigInteger)  # futures only
    trade_count: Mapped[int | None] = mapped_column(BigInteger)  # how many trades made this bar
    vwap: Mapped[Decimal | None] = mapped_column(PRICE)  # volume-weighted average price
    # The database itself refuses impossible candles, before any Python code is involved.
    __table_args__ = (
        CheckConstraint("low > 0", name="positive_prices"),
        CheckConstraint("high >= low", name="high_not_below_low"),
        CheckConstraint("open BETWEEN low AND high", name="open_within_range"),
        CheckConstraint("close BETWEEN low AND high", name="close_within_range"),
        CheckConstraint("volume >= 0", name="volume_not_negative"),
    )


class Tick(Base):
    """One live price update from the exchange. Also a hypertable (chunked by day)."""

    __tablename__ = "ticks"

    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    price: Mapped[Decimal] = mapped_column(PRICE)
    volume: Mapped[int | None] = mapped_column(BigInteger)  # total traded so far today
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (CheckConstraint("price > 0", name="positive_price"),)


class Article(Base):
    """One news article from Alpaca (written by Benzinga), plus FinBERT's mood for it.

    The sentiment columns start empty: downloading and scoring are separate jobs, so a
    slow AI model never holds up the download, and either can be re-run on its own.
    """

    __tablename__ = "news"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)  # Alpaca's
    headline: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32))  # "benzinga"
    author: Mapped[str] = mapped_column(String(256), default="")
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Filled in by FinBERT. Three probabilities that add up to 1, and one simple score.
    sentiment_model: Mapped[str | None] = mapped_column(String(64))  # which AI scored it
    positive: Mapped[Decimal | None] = mapped_column(SCORE)
    negative: Mapped[Decimal | None] = mapped_column(SCORE)
    neutral: Mapped[Decimal | None] = mapped_column(SCORE)
    sentiment: Mapped[Decimal | None] = mapped_column(SCORE)  # positive - negative: -1 to +1

    __table_args__ = (CheckConstraint("sentiment BETWEEN -1 AND 1", name="sentiment_in_range"),)


class ArticleSymbol(Base):
    """Which of our instruments an article is about. One article can be about several
    (a story about Indian IT can mention INFY and WIT), so this is its own table."""

    __tablename__ = "news_symbols"

    news_id: Mapped[int] = mapped_column(
        ForeignKey("news.id", ondelete="CASCADE"), primary_key=True
    )
    # Indexed, so "all news about INFY" is fast, not a scan of every row.
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), primary_key=True, index=True
    )


class FeatureRow(Base):
    """One row of model inputs: everything known about a symbol at the close of one day,
    plus the answer the model must learn to guess (tomorrow's move).

    Plain floats, not exact decimals: these are measurements for a model, not money.
    """

    __tablename__ = "features"

    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(16))

    ret_1d: Mapped[float] = mapped_column(Float)
    ret_5d: Mapped[float] = mapped_column(Float)
    ret_20d: Mapped[float] = mapped_column(Float)
    volatility_20d: Mapped[float] = mapped_column(Float)
    volume_ratio_20d: Mapped[float] = mapped_column(Float)
    rsi_14: Mapped[float] = mapped_column(Float)
    sma_gap_20d: Mapped[float] = mapped_column(Float)
    range_1d: Mapped[float] = mapped_column(Float)
    market_ret_1d: Mapped[float] = mapped_column(Float)
    news_count_1d: Mapped[float] = mapped_column(Float)
    news_mood_1d: Mapped[float] = mapped_column(Float)
    news_mood_5d: Mapped[float] = mapped_column(Float)

    # The answers. Empty for the newest day: tomorrow hasn't happened yet.
    target_ret_1d: Mapped[float | None] = mapped_column(Float)
    target_up: Mapped[bool | None]


class Prediction(Base):
    """One out-of-sample guess: "how likely is this symbol to go up the day after `day`?"

    Every row says which model made it and which MLflow run it came from, so any number
    on a chart can be traced back to the exact experiment that produced it.
    """

    __tablename__ = "predictions"

    model_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    prob_up: Mapped[float] = mapped_column(Float)
    feature_version: Mapped[str] = mapped_column(String(16))
    mlflow_run_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (CheckConstraint("prob_up BETWEEN 0 AND 1", name="prob_up_in_range"),)
