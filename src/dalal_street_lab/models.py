"""The database tables, written as Python classes (SQLAlchemy ORM).

Each class is one table, and each attribute is one column. Alembic reads these
classes to write the migrations that actually create the tables.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Money is stored as an exact decimal, never a float: 0.1 + 0.2 must equal 0.3.
PRICE = Numeric(14, 4)

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
