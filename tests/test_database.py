"""The database must match our models, use TimescaleDB, and refuse bad data."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dalal_street_lab.models import Base, Candle, Instrument, Tick

WHEN = datetime(2026, 10, 22, 9, 15, tzinfo=UTC)


def make_instrument(session: Session) -> Instrument:
    instrument = Instrument(
        instrument_key="NSE_EQ|INE002A01018",
        symbol="RELIANCE",
        name="Reliance Industries",
        exchange="NSE",
        segment="NSE_EQ",
        instrument_type="EQ",
        tick_size=Decimal("0.05"),
    )
    session.add(instrument)
    session.flush()  # send it to the database now, so it gets an id
    return instrument


def make_candle(instrument_id: int, **changes: object) -> Candle:
    values: dict[str, object] = {
        "instrument_id": instrument_id,
        "interval": "1day",
        "time": WHEN,
        "open": Decimal("2850.10"),
        "high": Decimal("2875.00"),
        "low": Decimal("2840.55"),
        "close": Decimal("2861.30"),
        "volume": 5_400_000,
    }
    values.update(changes)
    return Candle(**values)


def test_price_tables_are_hypertables(engine: Engine) -> None:
    with engine.connect() as conn:
        names = conn.execute(
            text("SELECT hypertable_name FROM timescaledb_information.hypertables")
        ).scalars()
        assert set(names) == {"candles", "ticks"}


def test_migrations_match_models(engine: Engine) -> None:
    """If someone edits models.py but forgets a migration, this fails."""
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        differences = [
            diff
            for diff in compare_metadata(context, Base.metadata)
            if diff[0] != "remove_index"  # TimescaleDB's own time indexes
        ]
    assert differences == []


def test_save_and_read_candle_exactly(session: Session) -> None:
    instrument = make_instrument(session)
    session.add(make_candle(instrument.id))
    session.flush()

    saved = session.scalars(select(Candle).where(Candle.instrument_id == instrument.id)).one()
    assert saved.close == Decimal("2861.30")  # exact, no float rounding
    assert saved.time == WHEN


def test_save_tick(session: Session) -> None:
    instrument = make_instrument(session)
    session.add(Tick(instrument_id=instrument.id, time=WHEN, price=Decimal("2861.35")))
    session.flush()
    assert session.scalars(select(Tick.price)).one() == Decimal("2861.35")


@pytest.mark.parametrize(
    "bad_values",
    [
        {"high": Decimal("2800"), "low": Decimal("2900")},  # high below low
        {"close": Decimal("3000")},  # close above high
        {"open": Decimal("2700")},  # open below low
        {"volume": -1},  # negative volume
    ],
)
def test_database_rejects_impossible_candles(
    session: Session, bad_values: dict[str, object]
) -> None:
    instrument = make_instrument(session)
    session.add(make_candle(instrument.id, **bad_values))
    with pytest.raises(IntegrityError, match="ck_candles_"):
        session.flush()


def test_database_rejects_duplicate_candle(session: Session) -> None:
    instrument = make_instrument(session)
    session.add(make_candle(instrument.id))
    session.flush()
    session.add(make_candle(instrument.id))
    with pytest.raises(IntegrityError, match="pk_candles"):
        session.flush()


def test_migration_can_be_undone_and_redone(alembic_config: Config, engine: Engine) -> None:
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    test_price_tables_are_hypertables(engine)
