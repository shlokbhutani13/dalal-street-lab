"""Download past price bars from Alpaca into the candles table.

Run it with:  uv run dalal-backfill                      (daily since 2016 + 1-minute for 90 days)
              uv run dalal-backfill --interval 1minute --days 30
"""

import argparse
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Candle, Instrument

log = structlog.get_logger()

# Our interval names -> what Alpaca calls them. Daily comes first: it's the quick one.
TIMEFRAMES = {"1day": "1Day", "1minute": "1Min"}
EARLIEST = date(2016, 1, 1)  # free accounts have data from 2016 onwards
DEFAULT_DAYS = {"1day": 3650, "1minute": 90}
PAGE_SIZE = 10_000  # the most bars Alpaca returns per request


def fetch_bars(
    client: AlpacaClient, symbol: str, interval: str, start: date
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of bars. Alpaca splits big answers into pages linked by next_page_token."""
    params: dict[str, Any] = {
        "symbols": symbol,
        "timeframe": TIMEFRAMES[interval],
        "start": max(start, EARLIEST).isoformat(),
        "end": datetime.now(UTC).isoformat(timespec="seconds"),
        "limit": PAGE_SIZE,
        "feed": client.feed,
        # Splits are adjusted, so a 2-for-1 split doesn't look like a 50% crash.
        "adjustment": "split",
    }
    while True:
        page = client.get(f"{client.data_url}/v2/stocks/bars", params=params)
        yield page["bars"].get(symbol, [])
        token = page.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def parse_bars(
    bars: list[dict[str, Any]], instrument_id: int, interval: str
) -> list[dict[str, Any]]:
    """Alpaca's short names: t=time o=open h=high l=low c=close v=volume n=trades vw=avg price.

    Bars with zero trades are skipped. On days when nobody traded a symbol on IEX, Alpaca
    still sends a "placeholder" bar: the last price repeated, volume 0, vwap 0. No trade
    happened, so it isn't a real price, and saving it would hide the gap.
    """
    return [
        {
            "instrument_id": instrument_id,
            "interval": interval,
            "time": datetime.fromisoformat(bar["t"]),  # "2026-10-22T13:30:00Z" (UTC)
            "open": Decimal(str(bar["o"])),
            "high": Decimal(str(bar["h"])),
            "low": Decimal(str(bar["l"])),
            "close": Decimal(str(bar["c"])),
            "volume": int(bar["v"]),
            "trade_count": int(bar["n"]),
            "vwap": Decimal(str(bar["vw"])),
        }
        for bar in bars
        if bar["n"] > 0 and bar["v"] > 0
    ]


UPDATABLE = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


def save_candles(session: Session, rows: list[dict[str, Any]]) -> None:
    """Upsert: new candles are added; candles we already have are refreshed."""
    for batch_start in range(0, len(rows), 1000):
        statement = insert(Candle).values(rows[batch_start : batch_start + 1000])
        statement = statement.on_conflict_do_update(
            index_elements=[Candle.instrument_id, Candle.interval, Candle.time],
            set_={col: statement.excluded[col] for col in UPDATABLE},
        )
        session.execute(statement)
    session.commit()


def resume_from(session: Session, instrument_id: int, interval: str, default: date) -> date:
    """Only download what's missing: start from the newest candle we already have."""
    newest = session.scalar(
        select(func.max(Candle.time)).where(
            Candle.instrument_id == instrument_id, Candle.interval == interval
        )
    )
    if newest is None:
        return default
    return max(default, newest.astimezone(UTC).date())  # re-fetch that day: it may be partial


def backfill(client: AlpacaClient, session: Session, interval: str, days: int) -> None:
    today = datetime.now(UTC).date()
    instruments = session.scalars(select(Instrument).where(Instrument.is_active)).all()
    if not instruments:
        raise SystemExit("No instruments in the database. Run: uv run dalal-instruments")

    for instrument in instruments:
        start = resume_from(session, instrument.id, interval, today - timedelta(days=days))
        saved = 0
        for page in fetch_bars(client, instrument.symbol, interval, start):
            rows = parse_bars(page, instrument.id, interval)
            save_candles(session, rows)
            saved += len(rows)
        log.info("backfilled", symbol=instrument.symbol, interval=interval, candles=saved)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Download past price bars from Alpaca.")
    parser.add_argument("--interval", choices=[*TIMEFRAMES, "all"], default="all")
    parser.add_argument("--days", type=int, help="how far back (default: 3650 daily, 90 minute)")
    args = parser.parse_args()

    intervals = list(TIMEFRAMES) if args.interval == "all" else [args.interval]
    client = AlpacaClient(get_settings())
    with new_session() as session:
        for interval in intervals:
            backfill(client, session, interval, args.days or DEFAULT_DAYS[interval])


if __name__ == "__main__":
    main()
