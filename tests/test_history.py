"""Paging through Alpaca's bars, converting them, and saving them without duplicates."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from requests import PreparedRequest
from responses import matchers
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import dalal_street_lab.alpaca as alpaca_module
from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import Settings
from dalal_street_lab.history import (
    backfill,
    fetch_bars,
    parse_bars,
    resume_from,
    save_candles,
)
from dalal_street_lab.models import Candle, Instrument

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def bar(minute: int, close: float = 52.10) -> dict[str, Any]:
    return {
        "t": f"2026-10-22T13:{30 + minute:02d}:00Z",
        "o": 52.00,
        "h": max(52.20, close),
        "l": 51.95,
        "c": close,
        "v": 1200,
        "n": 14,
        "vw": 52.07,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AlpacaClient:
    monkeypatch.setattr(alpaca_module, "SECONDS_BETWEEN_REQUESTS", 0)
    return AlpacaClient(Settings(postgres_password="x", alpaca_api_key="k", alpaca_secret_key="s"))


def make_instrument(session: Session, symbol: str = "INDY") -> Instrument:
    instrument = Instrument(
        instrument_key=f"uuid-{symbol}",
        symbol=symbol,
        name="iShares India 50 ETF",
        exchange="NASDAQ",
        segment="us_equity",
        instrument_type="ETF",
        tick_size=Decimal("0.01"),
    )
    session.add(instrument)
    session.flush()
    return instrument


@responses.activate
def test_fetch_follows_every_page(client: AlpacaClient) -> None:
    seen_params: list[dict[str, list[str]]] = []

    def fake_alpaca(request: PreparedRequest) -> tuple[int, dict[str, str], str]:
        params = parse_qs(urlparse(request.url or "").query)
        seen_params.append(params)
        body: dict[str, Any]
        if "page_token" not in params:  # first request: 2 bars, and "there's more"
            body = {"bars": {"INDY": [bar(0), bar(1)]}, "next_page_token": "abc"}
        else:  # second request: the last bar, and no token = finished
            body = {"bars": {"INDY": [bar(2)]}, "next_page_token": None}
        return 200, {}, json.dumps(body)

    responses.add_callback(responses.GET, BARS_URL, callback=fake_alpaca)

    pages = list(fetch_bars(client, "INDY", "1minute", date(2026, 10, 22)))

    assert [len(page) for page in pages] == [2, 1]
    assert seen_params[1]["page_token"] == ["abc"]  # the token was passed back
    first = seen_params[0]
    assert first["timeframe"] == ["1Min"]
    assert first["feed"] == ["iex"]
    assert first["adjustment"] == ["split"]


@responses.activate
def test_fetch_never_asks_before_2016(client: AlpacaClient) -> None:
    responses.get(
        BARS_URL,
        json={"bars": {}, "next_page_token": None},
        match=[matchers.query_param_matcher({"start": "2016-01-01"}, strict_match=False)],
    )
    assert list(fetch_bars(client, "INDY", "1day", date(2005, 1, 1))) == [[]]


def test_parse_keeps_exact_prices_and_utc_time() -> None:
    row = parse_bars([bar(0)], instrument_id=1, interval="1minute")[0]
    assert row["time"] == datetime(2026, 10, 22, 13, 30, tzinfo=UTC)  # 9:30 AM New York
    assert row["close"] == Decimal("52.1")
    assert row["trade_count"] == 14
    assert row["vwap"] == Decimal("52.07")


def test_placeholder_bars_with_no_trades_are_skipped() -> None:
    # What Alpaca sent for INDY on 2022-07-12: the last price repeated, nothing traded.
    placeholder = {
        "t": "2022-07-12T04:00:00Z",
        "o": 41.55,
        "h": 41.55,
        "l": 41.55,
        "c": 41.55,
        "v": 0,
        "n": 0,
        "vw": 0,
    }
    rows = parse_bars([placeholder, bar(0)], instrument_id=1, interval="1day")
    assert len(rows) == 1  # only the real one
    assert rows[0]["trade_count"] == 14


def test_save_twice_does_not_duplicate_and_refreshes_values(session: Session) -> None:
    instrument = make_instrument(session)
    save_candles(session, parse_bars([bar(0), bar(1)], instrument.id, "1minute"))
    # The last minute was still in progress: it closed higher in the second download.
    save_candles(session, parse_bars([bar(0), bar(1, close=52.40)], instrument.id, "1minute"))

    assert session.scalar(select(func.count()).select_from(Candle)) == 2
    last = session.scalars(select(Candle).order_by(Candle.time.desc())).first()
    assert last is not None
    assert last.close == Decimal("52.4")


def test_resume_from_newest_saved_day(session: Session) -> None:
    instrument = make_instrument(session)
    default = date(2026, 1, 1)
    assert resume_from(session, instrument.id, "1minute", default) == default
    save_candles(session, parse_bars([bar(0)], instrument.id, "1minute"))
    assert resume_from(session, instrument.id, "1minute", default) == date(2026, 10, 22)


@responses.activate
def test_backfill_saves_every_active_instrument(session: Session, client: AlpacaClient) -> None:
    for symbol in ["INDY", "GLD"]:
        make_instrument(session, symbol)
        responses.get(
            BARS_URL,
            json={"bars": {symbol: [bar(0), bar(1)]}, "next_page_token": None},
            match=[matchers.query_param_matcher({"symbols": symbol}, strict_match=False)],
        )
    backfill(client, session, "1minute", days=5)
    assert session.scalar(select(func.count()).select_from(Candle)) == 4
