"""Looking up and saving the watchlist, tested with a fake Alpaca (no internet needed)."""

import time
from pathlib import Path
from typing import Any

import pytest
import responses
from responses import matchers
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import dalal_street_lab.alpaca as alpaca_module
from dalal_street_lab.alpaca import AlpacaAuthError, AlpacaClient
from dalal_street_lab.config import Settings
from dalal_street_lab.instruments import (
    AlpacaAsset,
    UnusableSymbolError,
    Watchlist,
    fetch_assets,
    save,
)
from dalal_street_lab.models import Instrument

TRADING = "https://paper-api.alpaca.markets"


def settings(**changes: Any) -> Settings:
    values: dict[str, Any] = {
        "postgres_password": "x",
        "alpaca_api_key": "key-1",
        "alpaca_secret_key": "secret-1",
    }
    values.update(changes)
    return Settings(**values)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests shouldn't sit through the real rate-limit pauses."""
    monkeypatch.setattr(alpaca_module, "SECONDS_BETWEEN_REQUESTS", 0)


def asset(symbol: str, status: str = "active", tradable: bool = True) -> dict[str, Any]:
    return {
        "id": f"uuid-{symbol}",
        "class": "us_equity",
        "exchange": "NASDAQ",
        "symbol": symbol,
        "name": f"{symbol} Fund",
        "status": status,
        "tradable": tradable,
        "marginable": True,
        "shortable": True,
        "fractionable": True,
    }


def test_real_watchlist_file_is_valid() -> None:
    types = Watchlist.load(Path(__file__).parent.parent / "watchlist.toml").with_types()
    assert types["INDY"] == "ETF"  # Nifty 50
    assert types["INFY"] == "ADR"
    assert len(types) <= 30


def test_too_many_symbols_is_refused() -> None:
    watchlist = Watchlist(
        india_etfs=[f"E{i}" for i in range(20)],
        india_adrs=[f"A{i}" for i in range(11)],
        commodity_etfs=[],
    )
    with pytest.raises(ValueError, match="30"):
        watchlist.with_types()


def test_client_needs_keys() -> None:
    with pytest.raises(AlpacaAuthError, match=".env"):
        AlpacaClient(settings(alpaca_api_key=""))


@responses.activate
def test_client_sends_keys_in_headers() -> None:
    responses.get(
        f"{TRADING}/v2/assets/INDY",
        json=asset("INDY"),
        match=[
            matchers.header_matcher({"APCA-API-KEY-ID": "key-1", "APCA-API-SECRET-KEY": "secret-1"})
        ],
    )
    assert AlpacaClient(settings()).get(f"{TRADING}/v2/assets/INDY")["symbol"] == "INDY"


@responses.activate
def test_client_paces_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_module, "SECONDS_BETWEEN_REQUESTS", 0.2)
    responses.get(f"{TRADING}/v2/assets/INDY", json=asset("INDY"))
    client = AlpacaClient(settings())
    started = time.monotonic()
    for _ in range(3):
        client.get(f"{TRADING}/v2/assets/INDY")
    assert time.monotonic() - started >= 0.4  # 3 requests -> at least 2 pauses


@responses.activate
def test_fetch_assets_reports_every_unusable_symbol() -> None:
    responses.get(f"{TRADING}/v2/assets/INDY", json=asset("INDY"))
    responses.get(f"{TRADING}/v2/assets/NOPE", status=404)
    responses.get(f"{TRADING}/v2/assets/OLD", json=asset("OLD", status="inactive"))
    with pytest.raises(UnusableSymbolError, match=r"NOPE \(HTTP 404\).*OLD"):
        fetch_assets(AlpacaClient(settings()), ["INDY", "NOPE", "OLD"])


@responses.activate
def test_wrong_keys_stop_immediately_with_a_clear_message() -> None:
    responses.get(f"{TRADING}/v2/assets/INDY", status=401)
    with pytest.raises(AlpacaAuthError, match="rejected"):
        fetch_assets(AlpacaClient(settings()), ["INDY", "GLD"])


def test_save_inserts_updates_and_retires(session: Session) -> None:
    indy, gld, infy = (AlpacaAsset.model_validate(asset(s)) for s in ["INDY", "GLD", "INFY"])
    types = {"INDY": "ETF", "GLD": "ETF", "INFY": "ADR"}
    save(session, [indy, gld, infy], types)
    save(session, [indy, gld, infy], types)  # running twice must not create duplicates
    assert session.scalar(select(func.count()).select_from(Instrument)) == 3

    save(session, [indy, infy], types)  # GLD removed from the watchlist
    active = set(session.scalars(select(Instrument.symbol).where(Instrument.is_active)))
    assert active == {"INDY", "INFY"}
    assert session.scalar(select(func.count()).select_from(Instrument)) == 3  # kept, inactive
