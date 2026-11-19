"""The web service, with the test database and a fake Redis swapped in."""

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from dalal_street_lab.api import app, get_session, relay
from dalal_street_lab.features import FEATURE_VERSION
from dalal_street_lab.models import Candle, FeatureRow, Instrument, Prediction
from dalal_street_lab.stream import LIVE_CHANNEL

FEATURE_VALUES = {
    "ret_1d": 0.01,
    "ret_5d": 0.02,
    "ret_20d": 0.03,
    "volatility_20d": 0.01,
    "volume_ratio_20d": 0.0,
    "rsi_14": 55.0,
    "sma_gap_20d": 0.01,
    "range_1d": 0.02,
    "market_ret_1d": 0.005,
    "news_count_1d": 1.0,
    "news_mood_1d": 0.3,
    "news_mood_5d": 0.2,
}


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    def test_session() -> Iterator[Session]:
        yield session

    app.dependency_overrides[get_session] = test_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def add_symbol(session: Session, symbol: str, prob_up: float | None) -> Instrument:
    instrument = Instrument(
        instrument_key=f"uuid-{symbol}",
        symbol=symbol,
        name=symbol,
        exchange="NYSE",
        segment="us_equity",
        instrument_type="ETF",
        tick_size=Decimal("0.01"),
    )
    session.add(instrument)
    session.flush()
    session.add(
        Candle(
            instrument_id=instrument.id,
            interval="1day",
            time=datetime(2026, 11, 17, 5, tzinfo=UTC),
            open=Decimal("10"),
            high=Decimal("11"),
            low=Decimal("9"),
            close=Decimal("10.5"),
            volume=1000,
        )
    )
    session.add(
        FeatureRow(
            instrument_id=instrument.id,
            day=date(2026, 11, 17),
            feature_version=FEATURE_VERSION,
            **FEATURE_VALUES,
        )
    )
    if prob_up is not None:
        session.add(
            Prediction(
                model_name="logistic",
                instrument_id=instrument.id,
                day=date(2026, 11, 17),
                prob_up=prob_up,
                feature_version=FEATURE_VERSION,
                mlflow_run_id="run-1",
            )
        )
    session.flush()
    return instrument


def test_health_and_unknown_symbols(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/prices/NOPE").status_code == 404


def test_prices_and_predictions_for_the_chart(client: TestClient, session: Session) -> None:
    add_symbol(session, "INFY", 0.56)

    candles = client.get("/api/prices/infy").json()
    guesses = client.get("/api/predictions/INFY").json()

    assert candles == [
        {
            "time": "2026-11-17",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000,
        }
    ]
    assert guesses[0]["prob_up"] == 0.56
    assert guesses[0]["mlflow_run_id"] == "run-1"


def test_screener_sorts_and_puts_missing_guesses_last(client: TestClient, session: Session) -> None:
    add_symbol(session, "INFY", 0.52)
    add_symbol(session, "GLD", 0.61)
    add_symbol(session, "USO", None)

    rows = client.get("/api/screener?sort=prob_up").json()["rows"]

    assert [r["symbol"] for r in rows] == ["GLD", "INFY", "USO"]
    assert client.get("/api/screener?sort=password").status_code == 422


def test_dashboard_page_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Dalal Street Lab" in page.text


async def test_live_trades_are_relayed_from_redis() -> None:
    fake = fakeredis.FakeAsyncRedis(decode_responses=True)
    trade = {"symbol": "INFY", "price": "19.87", "size": 10, "time": "2026-11-18T15:30:00Z"}
    received: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        received.append(message)

    task = asyncio.create_task(relay(fake, send, max_messages=1))
    for _ in range(100):  # wait until the relay has subscribed, then publish
        if await fake.publish(LIVE_CHANNEL, json.dumps(trade)):
            break
        await asyncio.sleep(0.01)
    await asyncio.wait_for(task, timeout=5)

    assert received == [trade]


def test_metrics_page_reports_request_times_and_bot_equity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    fake.mset({"bot:equity": 101_000, "bot:last_equity": 100_000})
    monkeypatch.setattr("dalal_street_lab.metrics.redis.Redis.from_url", lambda *a, **k: fake)

    client.get("/health")
    page = client.get("/metrics").text

    assert 'dalal_api_request_seconds_count{method="GET",route="/health",status="200"}' in page
    assert "dalal_bot_daily_pnl_dollars 1000.0" in page
