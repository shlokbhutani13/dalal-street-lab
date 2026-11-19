"""The live stream, tested against a FAKE Alpaca websocket server running on this computer."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import fakeredis
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from websockets.asyncio.server import ServerConnection, serve

from dalal_street_lab.config import Settings
from dalal_street_lab.history import parse_bars, save_candles
from dalal_street_lab.models import Candle, Instrument, Tick
from dalal_street_lab.stream import LIVE_CHANNEL, LiveStream, StreamError, price_key

TRADE = {
    "T": "t",
    "S": "INFY",
    "i": 1,
    "x": "V",
    "p": 17.42,
    "s": 100,
    "t": "2026-10-26T18:30:01.123456789Z",
    "c": ["@"],
    "z": "A",
}
BAR = {
    "T": "b",
    "S": "INFY",
    "o": 17.40,
    "h": 17.45,
    "l": 17.38,
    "c": 17.42,
    "v": 1200,
    "n": 9,
    "vw": 17.41,
    "t": "2026-10-26T18:30:00Z",
}

Script = Callable[[ServerConnection], Any]


def settings(port: int, key: str = "good-key") -> Settings:
    return Settings(
        postgres_password="x",
        alpaca_api_key=key,
        alpaca_secret_key="secret",
        alpaca_stream_url=f"ws://127.0.0.1:{port}",
    )


async def alpaca_login(ws: ServerConnection) -> bool:
    """Behave like Alpaca's login: 'connected', check the keys, confirm the subscription."""
    await ws.send(json.dumps([{"T": "success", "msg": "connected"}]))
    auth = json.loads(await ws.recv())
    if auth["key"] != "good-key":
        await ws.send(json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]))
        return False
    await ws.send(json.dumps([{"T": "success", "msg": "authenticated"}]))
    subscribe = json.loads(await ws.recv())
    await ws.send(json.dumps([{"T": "subscription", "trades": subscribe["trades"]}]))
    return True


@pytest.fixture
async def fake_alpaca() -> AsyncIterator[Callable[[Script], Any]]:
    """Start a fake Alpaca server; each test says what it should send."""
    servers = []

    async def start(script: Script) -> int:
        server = await serve(script, "127.0.0.1", 0)
        servers.append(server)
        port: int = server.sockets[0].getsockname()[1]
        return port

    yield start
    for server in servers:
        server.close()
        await server.wait_closed()


class Recorder:
    """Stands in for the database writer and remembers what it was given."""

    def __init__(self) -> None:
        self.ticks: list[dict[str, Any]] = []
        self.bars: dict[int, list[dict[str, Any]]] = {}

    def __call__(self, ticks: list[dict[str, Any]], bars: dict[int, list[dict[str, Any]]]) -> None:
        self.ticks += ticks
        for instrument_id, rows in bars.items():
            self.bars.setdefault(instrument_id, []).extend(rows)


async def run_until(stream: LiveStream, stop: asyncio.Event, done: Callable[[], bool]) -> None:
    """Run the stream until `done()` is true (or 5 seconds pass), then stop it cleanly."""
    task = asyncio.gather(stream.run(stop), stream.flush_loop(stop))
    for _ in range(100):
        if done():
            break
        await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=5)


async def test_trades_reach_redis_and_the_database(fake_alpaca: Any) -> None:
    async def script(ws: ServerConnection) -> None:
        if await alpaca_login(ws):
            await ws.send(json.dumps([TRADE, BAR]))
            await asyncio.sleep(2)

    port = await fake_alpaca(script)
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    recorder = Recorder()
    stream = LiveStream(settings(port), {"INFY": 7}, redis, writer=recorder)

    listener = redis.pubsub()  # listen to the live channel, like dalal-watch does
    await listener.subscribe(LIVE_CHANNEL)
    assert (await listener.get_message(timeout=1))["type"] == "subscribe"  # confirmation

    await run_until(stream, asyncio.Event(), lambda: bool(recorder.ticks and recorder.bars))
    announced = await listener.get_message(timeout=1)

    # 1. The live scoreboard in Redis has the latest price.
    assert await redis.hgetall(price_key("INFY")) == {
        "price": "17.42",
        "size": "100",
        "time": "2026-10-26T18:30:01.123456+00:00",
    }
    # 2. The trade was announced on the live channel.
    assert announced is not None
    assert json.loads(announced["data"])["symbol"] == "INFY"
    # 3. The trade and the bar were handed to the database writer.
    assert recorder.ticks[0]["price"] == Decimal("17.42")
    assert recorder.bars == {7: [BAR]}


async def test_wrong_keys_stop_the_stream_instead_of_retrying_forever(fake_alpaca: Any) -> None:
    port = await fake_alpaca(alpaca_login)
    stream = LiveStream(
        settings(port, key="bad-key"), {"INFY": 7}, fakeredis.FakeAsyncRedis(), writer=Recorder()
    )
    with pytest.raises(StreamError, match="API keys"):
        await asyncio.wait_for(stream.run(asyncio.Event()), timeout=5)


async def test_dropped_connection_reconnects(fake_alpaca: Any) -> None:
    connections = 0

    async def script(ws: ServerConnection) -> None:
        nonlocal connections
        connections += 1
        if await alpaca_login(ws):
            if connections == 1:
                return  # first time: hang up right after subscribing (a dropped connection)
            await ws.send(json.dumps([TRADE]))
            await asyncio.sleep(2)

    port = await fake_alpaca(script)
    recorder = Recorder()
    stream = LiveStream(settings(port), {"INFY": 7}, fakeredis.FakeAsyncRedis(), writer=recorder)
    stream.backoff = 0.05  # don't wait a full second in a test
    await run_until(stream, asyncio.Event(), lambda: bool(recorder.ticks))

    assert connections == 2  # it came back by itself
    assert len(recorder.ticks) == 1


async def test_symbols_we_did_not_ask_for_are_ignored(fake_alpaca: Any) -> None:
    async def script(ws: ServerConnection) -> None:
        if await alpaca_login(ws):
            await ws.send(json.dumps([{**TRADE, "S": "AAPL"}, TRADE]))
            await asyncio.sleep(2)

    port = await fake_alpaca(script)
    recorder = Recorder()
    stream = LiveStream(settings(port), {"INFY": 7}, fakeredis.FakeAsyncRedis(), writer=recorder)
    await run_until(stream, asyncio.Event(), lambda: bool(recorder.ticks))
    assert [t["instrument_id"] for t in recorder.ticks] == [7]


def test_live_rows_fit_the_real_tables(session: Session) -> None:
    """What the stream produces must be saveable: same shapes the tables expect."""
    instrument = Instrument(
        instrument_key="uuid-INFY",
        symbol="INFY",
        name="Infosys",
        exchange="NYSE",
        segment="us_equity",
        instrument_type="ADR",
        tick_size=Decimal("0.01"),
    )
    session.add(instrument)
    session.flush()
    session.add(
        Tick(
            instrument_id=instrument.id,
            time=datetime.fromisoformat(str(TRADE["t"])),
            price=Decimal("17.42"),
            volume=100,
        )
    )
    save_candles(session, parse_bars([BAR], instrument.id, "1minute"))
    assert session.scalar(select(func.count()).select_from(Tick)) == 1
    assert session.scalar(select(func.count()).select_from(Candle)) == 1


def test_trades_are_counted_for_prometheus() -> None:
    from dalal_street_lab.metrics import STREAM_TRADES

    before = STREAM_TRADES.labels(symbol="INFY")._value.get()
    stream = LiveStream(settings(1), {"INFY": 7}, fakeredis.FakeAsyncRedis(), writer=Recorder())
    asyncio.run(
        stream.handle({"T": "t", "S": "INFY", "p": 19.9, "s": 5, "t": "2026-11-18T15:30:00Z"})
    )
    assert STREAM_TRADES.labels(symbol="INFY")._value.get() == before + 1
