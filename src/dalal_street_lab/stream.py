"""Stream live trades and minute bars from Alpaca.

Run it with:  uv run dalal-stream        (Ctrl+C to stop)

Where the data goes:
  trades -> Redis  (the live scoreboard: latest price per symbol, plus a live feed channel)
  trades -> ticks table, minute bars -> candles table   (written once a second, in batches)

This program uses asyncio: one thread juggles the websocket, Redis and a timer, switching
whenever one of them is waiting. Slow work (the database) runs in a helper thread, so the
websocket is never blocked and no message is missed.
"""

import asyncio
import contextlib
import json
import signal
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import redis.asyncio as aioredis
import structlog
import websockets
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from websockets.asyncio.client import ClientConnection

from dalal_street_lab.config import Settings, get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.history import parse_bars, save_candles
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Instrument, Tick

log = structlog.get_logger()

FLUSH_SECONDS = 1.0  # how often buffered trades and bars are written to the database
MAX_BACKOFF_SECONDS = 60  # longest wait between reconnect attempts
LIVE_CHANNEL = "live:trades"  # Redis channel that anyone can listen to for live trades

# Errors that retrying can't fix: stop and tell the human.
FATAL_ERRORS = {
    402: "Alpaca rejected the API keys. Check them in .env.",
    405: "Too many symbols for your plan (free limit: 30).",
    406: "Another stream is already connected. The free plan allows only one at a time.",
    409: "Your plan doesn't include this data feed.",
}


class StreamError(Exception):
    """A problem that reconnecting won't fix."""


def price_key(symbol: str) -> str:
    return f"live:price:{symbol}"


Writer = Callable[[list[dict[str, Any]], dict[int, list[dict[str, Any]]]], None]


def write_to_database(ticks: list[dict[str, Any]], bars: dict[int, list[dict[str, Any]]]) -> None:
    """Save a batch: trades into ticks, minute bars into candles."""
    with new_session() as session:
        if ticks:
            # Two trades in the same microsecond can't both be stored: keep the first.
            session.execute(insert(Tick).values(ticks).on_conflict_do_nothing())
            session.commit()
        for instrument_id, raw in bars.items():
            save_candles(session, parse_bars(raw, instrument_id, "1minute"))


class LiveStream:
    def __init__(
        self,
        settings: Settings,
        symbols: dict[str, int],  # symbol -> instrument id
        redis_client: aioredis.Redis,
        writer: Writer = write_to_database,
    ) -> None:
        self.settings = settings
        self.symbols = symbols
        self.redis = redis_client
        self.writer = writer
        self.ticks: list[dict[str, Any]] = []
        self.bars: dict[int, list[dict[str, Any]]] = {}
        self.backoff = 1.0
        self.received = 0

    # ---------- Connecting, logging in, subscribing ----------
    async def run(self, stop: asyncio.Event) -> None:
        """Stay connected until told to stop. If the connection drops, wait and reconnect,
        waiting longer each time (1s, 2s, 4s... up to 60s) so we don't hammer Alpaca."""
        while not stop.is_set():
            try:
                await self._session(stop)
            except StreamError:
                stop.set()  # let the flush loop save what we have, then stop
                raise
            except (OSError, websockets.ConnectionClosed) as error:
                log.warning("stream_disconnected", error=str(error), retry_in=self.backoff)
                with contextlib.suppress(TimeoutError):  # wait, but wake early if stopped
                    await asyncio.wait_for(stop.wait(), timeout=self.backoff)
                self.backoff = min(self.backoff * 2, MAX_BACKOFF_SECONDS)

    async def _session(self, stop: asyncio.Event) -> None:
        async with websockets.connect(self.settings.alpaca_stream_url) as ws:
            await self._login_and_subscribe(ws)
            self.backoff = 1.0  # connected fine: reset the waiting time
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except TimeoutError:
                    continue  # nothing arrived this second: check `stop`, keep listening
                for message in json.loads(raw):
                    await self.handle(message)

    async def _expect(self, ws: ClientConnection, wanted: str) -> None:
        """Read one control message; fail loudly if it's an error."""
        for message in json.loads(await ws.recv()):
            if message.get("T") == "error":
                self._raise_error(message)
            if message.get("T") == wanted or message.get("msg") == wanted:
                return
        raise StreamError(f"Expected '{wanted}' from Alpaca, got something else.")

    async def _login_and_subscribe(self, ws: ClientConnection) -> None:
        await self._expect(ws, "connected")
        secret = self.settings.alpaca_secret_key.get_secret_value()
        await ws.send(
            json.dumps({"action": "auth", "key": self.settings.alpaca_api_key, "secret": secret})
        )
        await self._expect(ws, "authenticated")

        symbols = sorted(self.symbols)
        await ws.send(
            json.dumps(
                {"action": "subscribe", "trades": symbols, "bars": symbols, "updatedBars": symbols}
            )
        )
        await self._expect(ws, "subscription")
        log.info("stream_subscribed", symbols=symbols)

    def _raise_error(self, message: dict[str, Any]) -> None:
        code = int(message.get("code", 0))
        reason = FATAL_ERRORS.get(code, f"Alpaca error {code}: {message.get('msg')}")
        if code in FATAL_ERRORS:
            raise StreamError(reason)
        raise websockets.ConnectionClosedError(None, None)  # anything else: reconnect

    # ---------- Handling each message ----------
    async def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("T")
        if kind == "t":
            await self._on_trade(message)
        elif kind in ("b", "u"):  # a finished minute bar, or a correction to one
            self._on_bar(message)
        elif kind == "error":
            self._raise_error(message)

    async def _on_trade(self, trade: dict[str, Any]) -> None:
        symbol = trade["S"]
        if symbol not in self.symbols:
            return
        self.received += 1
        time = datetime.fromisoformat(trade["t"])

        # 1. The live scoreboard: one Redis hash per symbol, always the latest trade.
        # 2. The live feed: publish the trade so anything listening sees it instantly.
        update = {"price": str(trade["p"]), "size": str(trade["s"]), "time": time.isoformat()}
        async with self.redis.pipeline() as pipe:  # send both in one round trip
            fields = ["price", update["price"], "size", update["size"], "time", update["time"]]
            pipe.hset(price_key(symbol), items=fields)  # items = field, value, field, value...
            pipe.publish(LIVE_CHANNEL, json.dumps({"symbol": symbol, **update}))
            await pipe.execute()

        # 3. The history book: buffered, then written to the ticks table in batches.
        self.ticks.append(
            {
                "instrument_id": self.symbols[symbol],
                "time": time,
                "price": Decimal(str(trade["p"])),
                "volume": int(trade["s"]),
            }
        )

    def _on_bar(self, bar: dict[str, Any]) -> None:
        if bar["S"] in self.symbols:
            self.bars.setdefault(self.symbols[bar["S"]], []).append(bar)

    # ---------- Writing batches to the database ----------
    async def flush(self) -> None:
        ticks, bars = self.ticks, self.bars
        self.ticks, self.bars = [], {}  # new messages go into fresh buffers right away
        if ticks or bars:
            # The database is slow compared with the websocket: run it in a helper thread.
            await asyncio.to_thread(self.writer, ticks, bars)
            log.debug("flushed", ticks=len(ticks), bars=sum(len(b) for b in bars.values()))

    async def flush_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):  # wait a second, or less if stopped
                await asyncio.wait_for(stop.wait(), timeout=FLUSH_SECONDS)
            await self.flush()
        await self.flush()  # one last time, so nothing buffered is lost on shutdown


def active_symbols() -> dict[str, int]:
    with new_session() as session:
        rows = session.execute(
            select(Instrument.symbol, Instrument.id).where(Instrument.is_active)
        ).all()
    return {symbol: instrument_id for symbol, instrument_id in rows}


async def run_stream() -> None:
    settings = get_settings()
    symbols = active_symbols()
    if not symbols:
        raise SystemExit("No instruments in the database. Run: uv run dalal-instruments")

    # Ctrl+C (SIGINT) and Kubernetes' "please stop" (SIGTERM) both mean: stop cleanly.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    stream = LiveStream(settings, symbols, redis_client)
    try:
        await asyncio.gather(stream.run(stop), stream.flush_loop(stop))
    finally:
        await redis_client.aclose()
        log.info("stream_stopped", trades_received=stream.received)


def main() -> None:
    configure_logging()
    asyncio.run(run_stream())


if __name__ == "__main__":
    main()
