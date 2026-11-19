"""Step 12: the web service. Everything the dashboard (or anyone) needs, over HTTP.

Run it with:  uv run dalal-api        then open http://localhost:8000
API docs:     http://localhost:8000/docs   (FastAPI writes these for you)

    GET  /health                       is the service up, and is the database reachable?
    GET  /api/instruments              the symbols we watch
    GET  /api/prices/{symbol}          daily candles, for the chart
    GET  /api/predictions/{symbol}     the model's out-of-sample guesses over time
    GET  /api/news/{symbol}            recent headlines with their FinBERT mood
    GET  /api/screener                 every symbol's latest numbers, sortable
    GET  /api/orders                   what the paper bot decided and what filled
    WS   /ws/trades                    live trades, pushed the moment they happen

Dependency injection: routes ask for a database session and a Redis client through
`Depends(...)` instead of making their own. Tests swap in a test database and a fake
Redis with one line each, without changing any route code.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from importlib.resources import files
from typing import Annotated, Any

import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from dalal_street_lab.config import get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.features import FEATURE_VERSION
from dalal_street_lab.models import (
    Article,
    ArticleSymbol,
    Candle,
    FeatureRow,
    Instrument,
    Order,
    Prediction,
)
from dalal_street_lab.stream import LIVE_CHANNEL

app = FastAPI(title="Dalal Street Lab", version="0.1.0")


# ---------- Dependencies (swappable in tests) ----------
def get_session() -> Iterator[Session]:
    with new_session() as session:
        yield session


async def get_redis() -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


SessionDep = Annotated[Session, Depends(get_session)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]


def instrument_or_404(session: Session, symbol: str) -> Instrument:
    found = session.scalar(select(Instrument).where(Instrument.symbol == symbol.upper()))
    if found is None:
        raise HTTPException(404, f"Unknown symbol {symbol!r}.")
    return found


# ---------- Routes ----------
@app.get("/health")
def health(session: SessionDep) -> dict[str, str]:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/instruments")
def instruments(session: SessionDep) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Instrument).where(Instrument.is_active).order_by(Instrument.symbol)
    ).all()
    return [{"symbol": r.symbol, "name": r.name, "type": r.instrument_type} for r in rows]


@app.get("/api/prices/{symbol}")
def prices(
    symbol: str, session: SessionDep, limit: Annotated[int, Query(ge=1, le=2000)] = 500
) -> list[dict[str, Any]]:
    instrument = instrument_or_404(session, symbol)
    rows = session.scalars(
        select(Candle)
        .where(Candle.instrument_id == instrument.id, Candle.interval == "1day")
        .order_by(Candle.time.desc())
        .limit(limit)
    ).all()
    return [
        {
            "time": row.time.date().isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": row.volume,
        }
        for row in reversed(rows)
    ]


@app.get("/api/predictions/{symbol}")
def predictions(symbol: str, session: SessionDep, model: str = "logistic") -> list[dict[str, Any]]:
    instrument = instrument_or_404(session, symbol)
    rows = session.scalars(
        select(Prediction)
        .where(Prediction.instrument_id == instrument.id, Prediction.model_name == model)
        .order_by(Prediction.day)
    ).all()
    return [
        {"day": r.day.isoformat(), "prob_up": r.prob_up, "mlflow_run_id": r.mlflow_run_id}
        for r in rows
    ]


@app.get("/api/news/{symbol}")
def news(
    symbol: str, session: SessionDep, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[dict[str, Any]]:
    instrument = instrument_or_404(session, symbol)
    rows = session.scalars(
        select(Article)
        .join(ArticleSymbol, ArticleSymbol.news_id == Article.id)
        .where(ArticleSymbol.instrument_id == instrument.id)
        .order_by(Article.published_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "headline": r.headline,
            "url": r.url,
            "published_at": r.published_at.isoformat(),
            "sentiment": None if r.sentiment is None else float(r.sentiment),
        }
        for r in rows
    ]


SORTABLE = ("symbol", "ret_1d", "ret_5d", "rsi_14", "news_mood_5d", "prob_up")


@app.get("/api/screener")
def screener(session: SessionDep, sort: str = "prob_up", model: str = "logistic") -> dict[str, Any]:
    """Each symbol's newest features, next to the model's newest guess for it."""
    if sort not in SORTABLE:
        raise HTTPException(422, f"sort must be one of: {', '.join(SORTABLE)}")
    newest_feature = (
        select(FeatureRow.instrument_id, func.max(FeatureRow.day).label("day"))
        .where(FeatureRow.feature_version == FEATURE_VERSION)
        .group_by(FeatureRow.instrument_id)
        .subquery()
    )
    newest_guess = (
        select(Prediction.instrument_id, func.max(Prediction.day).label("day"))
        .where(Prediction.model_name == model)
        .group_by(Prediction.instrument_id)
        .subquery()
    )
    rows = session.execute(
        select(Instrument.symbol, FeatureRow, Prediction.prob_up)
        .join(FeatureRow, FeatureRow.instrument_id == Instrument.id)
        .join(
            newest_feature,
            (newest_feature.c.instrument_id == FeatureRow.instrument_id)
            & (newest_feature.c.day == FeatureRow.day),
        )
        .outerjoin(newest_guess, newest_guess.c.instrument_id == Instrument.id)
        .outerjoin(
            Prediction,
            (Prediction.instrument_id == Instrument.id)
            & (Prediction.day == newest_guess.c.day)
            & (Prediction.model_name == model),
        )
        .where(FeatureRow.feature_version == FEATURE_VERSION)
    ).all()
    table = [
        {
            "symbol": symbol,
            "day": feature.day.isoformat(),
            "ret_1d": feature.ret_1d,
            "ret_5d": feature.ret_5d,
            "rsi_14": feature.rsi_14,
            "news_mood_5d": feature.news_mood_5d,
            "prob_up": prob_up,
        }
        for symbol, feature, prob_up in rows
    ]
    # Biggest first (A to Z for symbols). Symbols without a guess yet go last.
    known = [r for r in table if r[sort] is not None]
    known.sort(
        key=lambda r: str(r[sort]) if sort == "symbol" else float(r[sort]),  # type: ignore[arg-type]
        reverse=sort != "symbol",
    )
    table = known + [r for r in table if r[sort] is None]
    return {"sort": sort, "model": model, "rows": table}


@app.get("/api/orders")
def orders(
    session: SessionDep, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Order, Instrument.symbol)
        .join(Instrument, Instrument.id == Order.instrument_id)
        .order_by(Order.created_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "symbol": symbol,
            "signal_day": o.signal_day.isoformat(),
            "side": o.side,
            "qty": o.qty,
            "status": o.status,
            "filled_qty": o.filled_qty,
            "filled_avg_price": None if o.filled_avg_price is None else float(o.filled_avg_price),
            "reason": o.reason,
        }
        for o, symbol in rows
    ]


async def relay(
    redis: aioredis.Redis,
    send: Callable[[dict[str, Any]], Awaitable[None]],
    max_messages: int | None = None,
) -> None:
    """Forward every trade published on the live channel to `send`, as it arrives."""
    sent = 0
    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(LIVE_CHANNEL)
        while max_messages is None or sent < max_messages:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is None:
                await asyncio.sleep(0.05)
                continue
            await send(json.loads(message["data"]))
            sent += 1


@app.websocket("/ws/trades")
async def live_trades(websocket: WebSocket, redis: RedisDep) -> None:
    """Relay every trade the stream publishes to Redis, straight to the browser."""
    await websocket.accept()
    try:
        await relay(redis, websocket.send_json)
    except WebSocketDisconnect:
        return


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return files("dalal_street_lab").joinpath("dashboard.html").read_text(encoding="utf-8")


def main() -> None:
    uvicorn.run("dalal_street_lab.api:app", host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    main()
