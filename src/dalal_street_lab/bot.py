"""Step 11: the paper-trading bot. Fake money only.

Run after the close, once today's features are built:
    uv run dalal-bot             # dry run: shows the plan, sends nothing
    uv run dalal-bot --submit    # sends the orders to your Alpaca PAPER account
    uv run dalal-bot --sync      # checks which orders filled, and at what price

What it does, in order. Any safety check that fails stops it before an order goes out:

    1. paper account only?      the trading URL must be Alpaca's paper URL
    2. kill switch off?          DALAL_KILL_SWITCH=1 or a KILL_SWITCH file stops everything
    3. data fresh?               the newest features must be from the last closed trading day
    4. learn                     train the chosen model on every day with a known answer
    5. predict                   chance each symbol goes up tomorrow
    6. size positions            each pick gets an equal share, within the risk limits
    7. daily loss stop           after a bad day, only sells are allowed (no new risk)
    8. orders                    the difference between what we hold and what we want
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import Settings, get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.features import FEATURE_VERSION, FEATURES, closed_sessions
from dalal_street_lab.learners import Learner, Logistic, NeuralNet
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Candle, FeatureRow, Instrument, Order
from dalal_street_lab.training import load_dataset

log = structlog.get_logger()

PAPER_URL = "https://paper-api.alpaca.markets"
KILL_SWITCH_FILE = Path("KILL_SWITCH")
OPEN_STATUSES = ("submitted", "new", "accepted", "pending_new", "partially_filled")


class BotStoppedError(Exception):
    """A safety check failed, so the bot stopped before sending any order."""


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.10  # no single symbol above 10% of the account
    max_exposure_pct: float = 0.60  # at most 60% of the account invested at once
    daily_loss_stop_pct: float = 0.02  # down 2% since yesterday's close: no new buys


@dataclass
class PlannedOrder:
    symbol: str
    instrument_id: int
    side: str
    qty: int
    prob_up: float | None
    reason: str


@dataclass
class Plan:
    signal_day: date
    model_name: str
    equity: float
    orders: list[PlannedOrder] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


# ---------- Safety checks ----------
def check_paper_only(settings: Settings) -> None:
    if settings.alpaca_trading_url.rstrip("/") != PAPER_URL:
        raise BotStoppedError(
            f"Refusing to trade: {settings.alpaca_trading_url} is not {PAPER_URL}."
        )


def kill_switch_on() -> bool:
    return os.environ.get("DALAL_KILL_SWITCH", "") in ("1", "true", "yes") or (
        KILL_SWITCH_FILE.exists()
    )


def check_fresh(signal_day: date, now: datetime) -> None:
    """The features must be from the most recent trading day whose close has passed."""
    sessions = closed_sessions(now.date() - timedelta(days=14), now)
    if sessions.empty:
        raise BotStoppedError("No closed trading day in the last two weeks to trade on.")
    last_closed = sessions["day"].iloc[-1]
    if signal_day != last_closed:
        raise BotStoppedError(
            f"Stale data: newest features are for {signal_day}, but the last close was "
            f"{last_closed}. Run dalal-backfill and dalal-features first."
        )


# ---------- Reading what we need ----------
def latest_features(session: Session) -> pd.DataFrame:
    newest = session.scalar(
        select(func.max(FeatureRow.day)).where(FeatureRow.feature_version == FEATURE_VERSION)
    )
    rows = session.execute(
        select(Instrument.symbol, FeatureRow)
        .join(Instrument, Instrument.id == FeatureRow.instrument_id)
        .where(FeatureRow.feature_version == FEATURE_VERSION, FeatureRow.day == newest)
    ).all()
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "instrument_id": row.instrument_id,
                "day": row.day,
                **{name: getattr(row, name) for name in FEATURES},
            }
            for symbol, row in rows
        ]
    )


def latest_closes(session: Session) -> dict[str, float]:
    """The most recent daily close for each symbol: used to turn dollars into shares."""
    newest = (
        select(Candle.instrument_id, func.max(Candle.time).label("time"))
        .where(Candle.interval == "1day")
        .group_by(Candle.instrument_id)
        .subquery()
    )
    rows = session.execute(
        select(Instrument.symbol, Candle.close)
        .join(Candle, Candle.instrument_id == Instrument.id)
        .join(
            newest,
            (newest.c.instrument_id == Candle.instrument_id) & (newest.c.time == Candle.time),
        )
        .where(Candle.interval == "1day")
    ).all()
    return {symbol: float(close) for symbol, close in rows}


def make_learner(name: str) -> Learner:
    learners: dict[str, Learner] = {
        "logistic": Logistic(FEATURES),
        "neural_net": NeuralNet(FEATURES),
    }
    if name not in learners:
        raise BotStoppedError(f"Unknown model {name!r}. Choose from: {', '.join(learners)}.")
    return learners[name]


# ---------- Deciding ----------
def plan_orders(
    *,
    predictions: pd.DataFrame,
    positions: dict[str, int],
    closes: dict[str, float],
    equity: float,
    last_equity: float,
    threshold: float,
    limits: RiskLimits,
    signal_day: date,
    model_name: str,
) -> Plan:
    """Pure decision logic: no network and no database, so every rule is easy to test."""
    plan = Plan(signal_day=signal_day, model_name=model_name, equity=equity)
    buys_allowed = True
    if last_equity > 0 and (equity - last_equity) / last_equity <= -limits.daily_loss_stop_pct:
        buys_allowed = False
        plan.blocked.append(
            f"Daily loss stop: account is down more than {limits.daily_loss_stop_pct:.0%} "
            "since yesterday's close, so only sells are allowed today."
        )

    picks = predictions[predictions["prob_up"] > threshold]
    per_symbol = 0.0
    if len(picks):
        per_symbol = min(
            limits.max_position_pct * equity, limits.max_exposure_pct * equity / len(picks)
        )

    for row in predictions.to_dict("records"):
        symbol, price = row["symbol"], closes.get(row["symbol"])
        if price is None or price <= 0:
            plan.blocked.append(f"{symbol}: no recent price, skipped.")
            continue
        want = math.floor(per_symbol / price) if row["prob_up"] > threshold else 0
        have = positions.get(symbol, 0)
        change = want - have
        if change == 0:
            continue
        side = "buy" if change > 0 else "sell"
        reason = (
            f"prob_up {row['prob_up']:.3f} > {threshold}: hold {want} shares"
            if want
            else f"prob_up {row['prob_up']:.3f} <= {threshold}: exit"
        )
        if side == "buy" and not buys_allowed:
            plan.blocked.append(f"{symbol}: buy of {change} blocked by the daily loss stop.")
            continue
        plan.orders.append(
            PlannedOrder(
                symbol, int(row["instrument_id"]), side, abs(change), row["prob_up"], reason
            )
        )
    # Sell before buying, so cash is freed up first.
    plan.orders.sort(key=lambda order: order.side != "sell")
    return plan


# ---------- Talking to Alpaca ----------
def account_state(client: AlpacaClient) -> tuple[float, float, dict[str, int]]:
    account = client.get(f"{client.trading_url}/v2/account")
    if account.get("trading_blocked") or account.get("account_blocked"):
        raise BotStoppedError("Alpaca says this account is blocked from trading.")
    positions = client.get(f"{client.trading_url}/v2/positions")
    held = {item["symbol"]: int(float(item["qty"])) for item in positions}
    return float(account["equity"]), float(account["last_equity"]), held


def client_order_id(plan: Plan, order: PlannedOrder) -> str:
    """Same day + symbol + side = same id, so a second run can't place a duplicate."""
    return f"dsl-{plan.signal_day:%Y%m%d}-{order.symbol}-{order.side}"


def record_and_submit(
    session: Session, client: AlpacaClient | None, plan: Plan, submit: bool
) -> list[dict[str, Any]]:
    saved = []
    for order in plan.orders:
        record: dict[str, Any] = {
            "client_order_id": client_order_id(plan, order),
            "instrument_id": order.instrument_id,
            "model_name": plan.model_name,
            "signal_day": plan.signal_day,
            "side": order.side,
            "qty": order.qty,
            "prob_up": order.prob_up,
            "reason": order.reason,
            "status": "planned",
        }
        if submit and client is not None:
            response = client.post(
                f"{client.trading_url}/v2/orders",
                {
                    "symbol": order.symbol,
                    "qty": str(order.qty),
                    "side": order.side,
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": record["client_order_id"],
                },
            )
            record |= {
                "alpaca_order_id": response["id"],
                "status": "submitted",
                "submitted_at": datetime.now(UTC),
            }
        statement = insert(Order).values(record)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["client_order_id"],
                set_={key: statement.excluded[key] for key in record if key != "client_order_id"},
            )
        )
        saved.append(record)
    session.commit()
    return saved


def sync_orders(session: Session, client: AlpacaClient) -> int:
    """Ask Alpaca what happened to each open order, and save fills."""
    open_orders = session.scalars(select(Order).where(Order.status.in_(OPEN_STATUSES))).all()
    for order in open_orders:
        remote = client.get(
            f"{client.trading_url}/v2/orders:by_client_order_id",
            params={"client_order_id": order.client_order_id},
        )
        order.status = remote["status"]
        order.filled_qty = int(float(remote.get("filled_qty") or 0))
        price = remote.get("filled_avg_price")
        order.filled_avg_price = Decimal(str(price)) if price else None
    session.commit()
    return len(open_orders)


# ---------- The whole run ----------
def run_bot(
    session: Session,
    client: AlpacaClient | None,
    settings: Settings,
    *,
    model_name: str = "logistic",
    threshold: float = 0.5,
    limits: RiskLimits | None = None,
    now: datetime | None = None,
    submit: bool = False,
) -> Plan:
    check_paper_only(settings)
    if kill_switch_on():
        raise BotStoppedError("Kill switch is on. Remove KILL_SWITCH / DALAL_KILL_SWITCH to trade.")

    latest = latest_features(session)
    if latest.empty:
        raise BotStoppedError("No features yet. Run dalal-features first.")
    signal_day = latest["day"].iloc[0]
    check_fresh(signal_day, now or datetime.now(UTC))

    history = load_dataset(session)
    learner = make_learner(model_name)
    learner.fit(history, history["target_up"].to_numpy(dtype=bool))
    latest["prob_up"] = learner.predict_proba(latest)

    if client is None:
        raise BotStoppedError("Alpaca keys are needed to read the account.")
    equity, last_equity, positions = account_state(client)
    plan = plan_orders(
        predictions=latest,
        positions=positions,
        closes=latest_closes(session),
        equity=equity,
        last_equity=last_equity,
        threshold=threshold,
        limits=limits or RiskLimits(),
        signal_day=signal_day,
        model_name=model_name,
    )
    record_and_submit(session, client, plan, submit)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trade the model's daily picks.")
    parser.add_argument("--submit", action="store_true", help="send orders (paper account)")
    parser.add_argument("--sync", action="store_true", help="update fills for open orders")
    parser.add_argument("--model", default="logistic", choices=["logistic", "neural_net"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    client = AlpacaClient(settings)
    with new_session() as session:
        if args.sync:
            print(f"Checked {sync_orders(session, client)} open orders.")
            return
        try:
            plan = run_bot(
                session,
                client,
                settings,
                model_name=args.model,
                threshold=args.threshold,
                submit=args.submit,
            )
        except BotStoppedError as stop:
            log.warning("bot_stopped", reason=str(stop))
            raise SystemExit(f"Stopped: {stop}") from stop

    print(f"Signal day {plan.signal_day}, model {plan.model_name}, equity ${plan.equity:,.2f}")
    for order in plan.orders:
        print(f"  {order.side:4} {order.qty:>5} {order.symbol:6} {order.reason}")
    for note in plan.blocked:
        print(f"  BLOCKED  {note}")
    if not plan.orders:
        print("  No changes needed.")
    print("Sent to Alpaca (paper)." if args.submit else "Dry run: nothing sent. Add --submit.")


if __name__ == "__main__":
    main()
