"""The bot's safety rules and order logic. Alpaca is faked, so no real orders exist."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest
import responses
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalal_street_lab import bot
from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.bot import (
    BotStoppedError,
    Plan,
    PlannedOrder,
    RiskLimits,
    check_fresh,
    check_paper_only,
    plan_orders,
    record_and_submit,
    sync_orders,
)
from dalal_street_lab.config import Settings
from dalal_street_lab.models import Instrument, Order

PAPER = "https://paper-api.alpaca.markets"


def settings(url: str = PAPER) -> Settings:
    return Settings(
        postgres_password=SecretStr("x"),
        alpaca_api_key="key",
        alpaca_secret_key=SecretStr("secret"),
        alpaca_trading_url=url,
    )


def predictions(probs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": s, "instrument_id": i, "prob_up": p} for i, (s, p) in enumerate(probs.items())]
    )


def plan(**overrides: Any) -> Plan:
    arguments: dict[str, Any] = {
        "predictions": predictions({"INFY": 0.6, "HDB": 0.4, "GLD": 0.55}),
        "positions": {},
        "closes": {"INFY": 20.0, "HDB": 60.0, "GLD": 200.0},
        "equity": 100_000.0,
        "last_equity": 100_000.0,
        "threshold": 0.5,
        "limits": RiskLimits(),
        "signal_day": date(2026, 11, 17),
        "model_name": "logistic",
    }
    return plan_orders(**(arguments | overrides))


def test_refuses_anything_but_the_paper_account() -> None:
    check_paper_only(settings())
    with pytest.raises(BotStoppedError, match="not https://paper-api"):
        check_paper_only(settings("https://api.alpaca.markets"))


def test_kill_switch_stops_the_bot(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    monkeypatch.setenv("DALAL_KILL_SWITCH", "1")
    with pytest.raises(BotStoppedError, match="Kill switch"):
        bot.run_bot(session, None, settings())


def test_stale_data_is_refused() -> None:
    after_tuesday_close = datetime(2026, 11, 17, 22, 0, tzinfo=UTC)
    check_fresh(date(2026, 11, 17), after_tuesday_close)
    with pytest.raises(BotStoppedError, match="Stale data"):
        check_fresh(date(2026, 11, 16), after_tuesday_close)


def test_buys_only_confident_picks_within_the_position_cap() -> None:
    result = plan()
    orders = {order.symbol: order for order in result.orders}

    assert set(orders) == {"INFY", "GLD"}
    # Two picks share 60% of $100k, but each is capped at 10%: $10,000 apiece.
    assert orders["INFY"].qty == 500  # $10,000 / $20
    assert orders["GLD"].qty == 50  # $10,000 / $200


def test_total_exposure_is_capped() -> None:
    many = predictions({f"S{i}": 0.9 for i in range(10)})
    closes = {f"S{i}": 10.0 for i in range(10)}

    result = plan(predictions=many, closes=closes)

    invested = sum(order.qty * 10.0 for order in result.orders)
    assert invested <= 0.60 * 100_000


def test_sells_what_the_model_no_longer_likes_and_sells_come_first() -> None:
    result = plan(positions={"HDB": 30})

    assert result.orders[0].symbol == "HDB"
    assert result.orders[0].side == "sell"
    assert result.orders[0].qty == 30


def test_daily_loss_stop_blocks_buys_but_allows_sells() -> None:
    result = plan(positions={"HDB": 30}, equity=97_000.0, last_equity=100_000.0)

    assert [order.side for order in result.orders] == ["sell"]
    assert any("Daily loss stop" in note for note in result.blocked)


def make_instrument(session: Session) -> Instrument:
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
    return instrument


@responses.activate
def test_dry_run_saves_the_plan_without_calling_alpaca(session: Session) -> None:
    instrument = make_instrument(session)
    the_plan = Plan(date(2026, 11, 17), "logistic", 100_000.0)
    the_plan.orders.append(PlannedOrder("INFY", instrument.id, "buy", 5, 0.6, "test"))

    record_and_submit(session, AlpacaClient(settings()), the_plan, submit=False)

    assert len(responses.calls) == 0
    saved = session.scalars(select(Order)).one()
    assert saved.status == "planned"


@responses.activate
def test_submitted_orders_are_named_so_they_cannot_be_duplicated(session: Session) -> None:
    instrument = make_instrument(session)
    responses.post(f"{PAPER}/v2/orders", json={"id": "alpaca-123", "status": "accepted"})
    the_plan = Plan(date(2026, 11, 17), "logistic", 100_000.0)
    the_plan.orders.append(PlannedOrder("INFY", instrument.id, "buy", 5, 0.6, "test"))

    record_and_submit(session, AlpacaClient(settings()), the_plan, submit=True)

    sent = responses.calls[0].request.body
    text = sent.decode() if isinstance(sent, bytes) else str(sent)
    assert '"client_order_id": "dsl-20261117-INFY-buy"' in text
    saved = session.scalars(select(Order)).one()
    assert (saved.status, saved.alpaca_order_id) == ("submitted", "alpaca-123")


@responses.activate
def test_sync_saves_fills(session: Session) -> None:
    instrument = make_instrument(session)
    session.add(
        Order(
            client_order_id="dsl-20261117-INFY-buy",
            instrument_id=instrument.id,
            model_name="logistic",
            signal_day=date(2026, 11, 17),
            side="buy",
            qty=5,
            reason="test",
            status="submitted",
        )
    )
    session.flush()
    responses.get(
        f"{PAPER}/v2/orders:by_client_order_id",
        json={"status": "filled", "filled_qty": "5", "filled_avg_price": "19.87"},
    )

    assert sync_orders(session, AlpacaClient(settings())) == 1
    saved = session.scalars(select(Order)).one()
    assert (saved.status, saved.filled_qty, saved.filled_avg_price) == (
        "filled",
        5,
        Decimal("19.87"),
    )
