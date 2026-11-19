"""Backtest math, checked on tiny examples where the answer is easy to work out by hand."""

from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from dalal_street_lab.backtest import (
    Costs,
    daily_weights,
    log_to_mlflow,
    portfolio_returns,
    run_backtests,
    run_model,
    threshold_scan,
)

FREE = Costs(commission_bps=0, half_spread_bps=0, slippage_bps=0)


def frame(rows: list[tuple[str, str, int, float, float]]) -> pd.DataFrame:
    """(model, symbol, day number, prob_up, next day's return)."""
    start = date(2024, 1, 2)
    return pd.DataFrame(
        [
            {
                "model_name": m,
                "symbol": s,
                "day": start + timedelta(d),
                "prob_up": p,
                "next_return": r,
            }
            for m, s, d, p, r in rows
        ]
    )


def test_each_symbol_gets_an_equal_slice_only_when_the_model_is_confident() -> None:
    data = frame([("m", "A", 0, 0.6, 0.0), ("m", "B", 0, 0.4, 0.0)])

    weights = daily_weights(data, threshold=0.5)

    first_day = weights.iloc[0].to_dict()
    assert first_day["A"] == 0.5
    assert first_day["B"] == 0.0


def test_costs_are_paid_on_every_buy_and_sell() -> None:
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0]})
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.05]})

    daily, turnover = portfolio_returns(weights, returns, Costs(0, 0, 10))

    assert turnover.tolist() == [1.0, 0.0, 1.0]  # buy, hold, sell
    np.testing.assert_allclose(daily, [0.01 - 0.001, 0.02, -0.001])


def test_a_perfect_model_beats_buy_and_hold_and_a_backwards_one_loses() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for day in range(250):
        for symbol in ("A", "B", "C"):
            move = float(rng.normal(0, 0.01))
            rows.append(("perfect", symbol, day, 0.9 if move > 0 else 0.1, move))
            rows.append(("backwards", symbol, day, 0.1 if move > 0 else 0.9, move))
    results, curves = run_backtests(frame(rows), threshold=0.5, costs=Costs())
    sharpe = results["sharpe"].to_dict()
    total = results["total_return"].to_dict()
    hits = results["hit_rate"].to_dict()

    assert sharpe["perfect"] > sharpe["buy_and_hold"]
    assert total["backwards"] < 0
    # Not quite 100%: on days with tiny moves, trading costs eat the whole gain.
    assert hits["perfect"] > 0.95
    assert set(curves) == {"buy_and_hold", "perfect", "backwards"}


def test_drawdown_is_the_worst_fall_from_a_peak() -> None:
    data = frame([("m", "A", d, 0.9, r) for d, r in enumerate([0.10, -0.50, 0.20])])

    scores, _ = run_model(data, 0.5, FREE)

    assert scores["max_drawdown"] == pytest.approx(-0.5)
    assert scores["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)


def test_threshold_scan_shows_every_threshold_tried() -> None:
    data = frame([("m", "A", d, 0.53, 0.01) for d in range(10)])

    scan = threshold_scan(data, FREE)

    assert list(scan.index) == [0.5, 0.52, 0.55]
    assert scan["time_invested"].to_dict()[0.55] == 0.0  # 0.53 never clears 0.55


def test_backtest_is_recorded_in_mlflow(tmp_path: Path) -> None:
    data = frame([("m", "A", d, 0.6, 0.01) for d in range(10)])
    results, _ = run_backtests(data, 0.5, FREE)

    run_id = log_to_mlflow(results, 0.5, FREE, f"sqlite:///{tmp_path / 'mlflow.db'}")

    assert "m.sharpe" in mlflow.get_run(run_id).data.metrics
