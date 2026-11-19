"""Step 10: would the model's guesses have made money after trading costs?

Run it with:  uv run dalal-backtest

The rule (decided BEFORE looking at results, so we can't fool ourselves):
    every day after the close, for each symbol:
        if the model's chance of "up tomorrow" is above the threshold -> hold it tomorrow
        otherwise                                                   -> hold cash for it

Each symbol gets an equal slice of the money (1/N). Every time a slice is bought or sold
we pay costs, even though Alpaca charges $0 commission, because real trades still lose a
little to the spread (the gap between buy and sell prices) and to slippage (the price
moving while your order fills).

We compare against buying everything and holding it, and we warn if you go hunting for
the "best" threshold: trying many and keeping the winner is overfitting.
"""

import os
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalal_street_lab.db import new_session
from dalal_street_lab.features import FEATURE_VERSION
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import FeatureRow, Instrument, Prediction
from dalal_street_lab.training import DEFAULT_TRACKING_URI

log = structlog.get_logger()

TRADING_DAYS_PER_YEAR = 252
EXPERIMENT = "dalal-backtest"


@dataclass(frozen=True)
class Costs:
    """Costs in basis points (1 bp = 0.01%) of the money traded, paid on every buy or sell."""

    commission_bps: float = 0.0  # Alpaca charges nothing, but other brokers do
    half_spread_bps: float = 2.0  # you buy a little above and sell a little below the middle
    slippage_bps: float = 3.0  # the price moves a little before the order fills

    @property
    def per_trade(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000


def load_backtest_data(session: Session) -> pd.DataFrame:
    """Each prediction next to what really happened the next day (simple return)."""
    rows = session.execute(
        select(
            Prediction.model_name,
            Instrument.symbol,
            Prediction.day,
            Prediction.prob_up,
            FeatureRow.target_ret_1d,
        )
        .join(Instrument, Instrument.id == Prediction.instrument_id)
        .join(
            FeatureRow,
            (FeatureRow.instrument_id == Prediction.instrument_id)
            & (FeatureRow.day == Prediction.day)
            & (FeatureRow.feature_version == FEATURE_VERSION),
        )
        .where(FeatureRow.target_ret_1d.is_not(None))
    ).all()
    df = pd.DataFrame(rows, columns=["model_name", "symbol", "day", "prob_up", "log_return"])
    df["next_return"] = np.expm1(df["log_return"])  # log return -> normal % return
    return df.drop(columns="log_return")


def daily_weights(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rows = days, columns = symbols, values = share of money in that symbol tomorrow.

    Each symbol that has a guess that day owns a 1/N slice, held only if the guess clears
    the threshold.
    """
    probs = predictions.pivot(index="day", columns="symbol", values="prob_up").sort_index()
    slots = probs.notna().sum(axis=1).replace(0, np.nan)
    held = (probs > threshold).astype(float)
    return held.div(slots, axis=0).fillna(0.0)


def portfolio_returns(
    weights: pd.DataFrame, returns: pd.DataFrame, costs: Costs
) -> tuple[pd.Series, pd.Series]:
    """Daily portfolio return after costs, plus daily turnover (share of money traded)."""
    returns = returns.reindex_like(weights).fillna(0.0)
    gross = (weights * returns).sum(axis=1)
    previous = weights.shift(1).fillna(0.0)
    turnover = (weights - previous).abs().sum(axis=1)
    return gross - turnover * costs.per_trade, turnover


def performance(daily: pd.Series, turnover: pd.Series, holding: pd.Series) -> dict[str, float]:
    values: NDArray[np.float64] = daily.to_numpy(dtype=np.float64)
    equity = np.cumprod(1 + values)
    years = len(values) / TRADING_DAYS_PER_YEAR
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    drawdown = equity / np.maximum.accumulate(equity) - 1
    invested = holding.to_numpy() > 0
    return {
        "total_return": float(equity[-1] - 1),
        "annual_return": float(equity[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "annual_volatility": spread * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe": (
            float(np.mean(values) / spread * np.sqrt(TRADING_DAYS_PER_YEAR)) if spread else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
        "turnover_per_day": float(turnover.mean()),
        "time_invested": float(invested.mean()),
        "hit_rate": float((values[invested] > 0).mean()) if invested.any() else 0.0,
        "days": float(len(values)),
    }


def run_model(
    predictions: pd.DataFrame, threshold: float, costs: Costs
) -> tuple[dict[str, float], pd.Series]:
    """Backtest one model's predictions. Returns its scores and its equity curve."""
    returns = predictions.pivot(index="day", columns="symbol", values="next_return").sort_index()
    weights = daily_weights(predictions, threshold)
    daily, turnover = portfolio_returns(weights, returns, costs)
    scores = performance(daily, turnover, weights.sum(axis=1))
    return scores, (1 + daily).cumprod()


def buy_and_hold(predictions: pd.DataFrame, costs: Costs) -> tuple[dict[str, float], pd.Series]:
    """The benchmark: hold every symbol with an equal slice, every day."""
    everything = predictions.assign(prob_up=1.0)
    return run_model(everything, threshold=0.5, costs=costs)


def threshold_scan(
    predictions: pd.DataFrame, costs: Costs, thresholds: tuple[float, ...] = (0.5, 0.52, 0.55)
) -> pd.DataFrame:
    """How touchy the result is to the threshold. For information only: don't pick the best!"""
    rows = [{"threshold": t, **run_model(predictions, t, costs)[0]} for t in thresholds]
    return pd.DataFrame(rows).set_index("threshold")


def run_backtests(
    data: pd.DataFrame, threshold: float = 0.5, costs: Costs | None = None
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Every model plus buy-and-hold, on exactly the same days."""
    costs = costs or Costs()
    first_model = data["model_name"].iloc[0]
    benchmark, benchmark_curve = buy_and_hold(data[data["model_name"] == first_model], costs)
    results = {"buy_and_hold": benchmark}
    curves = {"buy_and_hold": benchmark_curve}
    for name, group in data.groupby("model_name", sort=True):
        results[str(name)], curves[str(name)] = run_model(group, threshold, costs)
    return pd.DataFrame(results).T, curves


def log_to_mlflow(results: pd.DataFrame, threshold: float, costs: Costs, tracking_uri: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="backtest") as run:
        mlflow.log_params(
            {
                "threshold": threshold,
                "commission_bps": costs.commission_bps,
                "half_spread_bps": costs.half_spread_bps,
                "slippage_bps": costs.slippage_bps,
                "feature_version": FEATURE_VERSION,
            }
        )
        for name, row in results.iterrows():
            mlflow.log_metrics({f"{name}.{key}": float(value) for key, value in row.items()})
        return str(run.info.run_id)


def main() -> None:
    configure_logging()
    threshold = float(os.environ.get("DALAL_THRESHOLD", "0.5"))
    costs = Costs()
    with new_session() as session:
        data = load_backtest_data(session)
    if data.empty:
        raise SystemExit("No predictions yet. Run: uv run dalal-train")

    results, _ = run_backtests(data, threshold, costs)
    run_id = log_to_mlflow(
        results, threshold, costs, os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    log.info("backtest_done", mlflow_run_id=run_id, threshold=threshold)

    shown = results[
        ["total_return", "annual_return", "sharpe", "max_drawdown", "turnover_per_day", "hit_rate"]
    ]
    print(shown.round(4).to_string())
    print(f"\nCosts: {costs.per_trade * 10_000:.1f} bp per trade. Threshold: {threshold}.")
    print(
        "Beat buy-and-hold on Sharpe AND drawdown before trusting a model. Trying many "
        "thresholds and keeping the best is overfitting: the winner is usually luck."
    )


if __name__ == "__main__":
    main()
