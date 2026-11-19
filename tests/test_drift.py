"""Drift checks, on fake data where we control whether anything changed."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from dalal_street_lab.drift import feature_drift, performance_drift, psi
from dalal_street_lab.features import FEATURES

rng = np.random.default_rng(5)


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift() -> None:
    reference = rng.normal(0, 1, 5_000)

    assert psi(reference, rng.normal(0, 1, 1_000)) < 0.05
    assert psi(reference, rng.normal(1.5, 1, 1_000)) > 0.25


def test_psi_copes_with_a_feature_that_is_mostly_zero() -> None:
    news_counts = np.concatenate([np.zeros(900), np.ones(100)])

    assert psi(news_counts, news_counts) == pytest.approx(0.0, abs=1e-6)


def features(days: int, shift_last: int = 0, stocks: int = 10) -> pd.DataFrame:
    """One row per stock per day, like the real features table."""
    start = date(2023, 1, 2)
    rows = []
    for i in range(days):
        shifted = i >= days - shift_last
        for _ in range(stocks):
            values = {name: rng.normal(2.0 if shifted else 0.0, 1.0) for name in FEATURES}
            rows.append({"day": start + timedelta(i)} | values)
    return pd.DataFrame(rows)


def test_feature_drift_flags_only_a_changed_market() -> None:
    steady = feature_drift(features(400))
    changed = feature_drift(features(400, shift_last=60))

    assert (steady["status"] != "alert").all()
    assert (changed["status"] == "alert").all()


def test_performance_drift_flags_a_model_that_stopped_working() -> None:
    start = date(2023, 1, 2)
    rows = []
    for i in range(300):
        up = bool(rng.random() > 0.5)
        stopped_working = i >= 240
        guess = rng.random() if stopped_working else (0.8 if up else 0.2)
        rows.append(
            {"model_name": "m", "day": start + timedelta(i), "prob_up": guess, "target_up": up}
        )

    result = performance_drift(pd.DataFrame(rows)).loc["m"].to_dict()

    assert result["backtest_auc"] > result["recent_auc"] + 0.05
    assert result["status"] == "alert"
