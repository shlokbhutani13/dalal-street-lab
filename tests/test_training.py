"""Walk-forward testing, checked on fake data where we know the right answer."""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalal_street_lab.features import FEATURES
from dalal_street_lab.learners import AlwaysUp, Learner, Logistic, Momentum, NeuralNet
from dalal_street_lab.models import Instrument, Prediction
from dalal_street_lab.training import (
    make_folds,
    run_experiment,
    save_predictions,
    score,
    summarize,
    walk_forward,
)


def planted_signal(days: int = 900, symbols: int = 3, seed: int = 1) -> pd.DataFrame:
    """Random features, except tomorrow really does tend to follow today's news mood.

    A model that learns should find the pattern. A coin flip shouldn't.
    """
    rng = np.random.default_rng(seed)
    start = date(2020, 8, 3)
    weekdays = [d for d in (start + timedelta(i) for i in range(days * 2)) if d.weekday() < 5]
    rows = []
    for instrument_id in range(1, symbols + 1):
        for day in weekdays[:days]:
            values = {name: rng.normal() for name in FEATURES}
            values["volatility_20d"] = abs(values["volatility_20d"]) + 0.5
            target = values["news_mood_1d"] + 0.8 * rng.normal() > 0
            rows.append(
                {"symbol": f"S{instrument_id}", "instrument_id": instrument_id, "day": day}
                | values
                | {"target_up": target}
            )
    return pd.DataFrame(rows).sort_values(["day", "symbol"], ignore_index=True)


def test_folds_never_let_training_overlap_testing() -> None:
    data = planted_signal(days=700)
    folds = make_folds(data["day"])

    assert folds[0].test_start == date(2023, 1, 1)
    trading_days = sorted(set(data["day"]))
    for fold in folds:
        assert fold.train_end < fold.test_start
        # The embargo: exactly one trading day sits between training and testing.
        gap = [d for d in trading_days if fold.train_end < d < fold.test_start]
        assert len(gap) == 1
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_end == later.test_start  # test windows touch but never overlap


def test_every_prediction_is_out_of_sample() -> None:
    data = planted_signal(days=700)
    folds = make_folds(data["day"])

    _, predictions = walk_forward(data, [AlwaysUp()], folds)

    assert predictions["day"].min() >= date(2023, 1, 1)
    assert not predictions.duplicated(["model_name", "instrument_id", "day"]).any()
    assert ((predictions["prob_up"] >= 0) & (predictions["prob_up"] <= 1)).all()


def test_real_models_find_a_planted_signal_and_baselines_do_not() -> None:
    data = planted_signal()
    folds = make_folds(data["day"])
    learners: list[Learner] = [
        AlwaysUp(),
        Momentum(),
        Logistic(FEATURES),
        NeuralNet(FEATURES, max_epochs=30),
    ]

    _, predictions = walk_forward(data, learners, folds)
    auc = summarize(predictions)["auc"].to_dict()

    assert auc["logistic"] > 0.75
    assert auc["neural_net"] > 0.70
    assert auc["momentum"] == pytest.approx(0.5, abs=0.05)
    # Pooled over folds its constant guess shifts a little, so it hovers near 0.5.
    assert auc["always_up"] == pytest.approx(0.5, abs=0.05)


def test_the_neural_net_is_repeatable() -> None:
    data = planted_signal(days=300)
    y = data["target_up"].to_numpy(dtype=bool)

    first, second = NeuralNet(FEATURES, max_epochs=5), NeuralNet(FEATURES, max_epochs=5)
    first.fit(data, y)
    second.fit(data, y)

    np.testing.assert_allclose(first.predict_proba(data), second.predict_proba(data))


def test_scores_reward_good_guesses() -> None:
    y = np.array([True, False, True, False])
    good = score(y, np.array([0.9, 0.1, 0.8, 0.2]))
    coin = score(y, np.array([0.5, 0.5, 0.5, 0.5]))

    assert good["accuracy"] == 1.0
    assert good["auc"] == 1.0
    assert good["log_loss"] < coin["log_loss"]


def test_experiment_is_recorded_in_mlflow(tmp_path: Path) -> None:
    data = planted_signal(days=700)
    folds = make_folds(data["day"])
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    summary, predictions, run_ids = run_experiment(data, [AlwaysUp(), Momentum()], folds, uri)

    assert set(run_ids) == {"always_up", "momentum"}
    assert set(predictions["mlflow_run_id"]) == set(run_ids.values())
    run = mlflow.get_run(run_ids["momentum"])
    assert "oos_accuracy" in run.data.metrics
    assert len(summary) == 2


def test_saving_twice_replaces_old_predictions(session: Session) -> None:
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
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic", "logistic"],
            "instrument_id": [instrument.id, instrument.id],
            "day": [date(2023, 1, 3), date(2023, 1, 4)],
            "prob_up": [0.55, 0.48],
            "mlflow_run_id": ["run-1", "run-1"],
        }
    )

    assert save_predictions(session, predictions) == 2
    assert save_predictions(session, predictions) == 2
    assert session.scalar(select(func.count()).select_from(Prediction)) == 2
