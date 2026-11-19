"""Walk-forward testing: the honest way to test a trading model.

Run it with:  uv run dalal-train
See the runs: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

A normal random train/test split would let the model train on Tuesday and be tested on
Monday, which is impossible in real life. Walk-forward copies real life instead:

    train on everything up to Dec 2022 -> test on Jan-Jun 2023
    train on everything up to Jun 2023 -> test on Jul-Dec 2023
    ... and so on, one 6-month "fold" at a time.

    |-------- train --------| gap |-- test --|
    |------------ train ------------| gap |-- test --|

The gap (embargo) drops the last training day before each test window, because its
target is the first test day's move: keeping it would leak one day of the answer.
Every test guess is made by a model that never saw that day: "out of sample".
"""

import os
from dataclasses import dataclass
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray
from sklearn.metrics import log_loss, roc_auc_score
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.db import new_session
from dalal_street_lab.features import FEATURE_VERSION, FEATURES
from dalal_street_lab.learners import Learner, default_learners
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import FeatureRow, Instrument, Prediction

log = structlog.get_logger()

FIRST_TEST_DAY = date(2023, 1, 1)
FOLD_MONTHS = 6
EMBARGO_DAYS = 1
EXPERIMENT = "dalal-walk-forward"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: date  # last training day (after the embargo gap)
    test_start: date
    test_end: date  # exclusive


def load_dataset(session: Session) -> pd.DataFrame:
    """Every feature row whose answer is known, in time order."""
    rows = session.execute(
        select(Instrument.symbol, FeatureRow)
        .join(Instrument, Instrument.id == FeatureRow.instrument_id)
        .where(FeatureRow.feature_version == FEATURE_VERSION, FeatureRow.target_up.is_not(None))
    ).all()
    records = [
        {
            "symbol": symbol,
            "instrument_id": row.instrument_id,
            "day": row.day,
            **{name: getattr(row, name) for name in FEATURES},
            "target_up": bool(row.target_up),
        }
        for symbol, row in rows
    ]
    columns = ["symbol", "instrument_id", "day", *FEATURES, "target_up"]
    return pd.DataFrame(records, columns=columns).sort_values(["day", "symbol"], ignore_index=True)


def make_folds(
    days: pd.Series,
    first_test_day: date = FIRST_TEST_DAY,
    months: int = FOLD_MONTHS,
    embargo_days: int = EMBARGO_DAYS,
) -> list[Fold]:
    """Six-month test windows, from `first_test_day` until the data runs out."""
    trading_days = sorted(set(days))
    if not trading_days:
        return []
    folds: list[Fold] = []
    start = pd.Timestamp(first_test_day)
    last_day = pd.Timestamp(trading_days[-1])
    while start <= last_day:
        end = start + pd.DateOffset(months=months)
        before = [d for d in trading_days if d < start.date()]
        if len(before) > embargo_days:
            train_end = before[-1 - embargo_days]
            folds.append(Fold(len(folds) + 1, train_end, start.date(), end.date()))
        start = end
    return folds


def score(y: NDArray[np.bool_], prob_up: NDArray[np.float64]) -> dict[str, float]:
    """Accuracy: share of days the direction was right. AUC: 0.5 = coin flip, 1 = perfect.
    Log loss: punishes being confidently wrong (lower is better)."""
    clipped = np.clip(prob_up, 1e-6, 1 - 1e-6)
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(np.mean((prob_up >= 0.5) == y)),
        "auc": float(roc_auc_score(y, prob_up)) if both_classes else float("nan"),
        "log_loss": float(log_loss(y, clipped, labels=[False, True])),
        "rows": float(len(y)),
    }


def walk_forward(
    data: pd.DataFrame,
    learners: list[Learner],
    folds: list[Fold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test every learner on every fold.

    Returns (fold scores, out-of-sample predictions). Nothing here touches MLflow or the
    database, so it's easy to test.
    """
    scores, predictions = [], []
    for fold in folds:
        train = data[data["day"] <= fold.train_end]
        test = data[(data["day"] >= fold.test_start) & (data["day"] < fold.test_end)]
        if train.empty or test.empty:
            continue
        y_train = train["target_up"].to_numpy(dtype=bool)
        y_test = test["target_up"].to_numpy(dtype=bool)
        for learner in learners:
            learner.fit(train, y_train)
            prob_up = np.clip(learner.predict_proba(test), 0.0, 1.0)
            scores.append({"model": learner.name, "fold": fold.number, **score(y_test, prob_up)})
            predictions.append(
                pd.DataFrame(
                    {
                        "model_name": learner.name,
                        "instrument_id": test["instrument_id"].to_numpy(),
                        "day": test["day"].to_numpy(),
                        "prob_up": prob_up,
                        "target_up": y_test,
                    }
                )
            )
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """One line per model, scored on ALL its out-of-sample guesses together."""
    rows = []
    for name, group in predictions.groupby("model_name", sort=False):
        result = score(group["target_up"].to_numpy(dtype=bool), group["prob_up"].to_numpy())
        rows.append({"model": name, **result})
    return pd.DataFrame(rows).set_index("model")


def run_experiment(
    data: pd.DataFrame,
    learners: list[Learner],
    folds: list[Fold],
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """walk_forward(), recorded in MLflow: one parent run, one child run per model."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    fold_scores, predictions = walk_forward(data, learners, folds)
    summary = summarize(predictions)
    run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name="walk_forward"):
        mlflow.log_params(
            {
                "feature_version": FEATURE_VERSION,
                "features": ",".join(FEATURES),
                "first_test_day": str(folds[0].test_start) if folds else "",
                "fold_months": FOLD_MONTHS,
                "embargo_days": EMBARGO_DAYS,
                "folds": len(folds),
                "rows": len(data),
            }
        )
        for learner in learners:
            with mlflow.start_run(run_name=learner.name, nested=True) as child:
                run_ids[learner.name] = child.info.run_id
                for row in fold_scores[fold_scores["model"] == learner.name].to_dict("records"):
                    for metric in ("accuracy", "auc", "log_loss"):
                        if not np.isnan(row[metric]):
                            mlflow.log_metric(f"fold_{metric}", row[metric], step=int(row["fold"]))
                overall = summary.loc[learner.name].to_dict()
                mlflow.log_metrics(
                    {
                        f"oos_{metric}": float(overall[metric])
                        for metric in ("accuracy", "auc", "log_loss")
                        if not np.isnan(overall[metric])
                    }
                )
    predictions["mlflow_run_id"] = predictions["model_name"].map(run_ids)
    return summary, predictions, run_ids


def save_predictions(session: Session, predictions: pd.DataFrame) -> int:
    """Predictions are derived data, so each model's rows are rebuilt, not patched."""
    names = list(predictions["model_name"].unique())
    records = [
        {
            "model_name": row["model_name"],
            "instrument_id": int(row["instrument_id"]),
            "day": row["day"],
            "prob_up": float(row["prob_up"]),
            "feature_version": FEATURE_VERSION,
            "mlflow_run_id": row["mlflow_run_id"],
        }
        for row in predictions.to_dict("records")
    ]
    session.execute(delete(Prediction).where(Prediction.model_name.in_(names)))
    for start in range(0, len(records), 2000):
        session.execute(insert(Prediction).values(records[start : start + 2000]))
    session.commit()
    return len(records)


def main() -> None:
    configure_logging()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    with new_session() as session:
        data = load_dataset(session)
        if data.empty:
            raise SystemExit("No features with known answers. Run: uv run dalal-features")
        folds = make_folds(data["day"])
        if not folds:
            raise SystemExit(f"Not enough data before {FIRST_TEST_DAY} to start testing.")
        summary, predictions, _ = run_experiment(
            data, default_learners(FEATURES), folds, tracking_uri
        )
        saved = save_predictions(session, predictions)

    log.info("predictions_saved", rows=saved, folds=len(folds), tracking_uri=tracking_uri)
    print(summary.drop(columns="rows").round(4).to_string())
    print("\nAUC near 0.50 is normal for daily returns. Above 0.60 usually means a leak.")


if __name__ == "__main__":
    main()
