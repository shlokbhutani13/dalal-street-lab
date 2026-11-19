"""Step 18: is the model still seeing the world it was trained on?

Run it with:  uv run dalal-drift        (exits with code 1 if something drifted)

Two checks:

1. Feature drift (PSI, the "population stability index"). Split a feature's training
   values into 10 equal-sized buckets. If recent days fall into those buckets in very
   different amounts, the market has changed character:
        PSI below 0.1   steady
        0.1 to 0.25     shifting, keep an eye on it
        above 0.25      drifted: the model is guessing outside its experience

2. Performance drift. AUC over the most recent days vs AUC over the whole backtest.
   Daily AUC is noisy, so only a clear drop (more than 0.05) raises an alert.
"""

import os
import sys

import mlflow
import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalal_street_lab.db import new_session
from dalal_street_lab.features import FEATURE_VERSION, FEATURES
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import FeatureRow, Prediction
from dalal_street_lab.training import DEFAULT_TRACKING_URI

log = structlog.get_logger()

RECENT_DAYS = 60
PSI_WARN = 0.10
PSI_ALERT = 0.25
AUC_DROP_ALERT = 0.05
EXPERIMENT = "dalal-drift"


def psi(reference: NDArray[np.float64], recent: NDArray[np.float64], buckets: int = 10) -> float:
    """Population stability index between two samples of one feature."""
    reference, recent = reference[~np.isnan(reference)], recent[~np.isnan(recent)]
    if len(reference) < buckets or len(recent) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:  # the feature barely varies (e.g. mostly zero news): use its values
        edges = np.unique(np.concatenate([reference, recent]))
        edges = np.append(edges, edges[-1] + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(recent, edges)[0] / len(recent)
    expected, actual = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_drift(features: pd.DataFrame, recent_days: int = RECENT_DAYS) -> pd.DataFrame:
    days = sorted(features["day"].unique())
    cutoff = days[-recent_days] if len(days) > recent_days else days[0]
    reference = features[features["day"] < cutoff]
    recent = features[features["day"] >= cutoff]
    rows = []
    for name in FEATURES:
        value = psi(reference[name].to_numpy(dtype=float), recent[name].to_numpy(dtype=float))
        status = "alert" if value > PSI_ALERT else "watch" if value > PSI_WARN else "ok"
        rows.append({"feature": name, "psi": value, "status": status})
    return pd.DataFrame(rows).set_index("feature")


def performance_drift(predictions: pd.DataFrame, recent_days: int = RECENT_DAYS) -> pd.DataFrame:
    """Per model: AUC on every out-of-sample day vs AUC on just the most recent days."""
    rows = []
    for name, group in predictions.groupby("model_name", sort=True):
        days = sorted(group["day"].unique())
        cutoff = days[-recent_days] if len(days) > recent_days else days[0]
        recent = group[group["day"] >= cutoff]

        def auc(frame: pd.DataFrame) -> float:
            labels = frame["target_up"].to_numpy(dtype=bool)
            if len(np.unique(labels)) < 2:
                return float("nan")
            return float(roc_auc_score(labels, frame["prob_up"].to_numpy()))

        overall, latest = auc(group), auc(recent)
        dropped = not np.isnan(latest) and overall - latest > AUC_DROP_ALERT
        rows.append(
            {
                "model": name,
                "backtest_auc": overall,
                "recent_auc": latest,
                "status": "alert" if dropped else "ok",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def load(session: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = session.scalars(
        select(FeatureRow).where(FeatureRow.feature_version == FEATURE_VERSION)
    ).all()
    features = pd.DataFrame(
        [{"day": r.day, **{name: getattr(r, name) for name in FEATURES}} for r in feature_rows]
    )
    guesses = session.execute(
        select(Prediction.model_name, Prediction.day, Prediction.prob_up, FeatureRow.target_up)
        .join(
            FeatureRow,
            (FeatureRow.instrument_id == Prediction.instrument_id)
            & (FeatureRow.day == Prediction.day)
            & (FeatureRow.feature_version == FEATURE_VERSION),
        )
        .where(FeatureRow.target_up.is_not(None))
    ).all()
    predictions = pd.DataFrame(guesses, columns=["model_name", "day", "prob_up", "target_up"])
    return features, predictions


def main() -> None:
    configure_logging()
    with new_session() as session:
        features, predictions = load(session)
    if features.empty:
        raise SystemExit("No features yet. Run: uv run dalal-features")

    by_feature = feature_drift(features)
    by_model = performance_drift(predictions) if not predictions.empty else pd.DataFrame()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics(
            {f"psi.{name}": float(v) for name, v in by_feature["psi"].items() if not np.isnan(v)}
        )
        for name, row in by_model.iterrows():
            if not np.isnan(row["recent_auc"]):
                mlflow.log_metric(f"recent_auc.{name}", float(row["recent_auc"]))

    print("Feature drift (PSI, recent vs earlier):")
    print(by_feature.round(3).to_string())
    if not by_model.empty:
        print("\nModel performance (AUC):")
        print(by_model.round(3).to_string())

    alerts = int((by_feature["status"] == "alert").sum())
    if not by_model.empty:
        alerts += int((by_model["status"] == "alert").sum())
    log.info("drift_checked", alerts=alerts)
    if alerts:
        print(f"\n{alerts} alert(s). Consider retraining, or pausing the bot.")
        sys.exit(1)
    print("\nNo drift alerts.")


if __name__ == "__main__":
    main()
