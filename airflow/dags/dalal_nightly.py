"""Nightly pipeline for Dalal Street Lab, as an Airflow DAG (Step 17).

Runs every weekday after the US close:

    backfill -> quality ---\
                            >-> features -> train -> backtest -> drift -> bot
    news -> sentiment -----/
    sync_fills (on its own)

Each task is one dalal-* command, so the worker needs the package installed (for example,
run Airflow from the project's Docker image). If any step fails, everything after it is
skipped. That includes the drift check: when it finds drift, the bot does not trade that day.

The bot only sends orders (to the Alpaca PAPER account) if the Airflow Variable
dalal_bot_submit is "true". Otherwise it plans and prints, nothing more.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

SUBMIT_FLAG = "{{ '--submit' if var.value.get('dalal_bot_submit', 'false') == 'true' else '' }}"

with DAG(
    dag_id="dalal_nightly",
    description="Prices, news, features, models, drift check, then the paper bot",
    schedule="30 22 * * 1-5",  # 22:30 UTC, after the US close
    start_date=datetime(2026, 1, 1),
    catchup=False,  # missed nights are not replayed: backfill already catches up on prices
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=1),
    },
    tags=["dalal", "trading"],
) as dag:

    def step(task_id: str, command: str, **kwargs: object) -> BashOperator:
        return BashOperator(task_id=task_id, bash_command=command, **kwargs)  # type: ignore[arg-type]

    backfill = step("backfill", "dalal-backfill")
    quality = step("quality", "dalal-quality", retries=0)  # bad data won't fix itself
    news = step("news", "dalal-news")
    sentiment = step("sentiment", "dalal-sentiment")
    features = step("features", "dalal-features")
    train = step("train", "dalal-train", execution_timeout=timedelta(hours=2))
    backtest = step("backtest", "dalal-backtest")
    drift = step("drift", "dalal-drift", retries=0)
    bot = step("bot", f"dalal-bot --model logistic {SUBMIT_FLAG}", retries=0)
    sync_fills = step("sync_fills", "dalal-bot --sync")

    backfill >> quality
    news >> sentiment
    [quality, sentiment] >> features >> train >> backtest >> drift >> bot
