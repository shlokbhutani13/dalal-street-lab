"""Checks the Airflow DAG loads and runs its steps in the right order.

Run with Airflow installed (CI does this in its own job):  python -m pytest airflow/test_dag.py
"""

import importlib.util
from pathlib import Path
from types import ModuleType

DAG_FILE = Path(__file__).parent / "dags" / "dalal_nightly.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dalal_nightly", DAG_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream(dag, task_id: str) -> set[str]:  # type: ignore[no-untyped-def]
    return set(dag.get_task(task_id).upstream_task_ids)


def test_dag_loads_with_every_step() -> None:
    dag = load().dag

    assert dag.dag_id == "dalal_nightly"
    assert set(dag.task_ids) == {
        "backfill", "quality", "news", "sentiment", "features",
        "train", "backtest", "drift", "bot", "sync_fills",
    }  # fmt: skip


def test_steps_run_in_order() -> None:
    dag = load().dag

    assert upstream(dag, "features") == {"quality", "sentiment"}
    assert upstream(dag, "train") == {"features"}
    assert upstream(dag, "drift") == {"backtest"}
    assert upstream(dag, "bot") == {"drift"}  # no trading on a day drift was found


def test_bot_is_a_dry_run_unless_switched_on() -> None:
    bot = load().dag.get_task("bot")

    assert "dalal_bot_submit" in bot.bash_command
    assert "--submit' if" in bot.bash_command
