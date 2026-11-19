# Dalal Street Lab

[![CI](https://github.com/shlokbhutani13/dalal-street-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/dalal-street-lab/actions/workflows/ci.yml)

A live AI paper-trading system for the Indian market, built the way a real trading data team would build it. It pulls prices and news, checks the data, scores the news with FinBERT, builds features without peeking at the future, tests models walk-forward, backtests them with trading costs and paper-trades the result, with a live dashboard, monitoring and drift checks on top.

> Paper trading only, with fake money. Nothing here is investment advice.

## Why "India through the US"?

Indian brokers need Indian residency, so this watches India through US-listed securities on a free [Alpaca](https://alpaca.markets) paper account:

- **India ETFs:** INDY (tracks the Nifty 50), INDA
- **Indian companies listed in the US:** Infosys, HDFC Bank, ICICI Bank, Wipro, Dr. Reddy's, MakeMyTrip
- **Commodities:** gold (GLD), silver (SLV), crude oil (USO)

The list lives in [`watchlist.toml`](watchlist.toml), so changing it needs no code changes.

## What's built

| Step | What it does | Command |
| --- | --- | --- |
| 1 | Project setup: uv, ruff, strict mypy, pre-commit, typed settings, structured logs, TimescaleDB and Redis in Docker | `dalal-check` |
| 2 | Database tables and migrations (SQLAlchemy 2 + Alembic). Price tables are TimescaleDB hypertables. | `alembic upgrade head` |
| 3 | Loads the watchlist into the database. Old symbols are retired, never deleted. | `dalal-instruments` |
| 4 | Downloads daily and minute price history from Alpaca, with retries, rate limiting and resume | `dalal-backfill` |
| 5 | Data quality checks with pandera and the real NYSE calendar (holidays, half days). Fails loudly if a rule breaks. | `dalal-quality` |
| 6 | Live prices over a websocket into Redis and the database, with auto-reconnect and clean shutdown | `dalal-stream`, `dalal-watch` |
| 7 | Downloads news and scores each article's mood with FinBERT (runs on CPU, CUDA or Apple MPS) | `dalal-news`, `dalal-sentiment` |
| 8 | Builds 12 daily features (returns, volatility, RSI, volume, market move, news mood) and next-day targets | `dalal-features` |
| 9 | Walk-forward testing (6-month folds, 1-day gap) of logistic regression and a small PyTorch net against two baselines, tracked in MLflow | `dalal-train` |
| 10 | Backtest of the predictions with spread and slippage, compared with buy-and-hold | `dalal-backtest` |
| 11 | Paper-trading bot: 10% per stock, 60% total, 2% daily loss stop, kill switch, no duplicate orders | `dalal-bot` |
| 12 | FastAPI service and a live dashboard (candles, model odds, screener, news, trades over a websocket) | `dalal-api` |
| 13 | Prometheus metrics and a Grafana dashboard (stream health, API speed, bot equity and P&L) | `docker compose up` |
| 14 | Docker image (CPU-only PyTorch, non-root), built and published to GHCR by CI | `docker run` |
| 15 | Kubernetes Helm chart: API, a single-copy stream, migrations on upgrade, nightly/bot/drift CronJobs | `helm install` |
| 16 | Nightly Airflow pipeline. If the drift check fails, the bot sits that day out. | `airflow/dags` |
| 17 | Drift checks: feature PSI and recent AUC vs the backtest | `dalal-drift` |

## Rules I stuck to

- **No peeking at the future.** A news story counts toward the next closing bell after it was published. A day only counts once the market has closed. Missing days stay missing instead of being filled in. A test rebuilds the features as of a past date and checks nothing changes.
- **Every result says where it came from:** which model scored the news and which version of the features was used.
- **Expect humble results.** On real daily data a model should be barely better than a coin flip (AUC around 0.50–0.53). Anything much higher usually means a leak.
- **Daily models only.** Free minute data is too patchy for most of these symbols, so minute bars are only for the live view.
- **Secrets stay in `.env`**, which is never committed.

## Run it

You need Docker, [uv](https://docs.astral.sh/uv/) and a free Alpaca paper account.

```bash
cp .env.example .env          # add your Alpaca paper keys and a database password
docker compose up -d          # TimescaleDB + Redis
uv sync
uv run alembic upgrade head
uv run dalal-instruments && uv run dalal-backfill && uv run dalal-quality
uv run dalal-news && uv run dalal-sentiment && uv run dalal-features
uv run dalal-train && uv run dalal-backtest
uv run dalal-bot              # dry run: prints the orders it would place
uv run dalal-api              # dashboard at http://localhost:8000
uv run dalal-drift
uv run dalal-stream           # live prices (market hours)
```

The bot only sends orders with `--submit`, and only ever to Alpaca's paper URL. `DALAL_KILL_SWITCH=1` or a file named `KILL_SWITCH` stops it.

## Run it on Kubernetes

```bash
kubectl create secret generic dalal-secrets \
  --from-literal=POSTGRES_PASSWORD=... --from-literal=ALPACA_API_KEY=... --from-literal=ALPACA_SECRET_KEY=...
helm install dalal deploy/helm/dalal-street-lab --set config.POSTGRES_HOST=<your db>
```

TimescaleDB and Redis are not in the chart: point it at ones you already run. The bot CronJob is a dry run until you set `bot.submit=true`. If you'd rather use Airflow, set `nightly.enabled=false` and load `airflow/dags/dalal_nightly.py`.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run mypy src tests
```

Tests run against a real throwaway database built from the actual migrations. The live stream is tested against a fake websocket server and fake Redis, and FinBERT is tested with a tiny local model, so no test needs the internet.

CI also lints and renders the Helm chart, checks it against the Kubernetes schemas, and loads the Airflow DAG.

## License

MIT
