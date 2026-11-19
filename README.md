# Dalal Street Lab

[![CI](https://github.com/shlokbhutani13/dalal-street-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/dalal-street-lab/actions/workflows/ci.yml)

A live AI paper-trading system for the Indian market, built the way a real trading data team would build it. It pulls prices and news, checks the data, scores the news with FinBERT, builds features without peeking at the future, and (next) trains models and paper-trades them.

> **Work in progress.** Steps 1–8 are done. The model, backtest and paper-trading bot are next. Paper trading only, with fake money. Nothing here is investment advice.

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
uv run dalal-stream           # live prices (market hours)
```

## Tests

```bash
uv run pytest
uv run ruff check . && uv run mypy src tests
```

Tests run against a real throwaway database built from the actual migrations. The live stream is tested against a fake websocket server and fake Redis, and FinBERT is tested with a tiny local model, so no test needs the internet.

## What's next

9. Walk-forward model testing (logistic regression and a PyTorch net vs. simple baselines), tracked in MLflow
10. Backtest with trading costs
11. Paper-trading bot with risk limits (position caps, daily loss stop, kill switch)
12. FastAPI service and live dashboard
13. Prometheus and Grafana monitoring
14. Docker images, then Kubernetes with Helm
15. Nightly Airflow pipeline
16. Watching for model drift

## License

MIT
