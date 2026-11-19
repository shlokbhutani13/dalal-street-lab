"""Step 13: numbers Prometheus collects every few seconds, so Grafana can chart them.

Prometheus "scrapes" (visits) each service's /metrics page and stores the numbers over
time. Three kinds of number:
    Counter    only goes up (trades received). Grafana turns it into "per second".
    Gauge      goes up and down (connected yes/no, account equity).
    Histogram  counts how long things took, in buckets, so we can ask for the slowest 5%.
"""

from collections.abc import Iterator

import redis
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

# ---------- Live stream ----------
STREAM_TRADES = Counter("dalal_stream_trades", "Live trades received", ["symbol"])
STREAM_RECONNECTS = Counter("dalal_stream_reconnects", "Times the websocket had to reconnect")
STREAM_CONNECTED = Gauge("dalal_stream_connected", "1 while the websocket is connected")

# ---------- Web API ----------
API_LATENCY = Histogram(
    "dalal_api_request_seconds",
    "Time taken to answer an API request",
    ["method", "route", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# ---------- Paper-trading bot ----------
# The bot runs once a day and exits, so it can't be scraped. It leaves its latest account
# numbers in Redis instead, and the API reports them on its /metrics page.
BOT_EQUITY_KEY = "bot:equity"
BOT_LAST_EQUITY_KEY = "bot:last_equity"


class BotAccountCollector(Collector):
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url

    def collect(self) -> Iterator[GaugeMetricFamily]:
        try:
            client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            equity, last_equity = client.mget(BOT_EQUITY_KEY, BOT_LAST_EQUITY_KEY)
        except redis.RedisError:
            return  # Redis down: report nothing rather than a wrong number
        if equity is None or last_equity is None:
            return
        yield GaugeMetricFamily("dalal_bot_equity_dollars", "Paper account value", float(equity))
        yield GaugeMetricFamily(
            "dalal_bot_daily_pnl_dollars",
            "Paper account change since yesterday's close",
            float(equity) - float(last_equity),
        )
