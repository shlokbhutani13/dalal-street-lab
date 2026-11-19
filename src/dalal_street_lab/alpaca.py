"""One shared, polite HTTP client for Alpaca: logs in with API keys, retries, and paces itself."""

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dalal_street_lab.config import Settings

# Alpaca's free plan allows 200 requests per minute. One every 0.35 s is about 170 per minute.
SECONDS_BETWEEN_REQUESTS = 0.35


class AlpacaAuthError(Exception):
    """Alpaca rejected our keys: check ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."""


class AlpacaClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.alpaca_api_key or not settings.alpaca_secret_key.get_secret_value():
            raise AlpacaAuthError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env first.")
        self.trading_url = settings.alpaca_trading_url
        self.data_url = settings.alpaca_data_url
        self.feed = settings.alpaca_feed
        self._last_request = 0.0

        # Retry temporary problems (rate limit 429, server errors 5xx) with growing waits.
        retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.http = requests.Session()
        self.http.mount("https://", HTTPAdapter(max_retries=retry))
        # Every request carries our keys in these two headers.
        self.http.headers.update(
            {
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret_key.get_secret_value(),
                "Accept": "application/json",
            }
        )

    def _pace(self) -> None:
        # Be polite: never send requests faster than the rate limit allows.
        wait = SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self._pace()
        response = self.http.get(url, params=params, timeout=30)
        if response.status_code in (401, 403):
            raise AlpacaAuthError("Alpaca rejected the API keys. Check them in .env.")
        response.raise_for_status()
        return response.json()

    def post(self, url: str, body: dict[str, Any]) -> Any:
        """Send an order. POSTs are never retried automatically (urllib3 skips them), so a
        network hiccup can't place the same order twice."""
        self._pace()
        response = self.http.post(url, json=body, timeout=30)
        if response.status_code in (401, 403):
            raise AlpacaAuthError("Alpaca rejected the API keys. Check them in .env.")
        response.raise_for_status()
        return response.json()
