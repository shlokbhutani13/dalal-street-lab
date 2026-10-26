"""All settings in one place, read from environment variables (or a .env file).

Nothing secret is ever written in code. Each setting has a type, so a typo like
POSTGRES_PORT=abc fails loudly at startup instead of causing a strange bug later.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- Database (TimescaleDB = PostgreSQL + time-series superpowers) ----------
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "dalal"
    postgres_user: str = "dalal"
    postgres_password: SecretStr  # required: no default, so it MUST come from .env

    # ---------- Redis (fast in-memory store for live prices) ----------
    redis_url: str = "redis://localhost:6379/0"

    # ---------- Alpaca (paper account at alpaca.markets) ----------
    alpaca_api_key: str = ""
    alpaca_secret_key: SecretStr = SecretStr("")
    alpaca_trading_url: str = "https://paper-api.alpaca.markets"  # paper = fake money
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_feed: str = "iex"  # free accounts get real-time data from the IEX exchange
    alpaca_stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    # ---------- App ----------
    log_level: str = "INFO"
    log_format: str = "console"  # "console" on your laptop, "json" on servers

    @property
    def database_url(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        """Same address, but tells SQLAlchemy to use the psycopg (version 3) driver."""
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    """Read the settings once and reuse them everywhere."""
    return Settings()  # the pydantic mypy plugin knows values come from the environment
