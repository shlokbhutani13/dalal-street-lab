"""`uv run dalal-check`: prove the database and Redis are up and reachable."""

import sys
from collections.abc import Callable

import psycopg
import redis

from dalal_street_lab.config import get_settings


def check_postgres(url: str) -> str:
    with psycopg.connect(url, connect_timeout=5) as conn:
        version = conn.execute("SHOW server_version").fetchone()
        timescale = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        ).fetchone()
    pg = version[0] if version else "unknown"
    ts = timescale[0] if timescale else "NOT INSTALLED"
    return f"PostgreSQL {pg}, TimescaleDB {ts}"


def check_redis(url: str) -> str:
    client = redis.Redis.from_url(url, socket_connect_timeout=5)
    client.ping()
    return f"Redis {client.info()['redis_version']}"


def main() -> None:
    settings = get_settings()
    checks: dict[str, Callable[[], str]] = {
        "database": lambda: check_postgres(settings.database_url),
        "redis": lambda: check_redis(settings.redis_url),
    }
    failed = False
    for name, check in checks.items():
        try:
            print(f"OK    {name:<9} {check()}")
        except Exception as error:  # report every failure, then exit with an error code
            print(f"FAIL  {name:<9} {type(error).__name__}: {error}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
