"""How the rest of the app connects to the database."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from dalal_street_lab.config import get_settings


def make_engine(url: str) -> Engine:
    # pool_pre_ping: test each pooled connection before using it, so a database
    # restart doesn't cause "connection closed" errors in the app.
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    """One engine (a pool of reusable connections) for the whole app."""
    return make_engine(get_settings().sqlalchemy_url)


def new_session() -> Session:
    """A session is one conversation with the database: reads, writes, then commit."""
    return sessionmaker(get_engine(), expire_on_commit=False)()
