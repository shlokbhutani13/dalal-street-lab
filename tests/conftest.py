"""Shared test setup: a separate, throwaway database for every test run."""

from collections.abc import Iterator

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dalal_street_lab.config import get_settings
from dalal_street_lab.db import make_engine


@pytest.fixture(scope="session")
def test_db_url() -> Iterator[str]:
    """Create `<your db>_test`, hand its address to the tests, then delete it."""
    settings = get_settings()
    test_db = f"{settings.postgres_db}_test"
    admin_url = settings.database_url  # connect to the real db only to create/drop the test db

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{test_db}"')

    yield settings.sqlalchemy_url.rsplit("/", 1)[0] + f"/{test_db}"

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')


@pytest.fixture(scope="session")
def alembic_config(test_db_url: str) -> Config:
    config = Config("alembic.ini")
    config.attributes["url"] = test_db_url  # point Alembic at the test database
    return config


@pytest.fixture(scope="session")
def engine(alembic_config: Config, test_db_url: str) -> Iterator[Engine]:
    """Build the tables with our real migrations, exactly like production."""
    command.upgrade(alembic_config, "head")
    engine = make_engine(test_db_url)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Each test gets a session whose changes are thrown away afterwards."""
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()
