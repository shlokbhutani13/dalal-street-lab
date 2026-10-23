"""Alembic's setup: where the database is, and what the tables should look like."""

from logging.config import fileConfig
from typing import Literal

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.schema import SchemaItem

from dalal_street_lab.config import get_settings
from dalal_street_lab.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The database address comes from our settings (.env), never from alembic.ini.
# Tests can pass a different address (a separate test database) through attributes.
url = config.attributes.get("url") or get_settings().sqlalchemy_url
config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))

# What the tables SHOULD look like: Alembic compares this against the real database.
target_metadata = Base.metadata


def include_object(
    obj: SchemaItem,
    name: str | None,
    type_: Literal[
        "schema", "table", "column", "index", "unique_constraint", "foreign_key_constraint"
    ],
    reflected: bool,
    compare_to: SchemaItem | None,
) -> bool:
    # TimescaleDB adds its own time index to every hypertable. It isn't in our models,
    # so without this, autogenerate would try to delete it in every new migration.
    return not (type_ == "index" and reflected and compare_to is None)


def run_migrations_offline() -> None:
    """Write the SQL to the screen instead of running it (alembic upgrade --sql)."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply the migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
