"""Settings must read the environment, hide secrets, and reject bad values."""

import pytest
from pydantic import ValidationError

from dalal_street_lab.config import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make each test independent of your real .env file and shell."""
    monkeypatch.chdir("/")  # no .env file here
    for name in ["POSTGRES_PASSWORD", "POSTGRES_PORT", "POSTGRES_HOST"]:
        monkeypatch.delenv(name, raising=False)


def test_builds_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    settings = Settings()
    assert settings.database_url == "postgresql://dalal:s3cret@localhost:5432/dalal"


def test_password_is_hidden_when_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    settings = Settings()
    assert "s3cret" not in repr(settings)
    assert "s3cret" not in str(settings.postgres_password)


def test_password_is_required() -> None:
    with pytest.raises(ValidationError):
        Settings()


def test_bad_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("POSTGRES_PORT", "not-a-number")
    with pytest.raises(ValidationError):
        Settings()
