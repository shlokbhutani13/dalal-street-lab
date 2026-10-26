"""Downloading news from Alpaca and saving it without duplicates."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from requests import PreparedRequest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import dalal_street_lab.alpaca as alpaca_module
from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import Settings
from dalal_street_lab.models import Article, ArticleSymbol, Instrument
from dalal_street_lab.news import download, fetch_news, resume_from, save_articles

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def article(
    news_id: int,
    symbols: list[str],
    headline: str = "Infosys raises revenue guidance",
    updated: str = "2026-10-22T10:00:00Z",
) -> dict[str, Any]:
    return {
        "id": news_id,
        "headline": headline,
        "author": "Benzinga Newsdesk",
        "created_at": "2026-10-22T10:00:00Z",
        "updated_at": updated,
        "summary": "",
        "url": f"https://www.benzinga.com/news/{news_id}",
        "images": [],
        "symbols": symbols,
        "source": "benzinga",
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AlpacaClient:
    monkeypatch.setattr(alpaca_module, "SECONDS_BETWEEN_REQUESTS", 0)
    return AlpacaClient(Settings(postgres_password="x", alpaca_api_key="k", alpaca_secret_key="s"))


@pytest.fixture
def symbol_ids(session: Session) -> dict[str, int]:
    ids = {}
    for symbol in ["INFY", "WIT"]:
        instrument = Instrument(
            instrument_key=f"uuid-{symbol}",
            symbol=symbol,
            name=symbol,
            exchange="NYSE",
            segment="us_equity",
            instrument_type="ADR",
            tick_size=Decimal("0.01"),
        )
        session.add(instrument)
        session.flush()
        ids[symbol] = instrument.id
    return ids


def count(session: Session, table: type[Article] | type[ArticleSymbol]) -> int | None:
    return session.scalar(select(func.count()).select_from(table))


@responses.activate
def test_fetch_follows_every_page_oldest_first(client: AlpacaClient) -> None:
    seen: list[dict[str, list[str]]] = []

    def fake_alpaca(request: PreparedRequest) -> tuple[int, dict[str, str], str]:
        params = parse_qs(urlparse(request.url or "").query)
        seen.append(params)
        body: dict[str, Any]
        if "page_token" not in params:
            body = {"news": [article(1, ["INFY"])], "next_page_token": "abc"}
        else:
            body = {"news": [article(2, ["WIT"])], "next_page_token": None}
        return 200, {}, json.dumps(body)

    responses.add_callback(responses.GET, NEWS_URL, callback=fake_alpaca)
    pages = list(fetch_news(client, ["INFY", "WIT"], datetime(2026, 10, 1, tzinfo=UTC)))

    assert [[a["id"] for a in page] for page in pages] == [[1], [2]]
    assert seen[0]["symbols"] == ["INFY,WIT"]  # one request covers every symbol
    assert seen[0]["sort"] == ["asc"]
    assert seen[0]["limit"] == ["50"]
    assert seen[1]["page_token"] == ["abc"]


def test_articles_are_linked_only_to_our_symbols(
    session: Session, symbol_ids: dict[str, int]
) -> None:
    saved = save_articles(
        session,
        [
            article(1, ["INFY", "WIT", "ACN"]),  # about two of ours (and Accenture)
            article(2, ["AAPL"]),  # not about anything we follow: skipped
        ],
        symbol_ids,
    )
    assert saved == 1
    links = session.scalars(select(ArticleSymbol.instrument_id).order_by("instrument_id")).all()
    assert links == sorted(symbol_ids.values())


def test_saving_twice_does_not_duplicate(session: Session, symbol_ids: dict[str, int]) -> None:
    save_articles(session, [article(1, ["INFY"])], symbol_ids)
    save_articles(session, [article(1, ["INFY"])], symbol_ids)
    assert count(session, Article) == 1
    assert count(session, ArticleSymbol) == 1


def test_an_edited_article_is_updated_and_its_old_mood_cleared(
    session: Session, symbol_ids: dict[str, int]
) -> None:
    save_articles(session, [article(1, ["INFY"])], symbol_ids)
    saved = session.get(Article, 1)
    assert saved is not None
    saved.sentiment, saved.sentiment_model = Decimal("0.9"), "ProsusAI/finbert"
    session.commit()

    # Same article again, unchanged: the mood is kept.
    save_articles(session, [article(1, ["INFY"])], symbol_ids)
    session.refresh(saved)
    assert saved.sentiment == Decimal("0.9")

    # Alpaca edited the headline later: new text, and the mood must be worked out again.
    edited = article(1, ["INFY"], headline="Infosys CUTS guidance", updated="2026-10-22T12:00:00Z")
    save_articles(session, [edited], symbol_ids)
    session.refresh(saved)
    assert saved.headline == "Infosys CUTS guidance"
    assert saved.sentiment is None
    assert saved.sentiment_model is None


def test_resume_from_the_newest_article(session: Session, symbol_ids: dict[str, int]) -> None:
    default = datetime(2020, 7, 27, tzinfo=UTC)
    assert resume_from(session, default) == default
    save_articles(session, [article(1, ["INFY"])], symbol_ids)
    assert resume_from(session, default) == datetime(2026, 10, 22, 10, tzinfo=UTC)


@responses.activate
def test_download_saves_news_for_active_instruments(
    session: Session, client: AlpacaClient, symbol_ids: dict[str, int]
) -> None:
    responses.get(
        NEWS_URL,
        json={"news": [article(1, ["INFY"]), article(2, ["WIT"])], "next_page_token": None},
    )
    assert download(client, session, days=7) == 2
    assert count(session, Article) == 2
