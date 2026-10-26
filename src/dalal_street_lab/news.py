"""Download news articles about our instruments from Alpaca into the news tables.

Run it with:  uv run dalal-news                 (everything since 2020-07-27, the first time)
              uv run dalal-news --days 7        (just the last week)

Running it again only downloads what's new: it resumes from the newest article we have.
"""

import argparse
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Article, ArticleSymbol, Instrument

log = structlog.get_logger()

# Our price history starts here (Step 5), so news before it can't be matched to prices.
EARLIEST = date(2020, 7, 27)
PAGE_SIZE = 50  # the most articles Alpaca returns per request

# When Alpaca edits an article, refresh these columns (and clear the old mood: see below).
UPDATABLE = ["headline", "summary", "author", "url", "updated_at"]
SENTIMENT_COLUMNS = ["sentiment_model", "positive", "negative", "neutral", "sentiment"]


def fetch_news(
    client: AlpacaClient, symbols: list[str], start: datetime
) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of articles, oldest first, linked by next_page_token like the bars."""
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "start": start.isoformat(timespec="seconds"),
        "end": datetime.now(UTC).isoformat(timespec="seconds"),
        "limit": PAGE_SIZE,
        # Oldest first: if the download stops halfway, what we saved has no holes,
        # and the next run resumes from the newest saved article.
        "sort": "asc",
        "include_content": "false",  # the full article text is big and FinBERT doesn't need it
    }
    while True:
        page = client.get(f"{client.data_url}/v1beta1/news", params=params)
        yield page.get("news", [])
        token = page.get("next_page_token")
        if not token:
            return
        params["page_token"] = token


def save_articles(
    session: Session, articles: list[dict[str, Any]], symbol_ids: dict[str, int]
) -> int:
    """Upsert articles and link each one to the instruments it mentions. Returns how many."""
    rows, links = [], []
    for article in articles:
        ours = [symbol_ids[s] for s in article.get("symbols", []) if s in symbol_ids]
        if not ours:
            continue  # only about symbols we don't follow
        rows.append(
            {
                "id": int(article["id"]),
                "headline": article["headline"].strip(),
                "summary": (article.get("summary") or "").strip(),
                "source": article.get("source") or "unknown",
                "author": (article.get("author") or "")[:256],
                "url": article.get("url") or None,
                "published_at": datetime.fromisoformat(article["created_at"]),
                "updated_at": datetime.fromisoformat(article["updated_at"]),
            }
        )
        links += [{"news_id": int(article["id"]), "instrument_id": i} for i in ours]
    if not rows:
        return 0

    statement = insert(Article).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[Article.id],
        set_={
            **{col: statement.excluded[col] for col in UPDATABLE},
            # The text changed, so the old mood may be wrong: clear it, to be scored again.
            **dict.fromkeys(SENTIMENT_COLUMNS),
        },
        # Only when Alpaca's copy is newer. Re-downloading the same article changes nothing.
        where=statement.excluded.updated_at > Article.updated_at,
    )
    session.execute(statement)
    session.execute(insert(ArticleSymbol).values(links).on_conflict_do_nothing())
    session.commit()
    return len(rows)


def resume_from(session: Session, default: datetime) -> datetime:
    """Start from the newest article we already have (it's re-fetched: harmless upsert)."""
    newest = session.scalar(select(func.max(Article.published_at)))
    return default if newest is None else max(default, newest)


def download(client: AlpacaClient, session: Session, days: int | None) -> int:
    instruments = session.execute(
        select(Instrument.symbol, Instrument.id).where(Instrument.is_active)
    ).all()
    if not instruments:
        raise SystemExit("No instruments in the database. Run: uv run dalal-instruments")
    symbol_ids = {symbol: instrument_id for symbol, instrument_id in instruments}

    earliest = datetime.combine(EARLIEST, time(), tzinfo=UTC)
    default = earliest if days is None else datetime.now(UTC) - timedelta(days=days)
    start = resume_from(session, max(default, earliest))
    log.info("news_download_started", since=start.date().isoformat(), symbols=len(symbol_ids))

    saved = 0
    for page in fetch_news(client, sorted(symbol_ids), start):
        saved += save_articles(session, page, symbol_ids)
        if page:
            log.debug("news_page", saved=saved, up_to=page[-1]["created_at"])
    log.info("news_downloaded", articles=saved)
    return saved


def per_symbol_counts(session: Session) -> list[tuple[str, int]]:
    rows = session.execute(
        select(Instrument.symbol, func.count(ArticleSymbol.news_id))
        .join(ArticleSymbol, ArticleSymbol.instrument_id == Instrument.id)
        .group_by(Instrument.symbol)
        .order_by(func.count(ArticleSymbol.news_id).desc())
    ).all()
    return [(symbol, count) for symbol, count in rows]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Download news about our instruments.")
    parser.add_argument("--days", type=int, help="how far back (default: since 2020-07-27)")
    args = parser.parse_args()

    client = AlpacaClient(get_settings())
    with new_session() as session:
        download(client, session, args.days)
        for symbol, count in per_symbol_counts(session):
            log.info("news_total", symbol=symbol, articles=count)


if __name__ == "__main__":
    main()
