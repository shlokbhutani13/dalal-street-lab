"""Score the mood of every news article with FinBERT, a free AI model from Hugging Face.

Run it with:  uv run dalal-sentiment

FinBERT is BERT (Google's language model) trained further on financial text. It reads a
headline and says how likely it is to be positive, negative or neutral for investors.
The first run downloads the model (about 440 MB) into ~/.cache/huggingface; later runs
reuse it.
"""

import argparse
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
import torch
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Article, ArticleSymbol, Instrument

log = structlog.get_logger()

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32  # articles per trip through the model: bigger is faster, up to memory limits
MAX_TOKENS = 128  # headlines are short; cutting long summaries here keeps it fast

# Anything that turns texts into {"positive": p, "negative": p, "neutral": p}.
# The real one is FinBert below; tests use a fake one, like the Recorder in Step 6.
Scorer = Callable[[list[str]], list[dict[str, float]]]


def pick_device() -> torch.device:
    """Use the fastest chip available: an NVIDIA GPU, an Apple GPU (mps), or the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBert:
    def __init__(self, model_name: str = MODEL_NAME, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        # The tokenizer cuts text into the word pieces the model knows, and turns them
        # into numbers. The model is the neural network itself.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # "exam mode": turns off the randomness used only while training
        # Read the label order from the model itself instead of guessing it.
        id2label = self.model.config.id2label  # FinBERT's: {0: positive, 1: negative, 2: neutral}
        self.labels = [str(id2label[i]).lower() for i in range(len(id2label))]

    def __call__(self, texts: list[str]) -> list[dict[str, float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():  # we're only predicting, so skip the training bookkeeping
            logits = self.model(**inputs).logits  # one raw score per label, per text
        # softmax turns raw scores into probabilities that add up to 1.
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        return [dict(zip(self.labels, row, strict=True)) for row in probabilities]


def text_for(article: Article) -> str:
    """What FinBERT reads: the headline, plus the summary when there is one."""
    if article.summary:
        return f"{article.headline}. {article.summary}"
    return article.headline


def as_score(value: float) -> Decimal:
    return Decimal(str(round(value, 4)))


def score_articles(
    session: Session, scorer: Scorer, model_name: str, batch_size: int = BATCH_SIZE
) -> int:
    """Score every article this model hasn't scored yet, oldest first, one batch at a time.

    Each batch is saved before the next starts, so stopping halfway loses nothing: the
    next run carries on. Articles scored by a different (older) model are scored again.
    """
    scored = 0
    while True:
        batch = session.scalars(
            select(Article)
            .where(Article.sentiment_model.is_distinct_from(model_name))
            .order_by(Article.published_at, Article.id)
            .limit(batch_size)
        ).all()
        if not batch:
            break
        for article, mood in zip(batch, scorer([text_for(a) for a in batch]), strict=True):
            article.positive = as_score(mood["positive"])
            article.negative = as_score(mood["negative"])
            article.neutral = as_score(mood["neutral"])
            article.sentiment = as_score(mood["positive"] - mood["negative"])
            article.sentiment_model = model_name
        session.commit()
        scored += len(batch)
        log.info("scored", articles=scored)
    return scored


def mood_report(session: Session, days: int = 30) -> list[tuple[str, int, Decimal]]:
    """Per symbol: how many scored articles in the last `days`, and their average mood."""
    since = datetime.now(UTC) - timedelta(days=days)
    rows = session.execute(
        select(Instrument.symbol, func.count(Article.id), func.avg(Article.sentiment))
        .join(ArticleSymbol, ArticleSymbol.instrument_id == Instrument.id)
        .join(Article, Article.id == ArticleSymbol.news_id)
        .where(Article.published_at >= since, Article.sentiment.is_not(None))
        .group_by(Instrument.symbol)
        .order_by(func.avg(Article.sentiment).desc())
    ).all()
    return [(symbol, count, round(Decimal(avg), 3)) for symbol, count, avg in rows]


def latest_headlines(session: Session, limit: int = 8) -> list[Article]:
    return list(
        session.scalars(
            select(Article)
            .where(Article.sentiment.is_not(None))
            .order_by(Article.published_at.desc())
            .limit(limit)
        ).all()
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Score news moods with FinBERT.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    finbert = FinBert()
    log.info("model_loaded", model=MODEL_NAME, device=str(finbert.device))
    with new_session() as session:
        score_articles(session, finbert, MODEL_NAME, args.batch_size)

        print("\nAverage mood, last 30 days  (-1 = very negative, +1 = very positive)")
        for symbol, count, mood in mood_report(session):
            print(f"  {symbol:<5} {mood:+.3f}   from {count} articles")
        print("\nLatest headlines")
        for article in latest_headlines(session):
            day = article.published_at.astimezone(UTC).strftime("%Y-%m-%d")
            print(f"  {day}  {article.sentiment:+.2f}  {article.headline[:90]}")


if __name__ == "__main__":
    main()
