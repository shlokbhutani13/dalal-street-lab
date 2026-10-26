"""Scoring news moods: the bookkeeping with a fake scorer, the PyTorch code with a tiny model."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizer

from dalal_street_lab.models import Article, ArticleSymbol, Instrument
from dalal_street_lab.sentiment import FinBert, mood_report, score_articles, text_for


class FakeScorer:
    """Stands in for FinBERT: 'raises' sounds good, 'cuts' sounds bad, anything else is meh."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[dict[str, float]]:
        self.calls.append(texts)
        moods = []
        for text in texts:
            if "raises" in text:
                moods.append({"positive": 0.9, "negative": 0.05, "neutral": 0.05})
            elif "cuts" in text:
                moods.append({"positive": 0.1, "negative": 0.8, "neutral": 0.1})
            else:
                moods.append({"positive": 0.1, "negative": 0.1, "neutral": 0.8})
        return moods


def add_article(session: Session, news_id: int, headline: str, symbol_id: int) -> Article:
    now = datetime.now(UTC) - timedelta(hours=news_id)
    article = Article(
        id=news_id,
        headline=headline,
        summary="",
        source="benzinga",
        published_at=now,
        updated_at=now,
    )
    session.add(article)
    session.flush()
    session.add(ArticleSymbol(news_id=news_id, instrument_id=symbol_id))
    session.flush()
    return article


@pytest.fixture
def infy(session: Session) -> int:
    instrument = Instrument(
        instrument_key="uuid-INFY",
        symbol="INFY",
        name="Infosys",
        exchange="NYSE",
        segment="us_equity",
        instrument_type="ADR",
        tick_size=Decimal("0.01"),
    )
    session.add(instrument)
    session.flush()
    return instrument.id


def test_every_article_is_scored_once_in_batches(session: Session, infy: int) -> None:
    for news_id, headline in enumerate(["Infosys raises guidance", "Infosys cuts jobs", "Q3"], 1):
        add_article(session, news_id, headline, infy)
    scorer = FakeScorer()

    assert score_articles(session, scorer, "fake-v1", batch_size=2) == 3
    assert [len(call) for call in scorer.calls] == [2, 1]  # batches of 2, then the last one

    good, bad, meh = (session.get(Article, i) for i in (1, 2, 3))
    assert good is not None and bad is not None and meh is not None
    assert good.sentiment == Decimal("0.85")  # 0.9 positive - 0.05 negative
    assert bad.sentiment == Decimal("-0.7")
    assert meh.sentiment == Decimal("0")
    assert good.sentiment_model == "fake-v1"  # we always know which AI gave the score

    # Running again does nothing: everything is already scored by this model.
    assert score_articles(session, scorer, "fake-v1") == 0


def test_a_new_model_rescores_everything(session: Session, infy: int) -> None:
    add_article(session, 1, "Infosys raises guidance", infy)
    score_articles(session, FakeScorer(), "fake-v1")
    assert score_articles(session, FakeScorer(), "fake-v2") == 1


def test_summary_is_read_when_there_is_one(session: Session, infy: int) -> None:
    article = add_article(session, 1, "Infosys beats", infy)
    assert text_for(article) == "Infosys beats"
    article.summary = "Revenue up 8%"
    assert text_for(article) == "Infosys beats. Revenue up 8%"


def test_mood_report_averages_per_symbol(session: Session, infy: int) -> None:
    add_article(session, 1, "Infosys raises guidance", infy)
    add_article(session, 2, "Infosys cuts jobs", infy)
    score_articles(session, FakeScorer(), "fake-v1")
    assert mood_report(session) == [("INFY", 2, Decimal("0.075"))]  # (0.85 + -0.7) / 2


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny, untrained BERT with FinBERT's labels, built on the spot (no download).

    Its answers are random, but it runs the exact same PyTorch code as the real FinBERT.
    """
    folder = tmp_path_factory.mktemp("tiny-finbert")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "infosys", "raises", "guidance"]
    (folder / "vocab.txt").write_text("\n".join(words))
    BertTokenizer(vocab_file=str(folder / "vocab.txt")).save_pretrained(folder)
    config = BertConfig(
        vocab_size=len(words),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        id2label={0: "positive", 1: "negative", 2: "neutral"},
        label2id={"positive": 0, "negative": 1, "neutral": 2},
    )
    torch.manual_seed(0)
    # from_config has no type hints inside transformers, so mypy is told to trust this one call.
    model = AutoModelForSequenceClassification.from_config(config)  # type: ignore[no-untyped-call]
    model.save_pretrained(folder)
    return folder


def test_finbert_gives_three_probabilities_that_add_up_to_one(tiny_model: Path) -> None:
    finbert = FinBert(str(tiny_model), device=torch.device("cpu"))
    moods = finbert(["Infosys raises guidance", "a much longer headline " * 100])

    assert len(moods) == 2  # one answer per text, even when the second needs cutting short
    for mood in moods:
        assert set(mood) == {"positive", "negative", "neutral"}
        assert sum(mood.values()) == pytest.approx(1.0)


def test_scored_values_fit_the_table(session: Session, infy: int, tiny_model: Path) -> None:
    add_article(session, 1, "Infosys raises guidance", infy)
    score_articles(session, FinBert(str(tiny_model), device=torch.device("cpu")), "tiny")
    saved = session.scalars(select(Article)).one()
    assert saved.sentiment is not None
    assert Decimal(-1) <= saved.sentiment <= Decimal(1)
