"""Look up our watchlist on Alpaca and save it to the instruments table.

Run it with:  uv run dalal-instruments
"""

import tomllib
from decimal import Decimal
from pathlib import Path

import requests
import structlog
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from dalal_street_lab.alpaca import AlpacaClient
from dalal_street_lab.config import get_settings
from dalal_street_lab.db import new_session
from dalal_street_lab.log import configure_logging
from dalal_street_lab.models import Instrument

log = structlog.get_logger()

WATCHLIST_FILE = Path("watchlist.toml")
MAX_LIVE_SYMBOLS = 30  # Alpaca's free plan streams up to 30 symbols at once
US_TICK_SIZE = Decimal("0.01")  # US stocks priced at $1 or more move in 1-cent steps


# ---------- 1. What our watchlist file must contain ----------
class Watchlist(BaseModel):
    india_etfs: list[str]
    india_adrs: list[str]
    commodity_etfs: list[str]

    @classmethod
    def load(cls, path: Path = WATCHLIST_FILE) -> "Watchlist":
        raw = tomllib.loads(path.read_text())
        return cls(
            india_etfs=raw["india"]["etfs"],
            india_adrs=raw["india"]["adrs"],
            commodity_etfs=raw["commodities"]["etfs"],
        )

    def with_types(self) -> dict[str, str]:
        """Every symbol, labelled with what kind of instrument it is."""
        types = {s: "ETF" for s in self.india_etfs + self.commodity_etfs}
        types.update({s: "ADR" for s in self.india_adrs})
        if len(types) > MAX_LIVE_SYMBOLS:
            raise ValueError(f"Watchlist has {len(types)} symbols; the free limit is 30.")
        return types


# ---------- 2. What Alpaca tells us about one asset ----------
class AlpacaAsset(BaseModel):
    id: str  # a permanent ID: symbols can change (e.g. after a merger), this never does
    symbol: str
    name: str
    exchange: str  # NASDAQ, NYSE, ARCA, ...
    asset_class: str = Field(alias="class")  # "class" is a Python keyword, so we rename it
    status: str  # active / inactive
    tradable: bool


class UnusableSymbolError(Exception):
    pass


def fetch_assets(client: AlpacaClient, symbols: list[str]) -> list[AlpacaAsset]:
    assets, problems = [], []
    for symbol in symbols:
        try:
            data = client.get(f"{client.trading_url}/v2/assets/{symbol}")
        except requests.HTTPError as error:  # e.g. 404: no such symbol on Alpaca
            problems.append(f"{symbol} (HTTP {error.response.status_code})")
            continue
        asset = AlpacaAsset.model_validate(data)
        if asset.status != "active" or not asset.tradable:
            problems.append(f"{symbol} (not active/tradable)")
        else:
            assets.append(asset)
    if problems:
        # Fail loudly: a silently shorter watchlist is a bug nobody notices.
        raise UnusableSymbolError(f"Can't use: {', '.join(problems)}")
    return assets


# ---------- 3. Save: insert new ones, update changed ones, retire the rest ----------
def save(session: Session, assets: list[AlpacaAsset], types: dict[str, str]) -> None:
    for asset in assets:
        values = {
            "instrument_key": asset.id,
            "symbol": asset.symbol,
            "name": asset.name[:128],
            "exchange": asset.exchange,
            "segment": asset.asset_class,
            "instrument_type": types[asset.symbol],
            "lot_size": 1,
            "tick_size": US_TICK_SIZE,
            "expiry": None,
            "is_active": True,
        }
        # "Upsert": insert, or if this instrument_key already exists, update it instead.
        statement = insert(Instrument).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[Instrument.instrument_key],
            set_={k: v for k, v in values.items() if k != "instrument_key"},
        )
        session.execute(statement)

    # Anything no longer on the watchlist is marked inactive, never deleted:
    # old candles and trades still point to it.
    keys = [a.id for a in assets]
    session.execute(
        update(Instrument).where(Instrument.instrument_key.not_in(keys)).values(is_active=False)
    )
    session.commit()


def main() -> None:
    configure_logging()
    types = Watchlist.load().with_types()
    assets = fetch_assets(AlpacaClient(get_settings()), list(types))
    with new_session() as session:
        save(session, assets, types)

    print(f"\n{'SYMBOL':<7} {'TYPE':<5} {'EXCHANGE':<9} NAME")
    for a in assets:
        print(f"{a.symbol:<7} {types[a.symbol]:<5} {a.exchange:<9} {a.name}")
    log.info("watchlist_saved", count=len(assets))


if __name__ == "__main__":
    main()
