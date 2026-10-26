"""Watch live trades as they happen, by listening to the Redis channel the stream publishes to.

Run it with:  uv run dalal-watch        (in a second terminal, while dalal-stream runs)
"""

import asyncio
import json

import redis.asyncio as aioredis

from dalal_street_lab.config import get_settings
from dalal_street_lab.stream import LIVE_CHANNEL


async def watch() -> None:
    client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    async with client.pubsub() as pubsub:
        await pubsub.subscribe(LIVE_CHANNEL)
        print(f"Listening on '{LIVE_CHANNEL}'. Trades appear as they happen (Ctrl+C to stop).\n")
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue  # skip the "subscribed" confirmation
            trade = json.loads(message["data"])
            clock = trade["time"][11:23]  # just HH:MM:SS.mmm (UTC)
            print(f"{clock}  {trade['symbol']:<5} ${trade['price']:>9}  x{trade['size']}")


def main() -> None:
    try:
        asyncio.run(watch())
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
