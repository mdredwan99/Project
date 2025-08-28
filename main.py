# -*- CryptoICT AI -*-
"""
Minimal main entrypoint for the Crypto Watchlist bot (Aggregator removed).
Initializes shared aiohttp session & binance semaphore and registers
telegram handlers from telegram_utils.
"""

import asyncio
import logging
from typing import Any

import aiohttp
from telegram.ext import ApplicationBuilder

# shared config, clients & settings
from config_and_utils import (
    logger,
    keep_alive,
    USER_TZ,
    MAX_CONCURRENCY,
    REQUEST_TIMEOUT,
)

# data layer
import data_api

# router (rule-based message router)
import telegram_utils as telegram_router

log = logging.getLogger("main")


async def post_init(application):
    """Create shared aiohttp session & semaphore. No background tasks started."""
    # initialize shared session used by data_api
    data_api.aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, ssl=False),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        headers={"User-Agent": "CryptoWatchlistBot/2.0 (+contact)"},
    )
    # initialize semaphore used in data_api for concurrency control
    data_api.binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    logger.info("post_init completed: aiohttp session created.")


def main() -> None:
    """Entrypoint. Builds application and registers handlers via telegram router."""
    keep_alive()
    from config_and_utils import BOT_TOKEN

    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()

    # Use the rule-based router to set up handlers (centralized in telegram_utils)
    telegram_router.setup_application(app)

    logger.info("Bot starting (press Ctrl+C to stop)...")
    app.run_polling()


if __name__ == "__main__":
    main()
