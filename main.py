# # -*- CryptoICT AI -*-

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any

import aiohttp
from telegram.constants import ParseMode
from telegram import Update as TGUpdate
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from config_and_utils import (
    logger, keep_alive, bot, USER_TZ, CHAT_ID, SLEEP_TIME,
    MAX_CONCURRENCY, REQUEST_TIMEOUT, AUTO_ADD_NEW_COINS,
    get_username, is_admin, parse_command, now_telegram_str, mark_alert
)
import data_api
from data_api import (
    get_top_gainers, get_ticker_24h, get_exchange_info,
    get_column, add_coin, add_coin_with_date, remove_coin_from_table, get_removed_map,
    fetch_signals_since, log_to_supabase, parse_ohlcv  # keep for parity with old
)
from analysis import detect_signals, volatility_coin_list
from fvg_coinlist import fvg_coinlist_handler  # New feature handler

log = logging.getLogger("main")

# When True, normal scanning pauses while FVG scan runs
FVG_SCAN_RUNNING: bool = False

# ===================== Helpers =====================
def chunked(lst: List[Any], n: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ===================== Alerts & send =====================
async def format_status(symbol: str) -> str:
    """Return human label for coin's status."""
    watchlist = await get_column("watchlist")
    haram = await get_column("haram")
    removed_map = await get_removed_map()
    s = symbol.upper()
    if s in haram:
        return "HARAM ⚠️"
    if s in removed_map:
        return f"REMOVED {removed_map[s]}"
    if s in watchlist:
        return "WATCHLIST ✅"
    return "NEW 🚀"

async def send_alert(symbol: str, reasons_by_tf: Dict[str, List[str]], final_opinion: str, profiles: List[str]):
    """Send a formatted alert to Telegram and log to Supabase."""
    timestamp = now_telegram_str()  # Telegram messages use USER_TZ (UTC+6)
    status_text = await format_status(symbol)
    pct24 = await get_ticker_24h(symbol)
    pct_str = f"{pct24:+.2f}%" if pct24 is not None else "N/A"

    def tf_line(tf: str) -> str:
        r = reasons_by_tf.get(tf, [])
        return f"{tf}: " + (" + ".join(r) if r else "No Signal")

    prof_line = ("Profiles: " + ", ".join(profiles)) if profiles else "Profiles: —"

    message = (
        "🚨 *Bullish Signal Detected!*\n"
        f"Coin: `{symbol}`\n"
        f"Status: {status_text}\n"
        f"Time: {timestamp}\n"
        f"24h Change: {pct_str}\n"
        "---------------------------------\n"
        f"{tf_line('15m')}\n"
        f"{tf_line('1h')}\n"
        f"{tf_line('4h')}\n"
        f"{tf_line('1d')}\n"
        "---------------------------------\n"
        f"{prof_line}\n"
        f"{final_opinion}"
    )

    # Flatten reasons for logging
    reasons_flat: List[str] = []
    for tf in ["15m", "1h", "4h", "1d"]:
        for r in reasons_by_tf.get(tf, []):
            reasons_flat.append(f"{tf}: {r}")
    reasons_text = " | ".join(reasons_flat) if reasons_flat else "No Signal"

    try:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
        mark_alert(symbol)
        await log_to_supabase(symbol, reasons_text, final_opinion)
        log.info("Telegram alert sent for %s", symbol)
    except Exception:
        log.exception("Telegram error sending alert for %s", symbol)

# ===================== Scanning & batching =====================
async def scan_one_with_backoff(coin: str, btc_cache: Dict[str, List[float]]):
    """Detect signals for one coin with rate-limit backoff."""
    from config_and_utils import RateLimitError
    max_attempts = 5
    delay = 1
    for attempt in range(max_attempts):
        try:
            reasons_by_tf, final_opinion, _, profiles = await detect_signals(coin, btc_1h_cache=btc_cache)
            if reasons_by_tf:
                await send_alert(coin, reasons_by_tf, final_opinion or "⚪ No edge", profiles)
            await asyncio.sleep(SLEEP_TIME)
            break
        except RateLimitError as e:
            ra = e.retry_after or delay
            log.warning("Rate limit for %s. Retry after %s s (attempt %s)", coin, ra, attempt + 1)
            await asyncio.sleep(ra)
            delay = min(delay * 2, 60)
        except Exception:
            log.exception("Error with %s", coin)
            break

async def auto_add_new_listings(symbols: List[str]):
    """Auto-add new USDT spot listings to watchlist (if enabled)."""
    if not AUTO_ADD_NEW_COINS:
        return
    watchlist = await get_column("watchlist")
    haram = await get_column("haram")
    removed = await get_removed_map()
    for s in symbols:
        if s in haram or s in removed or s in watchlist:
            continue
        await add_coin_with_date("watchlist", s)

async def _get_tradable_usdt_symbols() -> List[str]:
    """Fetch and filter Binance spot USDT tradable symbols (excludes leveraged tokens)."""
    exinfo = await get_exchange_info()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
    ]
    # Exclude leveraged tokens
    symbols = [c for c in symbols if not any(x in c for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])]
    return symbols

async def batch_scan_all_with_rate_limit():
    """Scan all symbols in rate-limited batches. Pauses if FVG scan running."""
    if FVG_SCAN_RUNNING:
        log.info("Skipping normal scanning: FVG scan in progress")
        await asyncio.sleep(30)
        return

    symbols = await _get_tradable_usdt_symbols()
    await auto_add_new_listings(symbols)

    from config_and_utils import BINANCE_BATCH_SIZE
    log.info("Total %s USDT spot symbols. Scanning in batches of %s ...", len(symbols), BINANCE_BATCH_SIZE)

    btc_cache: Dict[str, List[float]] = {}

    for batch in chunked(symbols, BINANCE_BATCH_SIZE):
        log.info("Scanning batch of %s coins...", len(batch))

        async def limited_scan(coin):
            if data_api.binance_sem is None:
                log.error("binance_sem not initialized yet. Skipping %s", coin)
                return
            sem = data_api.binance_sem
            async with sem:
                await scan_one_with_backoff(coin, btc_cache)

        tasks = [limited_scan(coin) for coin in batch]
        await asyncio.gather(*tasks)
        log.info("Sleeping 60 seconds for next batch...")
        await asyncio.sleep(60)

async def scanner_loop():
    """Continuous scanning loop with error handling/backoff."""
    while True:
        try:
            await batch_scan_all_with_rate_limit()
        except Exception:
            log.exception("Scanner loop error")
            log.info("Sleeping 10 minutes...")
            await asyncio.sleep(600)

# ===================== Aggregator (5 times/day) =====================
LAST_REPORT_SENT: Dict[str, datetime] = {}
AGG_TIMES_LOCAL: List[Tuple[int, int]] = [(6, 0), (10, 0), (14, 0), (18, 0), (22, 0)]

def _is_agg_time(dt: datetime) -> Optional[str]:
    """Return timestamp key if given time matches one of our aggregation times."""
    for hh, mm in AGG_TIMES_LOCAL:
        if dt.hour == hh and dt.minute == mm:
            return dt.strftime("%Y-%m-%d %H:%M")
    return None

async def _build_aggregated_message(window_end_local: datetime) -> Optional[str]:
    """Build aggregated signals message text or None if no rows."""
    since_utc = (window_end_local.astimezone(timezone.utc) - timedelta(hours=4))
    since_iso = since_utc.isoformat()
    rows = await fetch_signals_since(since_iso)
    if not rows:
        return None

    watchlist = await get_column("watchlist")
    haram = await get_column("haram")
    removed_map = await get_removed_map()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        coin = r.get("coin", "").upper()
        grouped.setdefault(coin, []).append(r)

    lines: List[str] = [f"📋 Aggregated Signals (`{window_end_local.strftime('%Y-%m-%d %H:%M %Z')}`)", ""]
    for coin, items in grouped.items():
        if coin in haram:
            st = "HARAM ⚠️"
        elif coin in removed_map:
            st = f"REMOVED@{removed_map[coin]}"
        elif coin in watchlist:
            st = "WATCHLIST ✅"
        else:
            st = "NEW 🚀"

        # Unique reasons
        reasons_set: List[str] = []
        for it in items:
            rtxt = it.get("reasons", "")
            if rtxt and rtxt not in reasons_set:
                reasons_set.append(rtxt)

        # Latest opinion if any
        opinions = [it.get("opinion") for it in items if it.get("opinion")]
        opinion_text = opinions[-1] if opinions else None

        pct24 = await get_ticker_24h(coin)
        pct_str = f"{pct24:+.2f}%" if pct24 is not None else "N/A"

        lines.append(f"- `{coin}`: {st}")
        lines.append("   Reasons: " + (" | ".join(reasons_set) if reasons_set else "No Signal"))
        if opinion_text:
            lines.append(f"   Final Opinion: {opinion_text}")
        lines.append(f"   24h Change: {pct_str}")

    return "\n".join(lines)

async def aggregator_loop():
    """Send aggregated report at fixed local times; ensure not duplicated too often."""
    global LAST_REPORT_SENT
    while True:
        try:
            now_local = datetime.now(USER_TZ)
            agg_time_key = _is_agg_time(now_local)
            if agg_time_key:
                # Avoid duplicate sends within 1 hour for the same key
                if agg_time_key in LAST_REPORT_SENT and (datetime.now(USER_TZ) - LAST_REPORT_SENT[agg_time_key]).total_seconds() < 3600:
                    await asyncio.sleep(20)
                    continue

                window_end_local = now_local.replace(second=0, microsecond=0)
                msg = await _build_aggregated_message(window_end_local)
                if msg:
                    try:
                        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                        log.info("Aggregator message sent for window %s", window_end_local.isoformat())
                    except Exception:
                        log.exception("Telegram send error in aggregator")

                LAST_REPORT_SENT[agg_time_key] = now_local
                await asyncio.sleep(70)
            else:
                await asyncio.sleep(20)
        except Exception:
            log.exception("Aggregator loop error")
            await asyncio.sleep(30)

# ===================== Top Gainer + Volatility List Handlers =====================
async def top_gainer_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    await update.message.reply_text("⏳ Fetching top gainers, please wait...")
    try:
        gainers = await get_top_gainers(10, filtering_on=False)
        lines = ["🔥 Top 10 Binance USDT Spot Gainers (24h) [Filter OFF]:\n"]
        for i, (coin, pct) in enumerate(gainers, 1):
            lines.append(f"{i}. `{coin}` : {pct:+.2f}%")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to fetch: {e}")

async def volatility_list_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return

    text = update.message.text.strip()
    m = re.search(r'volatility\s+coin\s+list\s*(\d+)?', text, flags=re.IGNORECASE)
    top_n = 20
    if m and m.group(1):
        try:
            top_n = max(5, min(100, int(m.group(1))))
        except Exception:
            top_n = 20

    await update.message.reply_text(f"⏳ Computing top {top_n} high-volatility coins (1H)...")
    try:
        # Pass needed data_api functions to avoid circular deps
        items = await volatility_coin_list(data_api.get_exchange_info, data_api.get_klines, top_n)
        lines = [f"🌪️ Top {top_n} High Volatility (1H):", ""]
        for i, (sym, vol) in enumerate(items, 1):
            lines.append(f"{i:>2}. `{sym}` — Vol={vol:.4f}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")

# ===================== Telegram commands =====================
async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Old version's /start restored to fix 'start is not defined'."""
    if not update.message:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text(
            f"⛔ Sorry @{username},\nyou do not have permission to use the Crypto Watchlist Tracking Bot. -RedwanICT"
        )
        return
    await update.message.reply_text(
        "👋 Hello! Commands:\n"
        "- `BTCUSDT Check`, `ETH Remove`, `BNB Haram`, `BTC Add`, `BNB Halal`\n"
        "- `Top Gainer List`\n"
        "- `Volatility Coin List` or `Volatility Coin List 30`\n"
        "- `1H FVG Coin List`, `4H FVG Coin List`, `1D FVG Coin List`",
        parse_mode=ParseMode.MARKDOWN
    )

async def status(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show watchlist/haram summary. Keep try/except robustness like old version."""
    if not update.message:
        return
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return

    wl_total = len(watchlist)
    hr_total = len(haram)

    parts: List[str] = []
    parts.append(f"📊 Watchlist ({wl_total}):")
    parts.append(" ".join(f"`{w}`" for w in watchlist) if watchlist else "—")
    parts.append("")
    parts.append(f"⚠️ Haram ({hr_total}):")
    parts.append(" ".join(f"`{h}`" for h in haram) if haram else "—")

    reply = "\n".join(parts)
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

async def handle_commands(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main text handler for admin actions and utility lists."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # FVG Coin List command (pauses normal scan while running)
    # Accept custom timeframe prefix (e.g., "15m FVG Coin list", "4H FVG Coin list", "2h FVG Coin list")
    if re.search(r'^\s*[0-9]+[mMhHdDwW]\s*FVG\s*COIN\s*LIST\s*$', text, flags=re.IGNORECASE):
        global FVG_SCAN_RUNNING
        FVG_SCAN_RUNNING = True
        try:
            await fvg_coinlist_handler(update, context)
        finally:
            FVG_SCAN_RUNNING = False
        return

    # Special list commands
    if re.search(r'^\s*top\s+gainer\s+list\s*$', text, flags=re.IGNORECASE):
        await top_gainer_handler(update, context)
        return
    if re.search(r'^\s*volatility\s+coin\s+list(?:\s+\d+)?\s*$', text, flags=re.IGNORECASE):
        await volatility_list_handler(update, context)
        return

    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return

    logger.info("Command from %s: %s", username, text)
    coins, action = parse_command(text)
    valid_actions = {"Check", "Remove", "Haram", "Add again", "Halal"}
    if not coins or action not in valid_actions:
        await update.message.reply_text(
            "❌ Invalid format. Use:\n"
            "Check, Remove, Haram, Add Again, Halal\n"
            "Example: BTC ETH Check\n"
            "Or send: Top Gainer List / Volatility Coin List",
        )
        return

    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
        removed_map = await get_removed_map()
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return

    already: List[str] = []
    added: List[str] = []
    removed: List[str] = []
    marked_haram: List[str] = []
    unharamed: List[str] = []

    for coin in coins:
        if action == "Check":
            if coin in haram:
                marked_haram.append(coin)
            elif coin in removed_map:
                removed.append(f"{coin} - {removed_map[coin]}")
            elif coin in watchlist:
                already.append(coin)
            else:
                await add_coin_with_date("watchlist", coin)
                added.append(coin)

        elif action == "Remove":
            if await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(coin)

        elif action == "Haram":
            await add_coin("haram", coin)
            marked_haram.append(coin)
            if coin in watchlist and await remove_coin_from_table("watchlist", coin):
                await add_coin_with_date("removed", coin)
                removed.append(f"{coin} (removed from watchlist due to haram)")

        elif action == "Add again":
            if coin in removed_map:
                await add_coin_with_date("watchlist", coin)
                await remove_coin_from_table("removed", coin)
                added.append(coin)

        elif action == "Halal":
            if await remove_coin_from_table("haram", coin):
                unharamed.append(coin)

    reply_parts: List[str] = []
    if already:
        reply_parts.append("🟢 Already in Watchlist:\n" + " ".join(f"`{x}`" for x in already))
    if added:
        reply_parts.append("✅ New Added:\n" + " ".join(f"`{x}`" for x in added))
    if marked_haram:
        reply_parts.append("⚠️ Marked as Haram:\n" + " ".join(f"`{x}`" for x in marked_haram))
    if removed:
        reply_parts.append("🗑️ Removed:\n" + " ".join(f"`{x}`" for x in removed))
    if unharamed:
        reply_parts.append("✅ Removed from Haram:\n" + " ".join(f"`{x}`" for x in unharamed))

    reply = "\n".join(reply_parts) or "✅ No changes made."
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

# ===================== App bootstrap =====================
async def post_init(application):
    """Create shared aiohttp session & semaphore, then start background tasks."""
    # IMPORTANT: keep ssl=False and headers as you had
    data_api.aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, ssl=False),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        headers={"User-Agent": "CryptoWatchlistBot/2.0 (+contact)"}
    )
    # Semaphore used in scanning — must be initialized before use
    data_api.binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Background tasks
    application.create_task(scanner_loop())
    application.create_task(aggregator_loop())

    logger.info("post_init completed: aiohttp session created and background tasks started.")

def main() -> None:
    """Entrypoint. Builds application and registers handlers."""
    keep_alive()
    from config_and_utils import BOT_TOKEN
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()

    # Commands / handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"(?i)^status$"), status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_commands))

    logger.info("Bot starting (press Ctrl+C to stop)...")
    app.run_polling()

if __name__ == "__main__":
    main()
