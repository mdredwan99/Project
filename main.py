# -*- CryptoICT AI -*-

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder

# keep_alive, shared clients & settings
from config_and_utils import (
    logger, keep_alive, bot, USER_TZ, CHAT_ID,
    MAX_CONCURRENCY, REQUEST_TIMEOUT
)

# data layer
import data_api
from data_api import (
    fetch_signals_since, get_column, get_removed_map, get_ticker_24h
)

# router & handlers (new rule-based system)
import telegram as telegram_router  # our new telegram.py


log = logging.getLogger("main")

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

# ===================== App bootstrap =====================
async def post_init(application):
    """Create shared aiohttp session & semaphore, then start background tasks."""
    data_api.aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, ssl=False),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        headers={"User-Agent": "CryptoWatchlistBot/2.0 (+contact)"}
    )
    data_api.binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Background tasks
    application.create_task(aggregator_loop())

    logger.info("post_init completed: aiohttp session created and background tasks started.")

def main() -> None:
    """Entrypoint. Builds application and registers handlers via telegram router."""
    keep_alive()
    from config_and_utils import BOT_TOKEN
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()

    # Let the router set up all handlers (rule-based regex+keyword)
    telegram_router.setup_application(app)

    logger.info("Bot starting (press Ctrl+C to stop)...")
    app.run_polling()

if __name__ == "__main__":
    main()                    try:
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
    """
    Use fvg_coinlist.get_fvg_coins_async('1h') to compute volatility-based list.
    This avoids depending on analysis.py.
    """
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

    await update.message.reply_text(f"⏳ Computing top {top_n} high-volatility coins (1H) using FVG filter...", parse_mode=ParseMode.MARKDOWN)
    try:
        items = await get_fvg_coins_async("1h")
        # items include 'volatility' key computed in fvg_coinlist
        ranked = sorted(items, key=lambda x: float(x.get("volatility", 0.0)), reverse=True)
        lines = [f"🌪️ Top {top_n} High Volatility (1H) (from FVG candidates):", ""]
        for i, item in enumerate(ranked[:top_n], 1):
            sym = item.get("symbol", "")
            vol = float(item.get("volatility", 0.0))
            lines.append(f"{i:>2}. `{sym}` — Vol={vol:.4f}  | Status: `{item.get('status','')}`")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")

# ===================== Telegram commands =====================
async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command"""
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
    """Show watchlist/haram summary."""
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

    # FVG Coin List command (delegated to fvg_coinlist handler)
    if re.search(r'^\s*[0-9]+[mMhHdDwW]\s*FVG\s*COIN\s*LIST\s*$', text, flags=re.IGNORECASE):
        await fvg_coinlist_handler(update, context)
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
    # Semaphore used by some data_api helpers (optional safety)
    data_api.binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Background tasks: aggregator only (scanner removed since analysis.py is gone)
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
