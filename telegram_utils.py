# -*- CryptoICT AI -*-

"""
Rule-based AI Regex + Keyword Router for Telegram commands.

লক্ষ্য:
- main.py তে আলাদা আলাদা কমান্ড এন্ট্রি যোগ করার দরকার না পড়ে।
- ইনকামিং টেক্সট এলে এখানে সব রুল-চেক হবে এবং উপযুক্ত হ্যান্ডলার কল হবে।
- অন্য ফাইলে থাকা হ্যান্ডলারও (যেমন fvg_coinlist_handler) এখানে রেফারেন্স ধরে কাজ করবে।
"""

import re
import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from telegram import Update as TGUpdate
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, Application, filters
from telegram.constants import ParseMode

# shared helpers
from config_and_utils import get_username, is_admin, logger
from fvg_coinlist import fvg_coinlist_handler, parse_tf_command, get_fvg_coins_async
from data_api import get_top_gainers
from supabase import (
    status_reply_text,
    process_watchlist_action_text,  # parses "BTC ETH Check" etc and executes
)

# =============== Simple helpers ===============
def _norm_text(text: Optional[str]) -> str:
    return text.strip() if text else ""

# =============== Handlers ===============
async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    if not update.message:
        return
    await update.message.reply_text(
        await status_reply_text(),
        parse_mode=ParseMode.MARKDOWN
    )

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
    Uses the FVG coin computation to rank by volatility (1h).
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

    await update.message.reply_text(
        f"⏳ Computing top {top_n} high-volatility coins (1H) using FVG filter...",
        parse_mode=ParseMode.MARKDOWN
    )
    try:
        items = await get_fvg_coins_async("1h")
        ranked = sorted(items, key=lambda x: float(x.get("volatility", 0.0)), reverse=True)
        lines = [f"🌪️ Top {top_n} High Volatility (1H) (from FVG candidates):", ""]
        for i, item in enumerate(ranked[:top_n], 1):
            sym = item.get("symbol", "")
            vol = float(item.get("volatility", 0.0))
            lines.append(f"{i:>2}. `{sym}` — Vol={vol:.4f}  | Status: `{item.get('status','')}`")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")

# =============== Rule-based Router ===============
# Each rule is a function that, given (text), either returns a handler coroutine or None.

async def _route_message(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Master router. Try rules in order—first match wins.
    Falls back to watchlist action parser; if that fails, returns invalid.
    """
    if not update.message or not update.message.text:
        return

    text = _norm_text(update.message.text)

    # 1) Exact commands first
    if re.match(r'(?i)^\s*status\s*$', text):
        await status(update, context)
        return
    if re.match(r'(?i)^\s*top\s+gainer\s+list\s*$', text):
        await top_gainer_handler(update, context)
        return
    if re.match(r'(?i)^\s*volatility\s+coin\s+list(?:\s+\d+)?\s*$', text):
        await volatility_list_handler(update, context)
        return

    # 2) FVG Coin List (delegated fully to fvg_coinlist.py)
    if parse_tf_command(text):
        await fvg_coinlist_handler(update, context)
        return

    # 3) Watchlist/Admin actions (Check/Remove/Haram/Add again/Halal)
    reply = await process_watchlist_action_text(update, text)
    if reply:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
        return

    # 4) No match
    await update.message.reply_text(
        "❌ Invalid command.\nTry: `status`, `Top Gainer List`, `Volatility Coin List`, or `4H FVG Coin List`.\n"
        "For watchlist ops: `BTC ETH Check` / `BNB Haram` / `BTC Add` / `BNB Halal`",
        parse_mode=ParseMode.MARKDOWN
    )

def setup_application(app: Application) -> None:
    """
    Register minimal handlers; the router does the rest.
    """
    app.add_handler(CommandHandler("start", start))
    # Any text message goes through the router:
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _route_message))
    # You can add more generic fallbacks if needed
