# -*- CryptoICT AI -*-

"""
Supabase-facing helpers:
- watchlist management commands (Check/Remove/Haram/Add again/Halal)
- status text builder
- (admin/members access hooks live here by wrapping config.is_admin if later expanded)
"""

from typing import Any, Dict, List, Optional

from telegram import Update as TGUpdate
from telegram.ext import ContextTypes

from config_and_utils import logger, get_username, is_admin, parse_command
from data_api import (
    get_column, add_coin, add_coin_with_date, remove_coin_from_table, get_removed_map
)

# =============== Status text ===============
async def status_reply_text() -> str:
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
    except Exception as e:
        return f"❌ Supabase error: {e}"

    wl_total = len(watchlist)
    hr_total = len(haram)

    parts: List[str] = []
    parts.append(f"📊 Watchlist ({wl_total}):")
    parts.append(" ".join(f"`{w}`" for w in watchlist) if watchlist else "—")
    parts.append("")
    parts.append(f"⚠️ Haram ({hr_total}):")
    parts.append(" ".join(f"`{h}`" for h in haram) if haram else "—")
    return "\n".join(parts)

# =============== Watchlist actions core ===============
VALID_ACTIONS = {"Check", "Remove", "Haram", "Add again", "Halal"}

async def apply_watchlist_actions(coins: List[str], action: str) -> str:
    try:
        watchlist = await get_column("watchlist")
        haram = await get_column("haram")
        removed_map = await get_removed_map()
    except Exception as e:
        return f"❌ Supabase error: {e}"

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

    return "\n".join(reply_parts) or "✅ No changes made."

# =============== Public entry used by router ===============
async def process_watchlist_action_text(update: TGUpdate, text: str) -> Optional[str]:
    """
    Parse + apply watchlist action if format matches; enforce admin access.
    Returns reply text if matched, else None.
    """
    username = get_username(update)
    coins, action = parse_command(text)

    # If parse failed, not our command—let router try other rules.
    if not coins or action not in VALID_ACTIONS:
        return None

    # Access control
    if not is_admin(username):
        return "⛔ You are not authorized to use commands."

    # Execute
    reply = await apply_watchlist_actions(coins, action)
    return reply
