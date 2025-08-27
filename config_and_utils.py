# -*- CryptoICT AI -*-

import os
import re
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, List
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from flask import Flask
from telegram import Bot as TGBot
from supabase import create_client, Client
from binance.spot import Spot

# ===================== env & logging =====================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
CHAT_ID_RAW = os.getenv("CHAT_ID", "").strip()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

if not BOT_TOKEN or not SUPABASE_URL or not SUPABASE_KEY or not CHAT_ID_RAW:
    raise EnvironmentError("Missing env: BOT_TOKEN/SUPABASE_URL/SUPABASE_KEY/CHAT_ID")

try:
    CHAT_ID: Any = int(CHAT_ID_RAW)
except Exception:
    CHAT_ID = CHAT_ID_RAW

BINANCE_BASE = "https://api.binance.com"
BINANCE_BATCH_SIZE = int(os.getenv("BINANCE_BATCH_SIZE", "15"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
SLEEP_TIME = float(os.getenv("SLEEP_TIME", "6"))
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))
USER_TZ = ZoneInfo(os.getenv("USER_TZ", "Asia/Dhaka"))  # UTC+6
AUTO_ADD_NEW_COINS = os.getenv("AUTO_ADD_NEW_COINS", "false").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("crypto-scanner")

# External clients (shared across modules)
bot = TGBot(token=BOT_TOKEN)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
binance_connector_client = Spot(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# ===================== Flask keep alive =====================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Bot is alive!"

def _run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

def keep_alive():
    asyncio.get_event_loop().run_in_executor(None, _run_flask)

# ===================== Exceptions =====================
class RateLimitError(Exception):
    def __init__(self, status: int, retry_after: Optional[int] = None, msg: Optional[str] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(msg or f"Rate limit: {status}")

# ===================== Telegram helpers =====================
def get_username(update) -> str:
    user = update.effective_user
    return user.username or str(user.id) if user else "Unknown"

def is_admin(username: Optional[str]) -> bool:
    return bool(username) and username.lower() in ["redwanict", "growingbullspremium", "mdredwan2024"  ]

def parse_command(text: str) -> Tuple[List[str], str]:
    if not text:
        return [], ""
    clean = text.strip()
    tokens = re.split(r'[\s,;]+', clean)
    actions_map = {
        "check": "Check",
        "remove": "Remove",
        "haram": "Haram",
        "add": "Add again",
        "addagain": "Add again",
        "add_again": "Add again",
        "halal": "Halal",
    }
    found_action = ""
    coins = []
    for t in tokens:
        tl = t.lower()
        if tl in actions_map:
            found_action = actions_map[tl]
            continue
        m = re.match(r'^([A-Za-z0-9]{2,10})(?:[/\-]?USDT)?$', t.upper())
        if m:
            base = m.group(1)
            coin = base if base.endswith("USDT") else f"{base}USDT"
            coins.append(coin.upper())
    return coins, found_action

# ===================== Time helpers =====================
def now_telegram_str() -> str:
    """Return time string for Telegram messages (USER_TZ, e.g. UTC+6)."""
    return datetime.now(USER_TZ).strftime("%d-%m-%Y %H:%M %Z")

# Backwards compatibility alias (used previously as now_local_str)
def now_local_str() -> str:
    return now_telegram_str()

# ===================== Cooldown =====================
LAST_ALERT_AT: dict[str, datetime] = {}

def cooldown_ok(symbol: str) -> bool:
    now = datetime.now(timezone.utc)
    last = LAST_ALERT_AT.get(symbol)
    return True if last is None else (now - last).total_seconds() >= (ALERT_COOLDOWN_MIN * 60)

def mark_alert(symbol: str):
    LAST_ALERT_AT[symbol] = datetime.now(timezone.utc)
