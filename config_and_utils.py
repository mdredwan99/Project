# -*- CryptoICT AI -*-

import os
import re
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, List, TYPE_CHECKING
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from flask import Flask
from telegram import Bot as TGBot

# NOTE:
# Some environments / type stubs may not expose create_client/Client or
# binance.spot.Spot members (pyright can complain). We import runtime
# callables defensively and import heavy/types only for type checking.
try:
    # create_client is needed at runtime; if import fails, provide a noop to avoid import errors
    from supabase import create_client  # type: ignore
except Exception:
    def create_client(*args, **kwargs):  # fallback noop (will produce runtime errors if used)
        return None  # type: ignore

if TYPE_CHECKING:
    # Import types only for static type checkers to reference
    from supabase import Client as SupabaseClient  # type: ignore
    from binance.spot import Spot as BinanceSpot  # type: ignore

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
USER_TZ = ZoneInfo(os.getenv("USER_TZ", "Asia/Dhaka"))  # default UTC+6
AUTO_ADD_NEW_COINS = os.getenv("AUTO_ADD_NEW_COINS", "false").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("crypto-scanner")

# External clients (shared across modules)
bot = TGBot(token=BOT_TOKEN)

# Create supabase client at runtime; annotate as Any to avoid static errors
supabase: Any = create_client(SUPABASE_URL, SUPABASE_KEY)

# Binance connector: import lazily to avoid type-stub issues; annotate as Any
try:
    from binance.spot import Spot  # type: ignore
    binance_connector_client: Any = Spot(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
except Exception:
    # If import fails (e.g., in a type-check-only environment), set to None
    binance_connector_client: Any = None

# ===================== Flask keep alive =====================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Bot is alive!"

def _run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

def keep_alive():
    """
    Run the Flask keep-alive server on a background thread/executor.
    Called from main before starting the Telegram app.
    """
    try:
        asyncio.get_event_loop().run_in_executor(None, _run_flask)
    except RuntimeError:
        # If no running loop (e.g., being executed in sync context), start the server in a new thread
        import threading
        t = threading.Thread(target=_run_flask, daemon=True)
        t.start()

# ===================== Exceptions =====================
class RateLimitError(Exception):
    def __init__(self, status: int, retry_after: Optional[int] = None, msg: Optional[str] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(msg or f"Rate limit: {status}")

# ===================== Telegram helpers =====================
def get_username(update) -> str:
    user = getattr(update, "effective_user", None)
    if not user:
        return "Unknown"
    return user.username or str(user.id)

def is_admin(username: Optional[str]) -> bool:
    if not username:
        return False
    return username.lower() in ["redwanict", "growingbullspremium", "mdredwan2024"]

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
        # Accept base symbols like BTC or full like BTCUSDT
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
