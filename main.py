# main.py
# -*- coding: utf-8 -*-
import os
import re
import asyncio
import threading
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask
from telegram import Update as TGUpdate
from telegram import Bot as TGBot
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from supabase import create_client, Client

# ── ENV SETUP ────────────────────────────────────────────────────────────

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
CHAT_ID = os.getenv("CHAT_ID", "")

if not BOT_TOKEN or not SUPABASE_URL or not SUPABASE_KEY or not CHAT_ID:
    raise EnvironmentError("Missing environment variables: BOT_TOKEN/SUPABASE_URL/SUPABASE_KEY/CHAT_ID")

bot = TGBot(token=BOT_TOKEN)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── FLASK KEEP-ALIVE ──────────────────────────────────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Bot is alive!"

def run_flask():
    flask_app.run(host="0.0.0.0", port=8080)

def keep_alive():
    threading.Thread(target=run_flask, daemon=True).start()

# ── SUPABASE UTILITIES ───────────────────────────────────────────────────────

def get_column(table: str) -> List[str]:
    try:
        response = supabase.table(table).select("coin").execute()
        return [r["coin"].upper() for r in (response.data or [])]
    except Exception as e:
        print(f"❌ Supabase get_column error: {e}")
        return []

def add_coin(table: str, coin: str) -> None:
    coin = coin.upper()
    if coin not in get_column(table):
        try:
            supabase.table(table).insert({"coin": coin}).execute()
        except Exception as e:
            print(f"❌ Supabase add_coin error: {e}")

def add_coin_with_date(table: str, coin: str) -> None:
    coin = coin.upper()
    if coin not in get_column(table):
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            supabase.table(table).insert({"coin": coin, "timestamp": timestamp}).execute()
        except Exception as e:
            print(f"❌ Supabase add_coin_with_date error: {e}")

def remove_coin_from_table(table: str, coin: str) -> bool:
    coin = coin.upper()
    try:
        response = supabase.table(table).select("id").eq("coin", coin).execute()
        if response.data:
            for row in response.data:
                supabase.table(table).delete().eq("id", row["id"]).execute()
            return True
    except Exception as e:
        print(f"❌ Supabase remove_coin_from_table error: {e}")
    return False

def get_removed_map() -> Dict[str, str]:
    try:
        response = supabase.table("removed").select("coin", "timestamp").execute()
        return {r["coin"].upper(): r["timestamp"] for r in (response.data or []) if r.get("timestamp")}
    except Exception as e:
        print(f"❌ Supabase get_removed_map error: {e}")
        return {}

# ── BINANCE REST UTILITIES ──────────────────────────────────────────────────

BINANCE_BASE = "https://api.binance.com"
BINANCE_BATCH_SIZE = 15
MAX_CONCURRENCY = 8
REQUEST_TIMEOUT = 12
SLEEP_TIME = 6

# Globals created post_init
aiohttp_session: Optional[aiohttp.ClientSession] = None
binance_sem: Optional[asyncio.Semaphore] = None

class RateLimitError(Exception):
    def __init__(self, status, retry_after=None, msg=None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(msg or f"Rate limit: {status}")

async def binance_request(path: str, params: Optional[dict] = None) -> Tuple[dict, Optional[str]]:
    global aiohttp_session
    if aiohttp_session is None:
        raise RuntimeError("HTTP session is not initialized")
    url = f"{BINANCE_BASE}{path}"
    if params is None:
        params = {}
    async with aiohttp_session.get(url, params=params) as resp:
        used_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT")
        if resp.status == 429:
            retry = resp.headers.get("Retry-After")
            body = await resp.text()
            raise RateLimitError(429, retry_after=int(retry) if retry and retry.isdigit() else None, msg=f"429 {body}")
        if resp.status != 200:
            body = await resp.text()
            raise Exception(f"HTTP {resp.status}: {body}")
        data = await resp.json(content_type=None)
        return data, used_weight

async def get_exchange_info():
    data, used = await binance_request("/api/v3/exchangeInfo")
    if used:
        print(f"ExchangeInfo weight: {used}")
    return data

async def get_klines(symbol: str, interval: str, limit: int = 150):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data, used = await binance_request("/api/v3/klines", params=params)
    return data

# ── NUMERIC HELPERS ─────────────────────────────────────────────────────────

def np_safe(arr: List[float]) -> np.ndarray:
    return np.array(arr, dtype=float) if arr else np.array([], dtype=float)

def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0

def rsi_series(closes: List[float], length: int = 14) -> np.ndarray:
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 50.0)
    delta = np.diff(closes_np)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    roll_up = np.convolve(gains, np.ones(length, dtype=float), 'valid') / length
    roll_down = np.convolve(losses, np.ones(length, dtype=float), 'valid') / length
    rs = np.divide(roll_up, roll_down, out=np.full_like(roll_up, np.nan), where=roll_down != 0)
    rsi = 100 - (100 / (1 + rs))
    pad = np.full(closes_np.size - rsi.size, 50.0)
    return np.concatenate([pad, rsi])

def atr_series(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> np.ndarray:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 0.0)
    prev_close = np.concatenate([[closes_np[0]], closes_np[:-1]])
    tr = np.maximum(highs_np - lows_np, np.maximum(np.abs(highs_np - prev_close), np.abs(lows_np - prev_close)))
    atr = np.convolve(tr, np.ones(length, dtype=float), 'valid') / length
    pad = np.full(tr.size - atr.size, atr[0] if atr.size else 0.0)
    return np.concatenate([pad, atr])

def ema_series(values: List[float], length: int) -> np.ndarray:
    values_np = np_safe(values)
    if values_np.size == 0:
        return values_np
    alpha = 2 / (length + 1)
    out = np.empty_like(values_np)
    out[0] = values_np[0]
    for i in range(1, values_np.size):
        out[i] = alpha * values_np[i] + (1 - alpha) * out[i - 1]
    return out

def cvd_proxy(closes: List[float], volumes: List[float]) -> np.ndarray:
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 2 or volumes_np.size != closes_np.size:
        return np.cumsum(np.zeros_like(closes_np))
    delta = np.diff(closes_np)
    sign = np.sign(delta)
    sign = np.concatenate([[0.0], sign])
    delta_vol = sign * volumes_np
    return np.cumsum(delta_vol)

def percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0

# ── ICT/SMC BUILDING BLOCKS ────────────────────────────────────────────────

def sell_side_liquidity_sweep_bullish(
    highs: List[float], lows: List[float], opens: List[float], closes: List[float], lookback: int = 20
) -> bool:
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if lows_np.size < lookback + 2:
        return False
    idx = -1
    prior_low = float(np.min(lows_np[idx - lookback:idx]))
    sweep = (lows_np[idx] < prior_low) and (closes_np[idx] > prior_low) and (closes_np[idx] > opens_np[idx])
    return bool(sweep)

def displacement_bullish(
    highs: List[float], lows: List[float], opens: List[float], closes: List[float], atr: np.ndarray,
    body_ratio=0.6, atr_mult=1.2
) -> bool:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if highs_np.size < 2 or atr.size != highs_np.size:
        return False
    idx = -1
    rng = highs_np[idx] - lows_np[idx]
    body = closes_np[idx] - opens_np[idx]
    if rng <= 0:
        return False
    cond = (closes_np[idx] > opens_np[idx]) and (rng > atr_mult * atr[idx]) and ((body / rng) >= body_ratio)
    return bool(cond)

def bullish_rsi_divergence(closes: List[float], rsi: np.ndarray, lb=10) -> bool:
    closes_np = np_safe(closes)
    if closes_np.size < lb + 3 or rsi.size != closes_np.size:
        return False
    try:
        p1 = int(np.argmin(closes_np[-lb-1:-1]))
        p2 = int(np.argmin(closes_np[-2*lb-1:-lb-1]))
    except ValueError:
        return False
    i1 = -lb + p1 - 1
    i2 = -2*lb + p2 - 1
    if abs(i1) >= closes_np.size or abs(i2) >= closes_np.size:
        return False
    price_ll = closes_np[i1] < closes_np[i2]
    rsi_hl = rsi[i1] > rsi[i2]
    return bool(price_ll and rsi_hl)

def whale_entry(volumes: List[float], closes: List[float], factor=3.0) -> bool:
    volumes_np = np_safe(volumes)
    closes_np = np_safe(closes)
    if volumes_np.size < 20 or closes_np.size < 2:
        return False
    last = volumes_np[-1]
    mean = float(np.mean(volumes_np[-20:]))
    return bool((last > mean * factor) and (closes_np[-1] > closes_np[-2]))

def cvd_imbalance_up(cvd: np.ndarray, bars=5, mult=1.6) -> bool:
    if cvd.size < bars + 1:
        return False
    slope = cvd[-1] - cvd[-bars]
    ref_seg = np.diff(cvd[-(bars + 10): -bars]) if cvd.size >= bars + 11 else np.diff(cvd)
    ref_std = safe_std(ref_seg)
    if ref_std == 0:
        return bool(slope > 0)
    return bool(slope > mult * ref_std)

def volatility_metric(closes: List[float], win=30) -> float:
    closes_np = np_safe(closes)
    if closes_np.size < win:
        return 0.0
    seg = closes_np[-win:]
    mu = float(np.mean(seg))
    return float(np.std(seg) / mu) if mu else 0.0

# ── FVG (Fair Value Gap) UTILITIES ──────────────────────────────────────────

def find_bullish_fvg_indices(highs: List[float], lows: List[float]) -> List[int]:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    out = []
    for n in range(2, len(highs_np)):
        if lows_np[n] > highs_np[n-2]:
            out.append(n)
    return out

def last_fvg_zone(highs: List[float], lows: List[float]) -> Optional[Tuple[float, float, int]]:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    idxs = find_bullish_fvg_indices(highs_np.tolist(), lows_np.tolist())
    if not idxs:
        return None
    n = idxs[-1]
    gap_top = lows_np[n]
    gap_bottom = highs_np[n-2]
    return (float(gap_top), float(gap_bottom), int(n))

def bullish_fvg_alert_logic(
    opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float], tf_label: str
) -> Optional[str]:
    opens_np = np_safe(opens)
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 60:
        return None
    zone = last_fvg_zone(highs_np.tolist(), lows_np.tolist())
    if not zone:
        return None
    gap_top, gap_bottom, idx_fvg = zone
    start = idx_fvg + 1
    if start + 3 >= closes_np.size:
        return None
    inside_idx = None
    for i in range(start, closes_np.size - 1):
        body_low = float(min(opens_np[i], closes_np[i]))
        body_high = float(max(opens_np[i], closes_np[i]))
        if (gap_bottom <= body_low) and (body_high <= gap_top):
            inside_idx = i
            break
    if inside_idx is None:
        return None
    cvd = cvd_proxy(closes_np.tolist(), volumes_np.tolist())
    ref_slice = volumes_np[max(0, inside_idx-20):inside_idx]
    ref_mean = float(np.mean(ref_slice)) if ref_slice.size else 0.0
    vol_rise = bool(volumes_np[inside_idx] > 1.25 * ref_mean)

    cvd_slice = cvd[max(0, inside_idx-20):inside_idx+1]
    cvd_diff = np.diff(cvd_slice) if cvd_slice.size >= 2 else np.array([])
    cvd_std = safe_std(cvd_diff)
    cvd_rise = bool(
        (cvd[inside_idx] - cvd[max(0, inside_idx-5)]) > 1.5 * cvd_std
    ) if cvd_std > 0 else bool(cvd[inside_idx] - cvd[max(0, inside_idx-5)] > 0)
    if not (vol_rise or cvd_rise):
        return None
    for j in range(inside_idx + 1, closes_np.size):
        if (closes_np[j] > gap_top) and (closes_np[j] > opens_np[j]):
            return f"Bullish FVG Confirmed ({tf_label})"
    return None

# ── DATA FETCH (multiple TF) ───────────────────────────────────────────────

INTERVALS_CORE = ["5m", "15m", "1h"]
INTERVALS_FVG  = ["1h", "4h", "1d"]

async def fetch_intervals(symbol: str, limits: Dict[str, int]) -> Dict[str, List[Any]]:
    async def _one(iv: str, lim: int) -> Tuple[str, List[Any]]:
        try:
            data = await get_klines(symbol, iv, lim)
            return iv, data if isinstance(data, list) else []
        except Exception as e:
            print(f"❌ Klines error {symbol} {iv}: {e}")
            return iv, []
    tasks = [_one(iv, limits.get(iv, 150)) for iv in set(INTERVALS_CORE + INTERVALS_FVG)]
    res = await asyncio.gather(*tasks)
    return {iv: data if isinstance(data, list) else [] for iv, data in res}

def parse_ohlcv(candles: list):
    if not candles:
        return [], [], [], [], [], []
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times

# ── RISK FILTERS / CONFLUENCE ──────────────────────────────────────────────

def htf_bullish_bias(closes_1h: List[float], closes_4h: List[float]) -> bool:
    if len(closes_1h) < 60:
        return False
    ema1h = ema_series(closes_1h, 50)
    cond_1h = (closes_1h[-1] > ema1h[-1]) and (ema1h[-1] > ema1h[-6])
    cond_4h = False
    if len(closes_4h) >= 60:
        ema4h = ema_series(closes_4h, 50)
        cond_4h = (closes_4h[-1] > ema4h[-1]) and (ema4h[-1] > ema4h[-6])
    return bool(cond_1h or cond_4h)

def atr_vol_gate(highs_15: List[float], lows_15: List[float], closes_15: List[float]) -> bool:
    atr15 = atr_series(highs_15, lows_15, closes_15, 14)
    atr_slice = atr15[-100:] if atr15.size >= 100 else atr15
    if atr_slice.size < 20:
        return False
    cur_atr = float(atr15[-1])
    p40 = percentile(atr_slice, 40)
    return bool(cur_atr >= p40 and cur_atr > 0)

LAST_ALERT_AT: Dict[str, datetime] = {}
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))

def cooldown_ok(symbol: str) -> bool:
    now = datetime.now(timezone.utc)
    last = LAST_ALERT_AT.get(symbol)
    if last is None:
        return True
    return (now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def mark_alert(symbol: str):
    LAST_ALERT_AT[symbol] = datetime.now(timezone.utc)

# ── SIGNAL ENGINE ──────────────────────────────────────────────────────────

async def detect_signals(symbol: str) -> List[str]:
    limits = {"5m": 240, "15m": 240, "1h": 240, "4h": 240, "1d": 240}
    data_map = await fetch_intervals(symbol, limits)
    if any(len(data_map.get(iv, [])) == 0 for iv in ["5m", "15m", "1h"]):
        return []
    o5, h5, l5, c5, v5, _ = parse_ohlcv(data_map["5m"])
    o15, h15, l15, c15, v15, _ = parse_ohlcv(data_map["15m"])
    o1h, h1h, l1h, c1h, v1h, _ = parse_ohlcv(data_map["1h"])
    o4h, h4h, l4h, c4h, v4h, _ = parse_ohlcv(data_map.get("4h", []))
    o1d, h1d, l1d, c1d, v1d, _ = parse_ohlcv(data_map.get("1d", []))
    if not htf_bullish_bias(c1h, c4h):
        return []
    if not atr_vol_gate(h15, l15, c15):
        return []
    atr5 = atr_series(h5, l5, c5, 14)
    atr15 = atr_series(h15, l15, c15, 14)
    rsi15 = rsi_series(c15, 14)
    cvd5 = cvd_proxy(c5, v5)
    vol_spike_5m = bool(len(v5) > 2 and v5[-1] > 2.0 * v5[-2])
    cvd_up = cvd_imbalance_up(cvd5, bars=5, mult=1.6)
    whale_up = whale_entry(v5, c5, factor=3.0)
    ssl_sweep_5m = sell_side_liquidity_sweep_bullish(h5, l5, o5, c5, lookback=20)
    ssl_sweep_15m = sell_side_liquidity_sweep_bullish(h15, l15, o15, c15, lookback=20)
    disp_5m = displacement_bullish(h5, l5, o5, c5, atr5, body_ratio=0.6, atr_mult=1.2)
    disp_15m = displacement_bullish(h15, l15, o15, c15, atr15, body_ratio=0.6, atr_mult=1.2)
    rsi_div_bull = bullish_rsi_divergence(c15, rsi15, lb=10)
    volume_5m_sum = float(sum(v5[-12:])) if len(v5) else 0.0
    volume_1h_sum = float(sum(v1h[-24:])) if len(v1h) else 0.0
    vdelta_1h = float(abs(v1h[-1] - v1h[0])) if len(v1h) > 1 else 0.0
    custom_1 = bool(volatility_metric(c15, win=30) > 0.5 and volume_5m_sum > 2_500_000)
    custom_2 = bool((vdelta_1h / volume_1h_sum > 0.2 and volume_5m_sum > 400_000) if volume_1h_sum else False)
    short_tf_votes = 0
    if vol_spike_5m:
        short_tf_votes += 1
    if cvd_up:
        short_tf_votes += 1
    if whale_up:
        short_tf_votes += 1
    if ssl_sweep_5m:
        short_tf_votes += 1
    if disp_5m:
        short_tf_votes += 1
    if rsi_div_bull:
        short_tf_votes += 1
    if short_tf_votes < 2:
        if not (ssl_sweep_15m and disp_15m):
            return []
    reasons: List[str] = []
    if vol_spike_5m:
        reasons.append("Volume Spike (5m)")
    if rsi_div_bull:
        reasons.append("RSI Bullish Divergence (15m)")
    if cvd_up:
        reasons.append("CVD Imbalance Up (5m)")
    if whale_up:
        reasons.append("Whale Entry (5m)")
    if ssl_sweep_5m or ssl_sweep_15m:
        reasons.append("Sell-side Liquidity Sweep")
    if (ssl_sweep_5m and disp_5m) or (ssl_sweep_15m and disp_15m):
        reasons.append("Smart Money Entry (Displacement)")
    if custom_1:
        reasons.append("Custom Filter 1")
    if custom_2:
        reasons.append("Custom Filter 2")
    fvg_alerts: List[str] = []
    for tf, candles in [("1h", data_map.get("1h", [])),
                        ("4h", data_map.get("4h", [])),
                        ("1d", data_map.get("1d", []))]:
        if not candles:
            continue
        o, h, lows, c, v, _ = parse_ohlcv(candles)
        label = tf.upper()
        alert = bullish_fvg_alert_logic(o, h, lows, c, v, label)
        if alert:
            fvg_alerts.append(alert)
    if fvg_alerts:
        reasons.extend(fvg_alerts)
    bullish_gate = any([
        any("Smart Money Entry" in r for r in reasons),
        any("Sell-side Liquidity Sweep" in r for r in reasons),
        any("Bullish FVG Confirmed" in r for r in reasons),
        whale_up, cvd_up, rsi_div_bull, vol_spike_5m
    ])
    if not bullish_gate:
        return []
    if not cooldown_ok(symbol):
        return []
    reasons = sorted(set(reasons), key=reasons.index)
    return reasons

# ── ALERT AND LOGGING ──────────────────────────────────────────────────────

async def send_alert(symbol: str, reasons: List[str]):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    coin_display = f"`{symbol}`"  # backtick around coin name
    message = (
        "🚨 *Bullish Signal Detected!*\n"
        f"*Coin:* {coin_display}\n"
        f"*Time:* {timestamp} UTC\n"
        f"*Reasons:* {' + '.join(reasons)}"
    )
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        mark_alert(symbol)
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def log_to_supabase(symbol: str, reasons: List[str]):
    try:
        # Supabase signals table: coin (string), timestamp (string), reasons (string)
        supabase.table("signals").insert({
            "coin": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasons": ", ".join(reasons)
        }).execute()
    except Exception as e:
        print(f"❌ Supabase log error: {e}")

# ── RATE-LIMIT FRIENDLY BATCHED SCAN ───────────────────────────────────────

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def scan_one_with_backoff(coin):
    max_attempts = 5
    delay = 1
    for attempt in range(max_attempts):
        try:
            reasons = await detect_signals(coin)
            if reasons:
                await send_alert(coin, reasons)
                log_to_supabase(coin, reasons)
                await asyncio.sleep(SLEEP_TIME)
            break
        except RateLimitError as e:
            ra = e.retry_after or delay
            print(f"⏳ Rate limit for {coin}. Retry after {ra}s (attempt {attempt+1})")
            await asyncio.sleep(ra)
            delay = min(delay * 2, 60)
        except Exception as e:
            print(f"❌ Error with {coin}: {e}")
            break

async def batch_scan_all_with_rate_limit():
    global binance_sem
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    exinfo = await get_exchange_info()
    coins = [
        s["symbol"]
        for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
    ]
    coins = [c for c in coins if not any(x in c for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])]

    print(f"🔍 Total {len(coins)} USDT spot symbols. Scanning in batches of {BINANCE_BATCH_SIZE} ...")

    for batch in chunked(coins, BINANCE_BATCH_SIZE):
        print(f"🔎 Scanning batch of {len(batch)} coins...")
        async def limited_scan(coin):
            if binance_sem is None:
                raise RuntimeError("Semaphore is not initialized")
            async with binance_sem:
                await scan_one_with_backoff(coin)
        tasks = [limited_scan(coin) for coin in batch]
        await asyncio.gather(*tasks)
        print("⏳ Sleeping 60 seconds for next batch...\n")
        await asyncio.sleep(60)

async def scanner_loop():
    while True:
        try:
            await batch_scan_all_with_rate_limit()
        except Exception as e:
            print(f"❌ Scanner loop error: {e}")
            print("⏳ Sleeping 10 minutes...\n")
            await asyncio.sleep(600)

# ── TELEGRAM COMMAND PARSING & HANDLERS ───────────────────────────────────

def parse_command(text: str) -> Tuple[List[str], str]:
    clean = " ".join(text.strip().split()).lower()
    actions = ["check", "remove", "haram", "add again", "halal"]
    action = next((a for a in actions if a in clean), "")
    coins = []
    if action:
        words = clean.split()
        for word in words:
            if word not in actions:
                coin = word.upper()
                coins.append(coin if coin.endswith("USDT") else f"{coin}USDT")
    else:
        matches = re.findall(r"([A-Z]{2,})/USDT", text.upper())
        coins = [f"{coin}USDT" for coin in matches]
        action = "Check" if coins else ""
    return coins, action.capitalize()

def get_username(update: TGUpdate) -> str:
    user = update.effective_user
    return user.username or str(user.id) if user else "Unknown"

def is_admin(username: Optional[str]) -> bool:
    return username == "RedwanICT"

async def start(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text(
            f"⛔ Sorry @{username},\n"
            "you do not have permission to use the Crypto Watchlist Tracking Bot.\n"
            "Please contact @RedwanICT – CEO of @CryptoICT_BD"
        )
        return
    await update.message.reply_text(
        f"👋 Hello @{username}! Send coin commands like:\n"
        "BTCUSDT SOLUSDT Check\n"
        "ETH XRP Remove\n"
        "BNB Haram\n"
        "BTC Add Again\n"
        "BNB Halal",
        parse_mode="Markdown"
    )

async def status(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    try:
        watchlist = get_column("watchlist")
        haram = get_column("haram")
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return
    reply_parts: List[str] = []
    if watchlist:
        reply_parts.append("📊 Watchlist:\n" + "\n".join(f"`{c}`" for c in watchlist))  # backtick
    if haram:
        reply_parts.append("⚠️ Haram:\n" + "\n".join(f"`{c}`" for c in haram))  # backtick
    reply = "\n\n".join(reply_parts) or "ℹ️ No data found."
    await update.message.reply_text(reply, parse_mode="Markdown")

async def handle_commands(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    username = get_username(update)
    if not is_admin(username):
        await update.message.reply_text("⛔ You are not authorized to use commands.")
        return
    text = update.message.text.strip()
    print(f"🔔 Command from {username}: {text}")
    coins, action = parse_command(text)
    valid_actions = {"Check", "Remove", "Haram", "Add again", "Halal"}
    if not coins or action not in valid_actions:
        await update.message.reply_text(
            "❌ Invalid format. Use keywords like:\n"
            "Check, Remove, Haram, Add Again, Halal\n"
            "Example: BTC ETH check",
            parse_mode="Markdown"
        )
        return
    try:
        watchlist = get_column("watchlist")
        haram = get_column("haram")
        removed_map = get_removed_map()
    except Exception as e:
        await update.message.reply_text(f"❌ Supabase error: {e}")
        return
    already, added, removed, marked_haram, unharamed = [], [], [], [], []
    for coin in coins:
        coin_display = f"`{coin}`"  # backtick for display
        if action == "Check":
            if coin in haram:
                marked_haram.append(coin_display)
            elif coin in removed_map:
                removed.append(f"{coin_display} - {removed_map[coin]}")
            elif coin in watchlist:
                already.append(coin_display)
            else:
                add_coin_with_date("watchlist", coin)
                added.append(coin_display)
        elif action == "Remove":
            if remove_coin_from_table("watchlist", coin):
                add_coin_with_date("removed", coin)
                removed.append(coin_display)
        elif action == "Haram":
            add_coin("haram", coin)
            marked_haram.append(coin_display)
            if coin in watchlist and remove_coin_from_table("watchlist", coin):
                add_coin_with_date("removed", coin)
                removed.append(f"{coin_display} (removed from watchlist due to haram)")
        elif action == "Add again":
            if coin in removed_map:
                add_coin_with_date("watchlist", coin)
                remove_coin_from_table("removed", coin)
                added.append(coin_display)
        elif action == "Halal":
            if remove_coin_from_table("haram", coin):
                unharamed.append(coin_display)
    reply_parts: List[str] = []
    if already:
        reply_parts.append("🟢 Already in Watchlist:\n" + "\n".join(already))
    if added:
        reply_parts.append("✅ New Added:\n" + "\n".join(added))
    if marked_haram:
        reply_parts.append("⚠️ Marked as Haram:\n" + "\n".join(marked_haram))
    if removed:
        reply_parts.append("🗑️ Removed:\n" + "\n".join(removed))
    if unharamed:
        reply_parts.append("✅ Removed from Haram:\n" + "\n".join(unharamed))
    reply = "\n\n".join(reply_parts) or "✅ No changes made."
    await update.message.reply_text(reply, parse_mode="Markdown")

# ── BOT ENTRYPOINT ─────────────────────────────────────────────────────────

async def post_init(application):
    global aiohttp_session, binance_sem
    aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, ssl=False),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        headers={"User-Agent": "CryptoWatchlistBot/1.2 (+contact)"}
    )
    binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    application.create_task(scanner_loop())
    print("✅ post_init completed: aiohttp session created and scanner started.")

def main() -> None:
    keep_alive()
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"(?i)^status$"), status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_commands))
    print("🚀 Bot starting (press Ctrl+C to stop)...")
    app.run_polling()

if __name__ == "__main__":
    main()
