# -*- coding: utf-8 -*-
import asyncio
import re
import numpy as np
from typing import Any, List, Sequence, Tuple, Optional, Dict, Protocol, cast
from collections.abc import Mapping, Iterable as IterableABC

from telegram import (
    Update as TGUpdate,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message as TGMessage,
)
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from data_api import (
    get_exchange_info,
    get_klines,
    get_column,
    get_removed_map,
    get_market_cap,
)

# NEW: use watchlist actions via existing helper
from supabase_helpers import process_watchlist_action_text


# -----------------------------------------------------------------------
# Configuration (tweak these to switch strict/relaxed behavior)
# -----------------------------------------------------------------------
ALLOW_PRIOR_BREAKOUTS: int = 3
DEFAULT_KLINES_LOOKBACK = 200
TREAT_WICK_AS_BREAKDOWN: bool = True


# =========================
# -------- OHLCV ----------
# =========================

def parse_ohlcv(candles: Sequence[Sequence[Any]]) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[int]]:
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times


# =========================
# ---- Swing helpers ------
# =========================

def swing_high_idx(highs: List[float], lookback: int = 3) -> List[int]:
    res: List[int] = []
    for i in range(lookback, len(highs) - lookback):
        window = highs[i - lookback : i + lookback + 1]
        if highs[i] == max(window):
            res.append(i)
    return res


def swing_low_idx(lows: List[float], lookback: int = 3) -> List[int]:
    res: List[int] = []
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] == min(window):
            res.append(i)
    return res


# =========================
# --- MSS & BOS (trend) ---
# =========================

def has_bull_mss(highs: List[float], lows: List[float], closes: List[float], lookback: int = 20) -> bool:
    lows_idx = swing_low_idx(lows, lookback=2)
    if len(lows_idx) < 2:
        return False
    last_sw_low = lows_idx[-1]
    prev_sw_low = lows_idx[-2]
    swept = lows[last_sw_low] < lows[prev_sw_low]
    if not swept:
        return False
    highs_idx = swing_high_idx(highs, lookback=2)
    for i in range(prev_sw_low, len(closes)):
        if any(closes[i] > highs[h] for h in highs_idx if h < i):
            return True
    return False


def has_bull_bos(highs: List[float], closes: List[float], lookback: int = 20) -> bool:
    swing_highs = swing_high_idx(highs, lookback=2)
    if not swing_highs:
        return False
    last_swing_high = swing_highs[-1]
    return bool(closes[-1] > highs[last_swing_high])


# =========================
# ---- Displacement -------
# =========================

def is_displacement_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float], i: int, atr_win: int = 14, body_k: float = 0.5, atr_mult: float = 1.2) -> bool:
    if i <= 0 or i >= len(closes):
        return False

    body = abs(closes[i] - opens[i])
    tr_arr: List[float] = []
    for j in range(1, len(closes)):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j - 1]),
            abs(lows[j] - closes[j - 1]),
        )
        tr_arr.append(tr)

    if not tr_arr:
        return False

    if len(tr_arr) >= atr_win:
        atr = float(np.mean(tr_arr[-atr_win:]))
    else:
        atr = float(np.mean(tr_arr))

    rng = highs[i] - lows[i]
    if rng <= 0:
        return False

    body_ratio = body / rng
    size_ok = (atr > 0 and body >= atr_mult * atr)
    dominance_ok = body_ratio >= body_k
    return bool(size_ok and dominance_ok)


# =========================
# -------- RSI ------------
# =========================

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return []
    deltas = np.diff(values)
    seed = deltas[:period]
    up = float(np.sum(seed[seed >= 0])) / period
    down = float(-np.sum(seed[seed < 0])) / period
    rs = up / down if down != 0 else 0.0
    rsi_values = [100 - (100 / (1 + rs))]
    up_avg, down_avg = up, down
    for delta in deltas[period:]:
        up_val = float(max(delta, 0))
        down_val = float(-min(delta, 0))
        up_avg = (up_avg * (period - 1) + up_val) / period
        down_avg = (down_avg * (period - 1) + down_val) / period
        rs = up_avg / down_avg if down_avg != 0 else 0.0
        rsi_values.append(100 - (100 / (1 + rs)))
    return [np.nan] * period + rsi_values


def bullish_rsi_divergence(closes: List[float], lows: List[float]) -> bool:
    rsi_vals = rsi(closes)
    if len(rsi_vals) < 5:
        return False
    price_low1, price_low2 = lows[-5], lows[-1]
    rsi_low1, rsi_low2 = rsi_vals[-5], rsi_vals[-1]
    return bool(price_low2 < price_low1 and rsi_low2 > rsi_low1)


# =========================
# ---- Other metrics ------
# =========================

def relative_strength(closes: List[float]) -> float:
    if len(closes) >= 20:
        return float(closes[-1]) / float(np.mean(closes[-20:]))
    else:
        return 0.0


def cvd_proxy(closes: List[float], volumes: List[float]) -> float:
    delta = [
        float(v) if closes[i] > closes[i - 1] else -float(v)
        for i, v in enumerate(volumes)
        if i > 0
    ]
    return float(np.sum(delta))


def volatility_metric(closes: List[float], win: int = 30) -> float:
    arr = np.array(closes, dtype=float)
    if arr.size < win:
        return 0.0
    mu = float(np.mean(arr[-win:]))
    return float(np.std(arr[-win:]) / mu) if mu else 0.0


def mean_volume(volumes: List[float], win: int = 30) -> float:
    arr = np.array(volumes, dtype=float)
    if arr.size < win:
        return float(np.mean(arr)) if arr.size else 0.0
    return float(np.mean(arr[-win:]))


def change_24h_from_series(closes: List[float], tf: str) -> float:
    if tf == "1h" and len(closes) > 24:
        base = closes[-25]
        return (closes[-1] - base) / base * 100 if base else 0.0
    if tf == "4h" and len(closes) > 6:
        base = closes[-7]
        return (closes[-1] - base) / base * 100 if base else 0.0
    if tf == "1d" and len(closes) > 1:
        base = closes[-2]
        return (closes[-1] - base) / base * 100 if base else 0.0
    return 0.0


# =========================================
# ------- Bullish FVG core (strict) -------
# =========================================
# -------------- UNCHANGED -----------------
def last_bullish_fvg_status(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    allow_prior_breakouts: int = ALLOW_PRIOR_BREAKOUTS,
) -> Optional[Tuple[str, float, float, float, int]]:
    idxs: List[int] = []
    for n in range(2, len(highs)):
        if lows[n] > highs[n - 2]:
            idxs.append(n)
    if not idxs:
        return None

    for n in reversed(idxs):
        if n >= len(closes) - 1:
            continue

        gap_bottom = float(highs[n - 2])
        gap_top = float(lows[n])
        gap_size = gap_top - gap_bottom
        if gap_size <= 0:
            continue

        if gap_bottom <= 0 or (gap_size / gap_bottom) < 0.03:
            continue

        mid = gap_bottom + 0.5 * gap_size

        breakout_idxs: List[int] = []
        breakdown_idxs: List[int] = []
        for i in range(n + 1, len(closes)):
            if closes[i] > gap_top and closes[i] > opens[i]:
                breakout_idxs.append(i)
            if (lows[i] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[i] < gap_bottom):
                breakdown_idxs.append(i)

        if breakdown_idxs:
            continue

        if len(breakout_idxs) > allow_prior_breakouts:
            continue

        if len(breakout_idxs) > 1:
            continue

        pre_inside_respected = False
        for i in range(n + 1, len(closes)):
            body_low_i = min(opens[i], closes[i])
            if body_low_i >= gap_bottom and body_low_i <= gap_top:
                break
            if lows[i] <= mid and closes[i] > gap_top and closes[i] > opens[i]:
                pre_inside_respected = True
                break
        if pre_inside_respected:
            continue

        inside_idx: Optional[int] = None
        for i in range(n + 1, len(closes)):
            body_low = min(opens[i], closes[i])
            if body_low >= gap_bottom and body_low <= gap_top:
                inside_idx = i
                break
        if inside_idx is None:
            continue

        breakouts_before_inside = [b for b in breakout_idxs if b < inside_idx]
        if len(breakouts_before_inside) > allow_prior_breakouts:
            continue

        failed_after_inside = False
        for k in range(inside_idx, len(closes)):
            if (lows[k] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[k] < gap_bottom):
                failed_after_inside = True
                break
        if failed_after_inside:
            continue

        respected = False
        respect_idx: Optional[int] = None
        mid_touched = False

        for i in range(inside_idx + 1, len(closes)):
            if lows[i] <= mid:
                mid_touched = True

            if mid_touched and closes[i] > gap_top and closes[i] > opens[i]:
                respected = True
                respect_idx = i
                break

        breakouts_after_inside = [b for b in breakout_idxs if b >= inside_idx]
        if breakouts_after_inside and respect_idx is None:
            continue
        if breakouts_after_inside and respect_idx is not None:
            if breakouts_after_inside[0] != respect_idx:
                continue

        if respected and respect_idx is not None:
            double_after = False
            zone_pct = (gap_top - gap_bottom) / gap_bottom
            for j in range(respect_idx + 1, len(closes)):
                pump_pct = (closes[j] - gap_top) / gap_top
                if pump_pct >= zone_pct:
                    double_after = True
                    break
            if double_after:
                continue

        if respected and respect_idx is not None:
            for k in range(respect_idx, len(closes)):
                if (lows[k] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[k] < gap_bottom):
                    respected = False
                    respect_idx = None
                    break

        if respected:
            return ("RESPECTED", gap_bottom, gap_top, gap_size, n)
        else:
            return ("INSIDE", gap_bottom, gap_top, gap_size, n)

    return None
# -------------- UNCHANGED -----------------


# =========================================
# ------- Market Cap & Category -----------
# =========================================
def marketcap_category(mcap: Optional[float]) -> Tuple[str, str]:
    if mcap is None:
        return "Unknown", "⚫"
    if mcap >= 10_000_000_000:
        return "High", "🟢"
    elif mcap >= 1_000_000_000:
        return "Mid", "🟡"
    elif mcap >= 100_000_000:
        return "Low", "🟠"
    else:
        return "Micro", "🔴"


# =========================================
# ------- FVG Win Rate & TP Calc ----------
# =========================================

async def analyze_fvg_win_rate(symbol: str, tf: str, opens: List[float], highs: List[float], lows: List[float], closes: List[float], fvg_bottom: float, fvg_top: float, fvg_idx: int) -> Dict[str, Any]:
    """
    Simplified win-rate analysis: keep the original 5% win-rate computation,
    but remove the Best TP search and its extra calculations as requested.
    """
    n = len(highs)
    zones: List[Tuple[int, float, float]] = []
    for i in range(2, max(3, n - 8)):
        if lows[i] > highs[i - 2]:
            gb = float(highs[i - 2])
            gt = float(lows[i])
            if gb > 0 and (gt - gb) / gb >= 0.03:
                zones.append((i, gb, gt))

    zones = zones[-100:]
    results: List[bool] = []
    for idx, gb, gt in zones:
        entry = closes[idx]
        sl = gb
        win = False
        for j in range(idx + 1, min(n, idx + 8)):
            # keep the original 5% TP check for the win-rate calc
            if highs[j] >= entry * 1.05:
                win = True
                break
            if lows[j] < sl:
                win = False
                break
        results.append(win)

    win_rate = (sum(results) / len(results) * 100) if results else 0.0

    # Return only the 5% win rate — do not compute Best TP or related success-rate.
    return {
        "win_rate_5pct": win_rate,
    }


# =========================================
# ------------- MAIN SCAN -----------------
# =========================================

async def get_fvg_coins_async(tf: str) -> List[Dict[str, Any]]:
    exinfo = await get_exchange_info()
    symbols: List[str] = [
        s["symbol"]
        for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]

    watchlist_raw = await get_column("watchlist")
    haram_raw = await get_column("haram")

    watchlist = set([s.upper() for s in (watchlist_raw or [])])
    haram = set([s.upper() for s in (haram_raw or [])])

    removed_raw: Any = await get_removed_map()
    removed: set[str]
    if isinstance(removed_raw, Mapping):
        removed = {str(k).upper() for k in removed_raw.keys()}
    elif isinstance(removed_raw, IterableABC):
        removed = {str(x).upper() for x in removed_raw}
    else:
        removed = set()

    sem = asyncio.Semaphore(8)
    results: List[Optional[Dict[str, Any]]] = []

    async def check_symbol(sym: str) -> Optional[Dict[str, Any]]:
        async with sem:
            try:
                candles = await get_klines(sym, tf, DEFAULT_KLINES_LOOKBACK)
                opens, highs, lows, closes, volumes, times = parse_ohlcv(candles)
                if len(closes) < 60:
                    return None

                trend_ok = has_bull_bos(highs, closes) or has_bull_mss(highs, lows, closes)
                if not trend_ok:
                    return None

                fvg_info = last_bullish_fvg_status(opens, highs, lows, closes, allow_prior_breakouts=ALLOW_PRIOR_BREAKOUTS)
                if not fvg_info:
                    return None

                status_fvg, gap_bottom, gap_top, gap_size, fvg_idx = fvg_info

                current_below = (lows[-1] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[-1] < gap_bottom)
                if current_below:
                    return None

                disp_ok = is_displacement_candle(opens, highs, lows, closes, len(closes) - 2)
                rsi_div = bullish_rsi_divergence(closes, lows)
                rs_val = relative_strength(closes)
                cvd_val = cvd_proxy(closes, volumes)
                vol_metric = volatility_metric(closes)
                avg_vol = mean_volume(volumes)
                change_24h = change_24h_from_series(closes, tf)

                if sym in haram:
                    status = "HARAM"
                elif sym in removed:
                    status = "REMOVED"
                elif sym in watchlist:
                    status = "WATCHLIST"
                else:
                    status = "NEW"

                try:
                    mcap = await get_market_cap(sym)
                except Exception:
                    mcap = None

                mcat, memoji = marketcap_category(mcap)

                win_rate_analysis = None
                try:
                    win_rate_analysis = await analyze_fvg_win_rate(sym, tf, opens, highs, lows, closes, gap_bottom, gap_top, fvg_idx)
                except Exception:
                    win_rate_analysis = {}

                gap_size_pct = (gap_size / gap_bottom) * 100 if gap_bottom else 0.0

                return {
                    "symbol": sym,
                    "fvg_status": status_fvg,
                    "gap_bottom": gap_bottom,
                    "gap_top": gap_top,
                    "gap_size_pct": gap_size_pct,
                    "volatility": vol_metric,
                    "avg_volume": avg_vol,
                    "status": status,
                    "inside_fvg": gap_bottom <= closes[-1] <= gap_top,
                    "rsi_div": rsi_div,
                    "rs_val": rs_val,
                    "cvd_val": cvd_val,
                    "disp_candle": disp_ok,
                    "change_24h": change_24h,
                    "market_cap": mcap,
                    "market_cat": mcat,
                    "market_emoji": memoji,
                    "win_rate_analysis": win_rate_analysis,
                    "fvg_zone_text": f"{gap_bottom:.8f} – {gap_top:.8f}",
                    "fvg_idx": fvg_idx,
                    "tf": tf,
                }
            except Exception:
                return None

    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        tasks = [check_symbol(s) for s in symbols[i : i + batch_size]]
        results.extend(await asyncio.gather(*tasks))
        await asyncio.sleep(0.5)

    return [r for r in results if r]


# =========================================
# ----- Chart Button & Stylish Format -----
# =========================================

def tradingview_link(symbol: str) -> str:
    tv_symbol = symbol.replace("USDT", "USDT.P")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}"


def format_coin_message(c: Dict[str, Any], tf_label: str) -> str:
    # win rate display: keep only the 5% win rate as requested
    win_txt = ""
    win_data = c.get("win_rate_analysis") or {}
    wr = win_data.get("win_rate_5pct", None)

    if wr is not None:
        win_txt = win_txt + f"\n🏆 *FVG Win Rate*: `{wr:.2f}%`"

    signals: List[str] = []
    if c.get("rsi_div"):
        signals.append("📈 RSI Divergence")
    if c.get("rs_val", 0) > 1:
        signals.append("💪 Relative Strength↑")
    if c.get("cvd_val", 0) > 0:
        signals.append("📊 CVD↑")
    if c.get("disp_candle"):
        signals.append("🔥 Displacement")

    extra_str = " | ".join(signals) if signals else "—"

    avg_vol = c.get("avg_volume", 0)
    try:
        avg_vol_str = f"{int(avg_vol):,}"
    except Exception:
        avg_vol_str = str(avg_vol)

    # prettier status labels with emoji for NEW / REMOVED / WATCHLIST
    raw_status = c.get("status", "")
    status_map = {
        "NEW": "🆕 NEW",
        "REMOVED": "🗑️ REMOVED",
        "WATCHLIST": "⭐ WATCHLIST",
        "HARAM": "🚫 HARAM",
    }
    status_display = status_map.get(raw_status, raw_status)

    msg = (
        f"🚀 `{c.get('symbol','')}` — {c.get('market_emoji','')} {c.get('market_cat','')} Market Cap"
        f"\n🕒 Timeframe: *{tf_label}*"
        f"\n━━━━━━━━━━━━━━"
        f"\n✨ *FVG Status*: `{c.get('fvg_status','')}`"
        f"\n🔲 *FVG Zone*: `{c.get('fvg_zone_text','')}`"
        f"\n📏 *Gap Size*: `{c.get('gap_size_pct',0.0):.2f}%`"
        f"\n📊 *Volatility*: `{c.get('volatility',0.0):.4f}`"
        f"\n📦 *Avg Volume*: `{avg_vol_str}`"
        f"\n📈 *24H Change*: `{c.get('change_24h',0.0):.2f}%`"
        f"\n🔖 *Status*: `{status_display}`"
        f"\n🚦 Signals: {extra_str}"
        f"{win_txt}"
        f"\n━━━━━━━━━━━━━━"
        f"\n✨ Trade wisely!"
    )
    return msg


# =========================================
# ------- Smart Parsing & Filters ----------
# =========================================

def _parse_number_with_suffix(token: str) -> Optional[float]:
    """e.g. '2m' -> 2_000_000; '750k' -> 750_000"""
    m = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmbKMB])?\s*$', token)
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or '').lower()
    if suf == 'k':
        val *= 1_000
    elif suf == 'm':
        val *= 1_000_000
    elif suf == 'b':
        val *= 1_000_000_000
    return val

def parse_filters(text: str) -> Dict[str, Any]:
    raw = text.lower()

    # defaults
    cfg = {
        "top_n": None,                 # int
        "only_respected": False,       # True -> RESPECTED only
        "inside_only": False,          # True -> INSIDE only
        "high_volume": False,          # True -> top volume quartile
        "min_volume": None,            # float absolute
        "watchlist_only": False,       # only symbols already in watchlist
        "sort_by": "default",          # default|wr|gap|rs|vol|change|mcap
    }

    # top N
    m = re.search(r'\btop\s+(\d{1,3})\b', raw)
    if m:
        cfg["top_n"] = max(1, min(100, int(m.group(1))))

    # respected / inside
    if re.search(r'\b(strong|respected)\b', raw):
        cfg["only_respected"] = True
    if re.search(r'\binside\s+only\b', raw):
        cfg["inside_only"] = True

    # volume filters
    if re.search(r'\bhigh\s+vol(ume)?\b', raw):
        cfg["high_volume"] = True
    m2 = re.search(r'\bmin\s+vol(ume)?\s+([0-9]+(?:\.[0-9]+)?[kmbKMB]?)\b', raw)
    if m2:
        val = _parse_number_with_suffix(m2.group(2))
        if val is not None:
            cfg["min_volume"] = float(val)

    # watchlist only
    if re.search(r'\bwatchlist\s+only\b', raw):
        cfg["watchlist_only"] = True

    # sorting
    m3 = re.search(r'\bsort\s+(wr|gap|rs|vol|change|mcap)\b', raw)
    if m3:
        cfg["sort_by"] = m3.group(1)

    return cfg


def smart_parse_tf(text: str) -> Optional[Tuple[str, str]]:
    """
    Smart/Fuzzy TF parser.
    Returns (TF_LABEL, tf_key) like ("4H", "4h")
    """
    raw = text.strip()

    # Original strict format first (kept for backward compatibility)
    strict = re.match(
        r"^([0-9]+m|[0-9]+h|[0-9]+d|[0-9]+w|[0-9]+M)\s+FVG\s+COIN\s*LIST$",
        raw, re.IGNORECASE
    )
    if strict:
        tf_label = strict.group(1)
        tf = tf_label if (tf_label.endswith("M") and tf_label.isupper()) else tf_label.lower()
        return tf_label.upper(), tf

    s = raw.lower()

    # Must mention 'fvg' to qualify as this command family
    if "fvg" not in s:
        return None

    # Flexible phrases
    pairs = [
        (r'\b15\s*m(in)?\b', ("15M", "15m")),
        (r'\b30\s*m(in)?\b', ("30M", "30m")),
        (r'\b45\s*m(in)?\b', ("45M", "45m")),
        (r'\b1\s*h(our)?\b', ("1H", "1h")),
        (r'\b4\s*h(our)?\b', ("4H", "4h")),
        (r'\b1\s*d(ay)?\b', ("1D", "1d")),
        (r'\b1\s*w(eek|k)?\b', ("1W", "1w")),
    ]
    for pat, tfpair in pairs:
        if re.search(pat, s):
            return tfpair

    # If only “fvg” found but no TF -> ask user via keyboard (handled by handler)
    return None


def build_tf_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("1H", callback_data="FVG_TF|1H"),
            InlineKeyboardButton("4H", callback_data="FVG_TF|4H"),
            InlineKeyboardButton("1D", callback_data="FVG_TF|1D"),
        ],
        [
            InlineKeyboardButton("15m", callback_data="FVG_TF|15M"),
            InlineKeyboardButton("30m", callback_data="FVG_TF|30M"),
            InlineKeyboardButton("1W", callback_data="FVG_TF|1W"),
        ]
    ]
    return InlineKeyboardMarkup(rows)


# =========================================
# ------- Telegram Handler (SMART) --------
# =========================================

# Protocol for anything with 'reply_text' used by _scan_and_send
class MessageSender(Protocol):
    async def reply_text(self, text: str, **kwargs: Any) -> Any:
        ...


async def _scan_and_send(sender: MessageSender, context: ContextTypes.DEFAULT_TYPE, tf_label: str, tf_key: str, filters: Optional[Dict[str, Any]] = None) -> None:
    """Core routine: scan, filter/sort, send results + summary."""
    filters = filters or {}

    # initial message
    await sender.reply_text(
        f"⏳ Scanning Binance ({tf_label}) FVG coins...",
        parse_mode=ParseMode.MARKDOWN,
    )

    coins = await get_fvg_coins_async(tf_key)
    if not coins:
        await sender.reply_text(
            f"😔 No coins matched the criteria for {tf_label}. Try again later.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Apply filters
    data = coins

    if filters.get("only_respected"):
        data = [c for c in data if c.get("fvg_status") == "RESPECTED"]
    if filters.get("inside_only"):
        data = [c for c in data if c.get("fvg_status") == "INSIDE"]
    if filters.get("watchlist_only"):
        data = [c for c in data if c.get("status") == "WATCHLIST"]
    if filters.get("min_volume") is not None:
        mv = float(filters["min_volume"])
        data = [c for c in data if float(c.get("avg_volume", 0)) >= mv]
    if filters.get("high_volume"):
        vols = [float(c.get("avg_volume", 0)) for c in data]
        if vols:
            thr = sorted(vols)[int(0.75 * (len(vols)-1))]  # ~top quartile
            data = [c for c in data if float(c.get("avg_volume", 0)) >= thr]

    # Sorting
    def coin_sort_key_default(c: Dict[str, Any]) -> Tuple[int, float]:
        wr = (c.get("win_rate_analysis") or {}).get("win_rate_5pct", 0.0)
        respected = 1 if c.get("fvg_status") == "RESPECTED" else 0
        return respected, float(wr or 0.0)

    sort_by = filters.get("sort_by", "default")
    if sort_by == "default":
        data.sort(key=coin_sort_key_default, reverse=True)
    elif sort_by == "wr":
        data.sort(key=lambda c: float((c.get("win_rate_analysis") or {}).get("win_rate_5pct") or 0.0), reverse=True)
    elif sort_by == "gap":
        data.sort(key=lambda c: float(c.get("gap_size_pct", 0.0)), reverse=True)
    elif sort_by == "rs":
        data.sort(key=lambda c: float(c.get("rs_val", 0.0)), reverse=True)
    elif sort_by == "vol":
        data.sort(key=lambda c: float(c.get("avg_volume", 0.0)), reverse=True)
    elif sort_by == "change":
        data.sort(key=lambda c: float(c.get("change_24h", 0.0)), reverse=True)
    elif sort_by == "mcap":
        data.sort(key=lambda c: float(c.get("market_cap") or 0.0), reverse=True)

    # top N limit
    top_n = filters.get("top_n")
    if isinstance(top_n, int) and top_n > 0:
        data = data[: top_n]

    # Send rows
    respected_cnt = 0
    inside_cnt = 0

    for c in data:
        if c.get("fvg_status") == "RESPECTED":
            respected_cnt += 1
        elif c.get("fvg_status") == "INSIDE":
            inside_cnt += 1

        msg = format_coin_message(c, tf_label)

        symbol = c.get("symbol", "")
        status = c.get("status", "")
        is_watch = (status == "WATCHLIST")
        # removed unused is_haram assignment to avoid linter warning

        wl_buttons = []
        if is_watch:
            wl_buttons.append(InlineKeyboardButton("➖ Remove WL", callback_data=f"WL|REMOVE|{symbol}"))
        else:
            wl_buttons.append(InlineKeyboardButton("➕ Add WL", callback_data=f"WL|ADD|{symbol}"))

        # Removed the Haram/Halal button from message options as requested.

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Chart", url=tradingview_link(symbol))],
            wl_buttons
        ])

        try:
            await sender.reply_text(
                msg,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
            )
        except Exception:
            await sender.reply_text(
                msg,
                parse_mode=ParseMode.MARKDOWN,
            )
        await asyncio.sleep(0.25)

    # Summary
    await sender.reply_text(
        f"📌 *Summary* — {tf_label}\n"
        f"• Total: `{len(data)}`\n"
        f"• RESPECTED: `{respected_cnt}`\n"
        f"• INSIDE: `{inside_cnt}`",
        parse_mode=ParseMode.MARKDOWN
    )


async def fvg_coinlist_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    message = update.message
    text = cast(str, message.text).strip()

    # Parse filters first (they do not affect detection, only view)
    filters = parse_filters(text)

    # Smart timeframe parsing (backward compatible)
    tf = smart_parse_tf(text)

    if not tf:
        # If message mentions FVG but TF missing -> suggest with keyboard
        if re.search(r'\bfvg\b', text, re.IGNORECASE):
            await message.reply_text(
                "🧠 Choose a timeframe for FVG scan:",
                reply_markup=build_tf_keyboard()
            )
            return

        # Otherwise show smart suggestion
        await message.reply_text(
            "⚠️ Couldn’t parse your command.\n"
            "Examples:\n"
            "• `4H FVG Coin List`\n"
            "• `show 1d fvg top 10 strong only sort wr`\n"
            "• `fvg 4h high volume min volume 2m`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    tf_label, tf_key = tf
    # Pass message object as the 'sender' so _scan_and_send can reply via same message context
    await _scan_and_send(cast(MessageSender, message), context, tf_label, tf_key, filters=filters)


# =========================================
# ------------- Callback Handlers ----------
# =========================================

async def fvg_tf_callback_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle timeframe selection from inline keyboard."""
    if not update.callback_query:
        return
    query = update.callback_query
    await query.answer()

    # query.message can be None (MaybeInaccessibleMessage) — check and bail if missing
    if query.message is None:
        try:
            await query.edit_message_text("⚠️ Unable to run scan: original message not available.")
        except Exception:
            pass
        return

    # Cast to TGMessage before using members
    qmsg = cast(TGMessage, query.message)

    data = (query.data or "")
    # Expected: FVG_TF|1H
    m = re.match(r'^FVG_TF\|([0-9]+[mMhHdDwW])$', data)
    if not m:
        await qmsg.reply_text("⚠️ Invalid TF selection.")
        return

    tf_label = m.group(1).upper()
    # Normalize into key
    tf_key = tf_label if (tf_label.endswith("M") and tf_label.isupper()) else tf_label.lower()

    # Inform user and then run scan, using the callback message as sender
    await qmsg.reply_text(f"⏳ Scanning Binance ({tf_label}) FVG coins...", parse_mode=ParseMode.MARKDOWN)

    # Cast the message object to TGMessage (helps static checkers)
    sender_msg = cast(TGMessage, query.message)
    await _scan_and_send(cast(MessageSender, sender_msg), context, tf_label, tf_key, filters={})


async def watchlist_callback_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Add/Remove/Haram/Halal buttons."""
    if not update.callback_query:
        return
    query = update.callback_query
    await query.answer()

    if query.message is None:
        try:
            await query.edit_message_text("⚠️ Unable to perform action: original message not available.")
        except Exception:
            pass
        return

    qmsg = cast(TGMessage, query.message)

    data = (query.data or "")
    # Expected: WL|ADD|BTCUSDT  / WL|REMOVE|BTCUSDT / WL|HARAM|... / WL|HALAL|...
    m = re.match(r'^WL\|(ADD|REMOVE|HARAM|HALAL)\|([A-Z0-9]+)$', data)
    if not m:
        await qmsg.reply_text("⚠️ Invalid action.")
        return

    action, symbol = m.group(1), m.group(2)
    # Compose a command string that the existing helper understands
    cmd_map = {
        "ADD":    f"{symbol} Add",
        "REMOVE": f"{symbol} Remove",
        "HARAM":  f"{symbol} Haram",
        "HALAL":  f"{symbol} Halal",
    }
    cmd_text = cmd_map[action]

    try:
        # Reuse existing permission logic & DB ops
        # process_watchlist_action_text expects (update, text) in original design,
        # so we call it with the original update (callback update) and the command text.
        reply = await process_watchlist_action_text(update, cmd_text)
        if reply:
            await qmsg.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
        else:
            await qmsg.reply_text("ℹ️ No change.")
    except Exception as e:
        await qmsg.reply_text(f"❌ Failed: {e}")
