# -*- coding: utf-8 -*-
import asyncio
import re
import numpy as np
from datetime import datetime
from typing import Any, List, Sequence, Tuple, Optional, Dict
from collections.abc import Mapping, Iterable as IterableABC

from telegram import Update as TGUpdate, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from data_api import (
    get_exchange_info,
    get_klines,
    get_column,
    get_removed_map,
    get_market_cap,
)


# -----------------------------------------------------------------------
# Configuration (tweak these to switch strict/relaxed behavior)
# -----------------------------------------------------------------------
ALLOW_PRIOR_BREAKOUTS: int = 3  # STRICT MODE: skip any FVG zone that had more than this many breakouts BEFORE inside
DEFAULT_KLINES_LOOKBACK = 200

# If True, any wick/low that dips below gap_bottom will be treated as a breakdown.
# If False, only candle closes below gap_bottom count as breakdowns (legacy behavior).
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

def last_bullish_fvg_status(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    allow_prior_breakouts: int = ALLOW_PRIOR_BREAKOUTS,
) -> Optional[Tuple[str, float, float, float, int]]:
    """
    Modified FVG detection to allow one breakout that happens AFTER price first enters the FVG (inside),
    so that a first post-inside breakout can be considered a RESPECTED alert.

    Rules (summary):
    - Detect gap where lows[n] > highs[n-2]
    - Require gap size >= 3%
    - If any breakdown (wick/close depending on config) happens after gap creation -> skip zone
    - Count breakouts that happen BEFORE the inside candle and skip zone if count > allow_prior_breakouts
    - If price enters the FVG (inside) we detect that based on the BOTTOM of the body lying within the FVG:
        body_low in [gap_bottom, gap_top] is considered "inside" (this matches examples where body dips into FVG)
    - After inside, allow the first breakout AFTER inside to be the RESPECT candle (i.e., we permit it so zone can become RESPECTED)
    - Avoid double pumps, breakdown after respect, etc.
    Returns: ("INSIDE"|"RESPECTED", gap_bottom, gap_top, gap_size, fvg_idx) or None
    """
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

        # size >= 3%
        if gap_bottom <= 0 or (gap_size / gap_bottom) < 0.03:
            continue

        mid = gap_bottom + 0.5 * gap_size

        # scan for breakouts / breakdowns after gap creation
        breakout_idxs: List[int] = []
        breakdown_idxs: List[int] = []
        for i in range(n + 1, len(closes)):
            if closes[i] > gap_top and closes[i] > opens[i]:
                breakout_idxs.append(i)
            # breakdown detection: use lows (wick) if configured, otherwise use closes
            if (lows[i] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[i] < gap_bottom):
                breakdown_idxs.append(i)

        # STRICT RULE: if ANY breakdown occurred after gap creation -> skip zone forever!
        if breakdown_idxs:
            continue

        # STRICT RULE: if ANY breakout already occurred after gap creation -> skip zone
        if len(breakout_idxs) > allow_prior_breakouts:
            continue

        # defensive double pump avoid (if multiple breakouts detected)
        if len(breakout_idxs) > 1:
            continue

        # breakdown then pump -> detect if any breakout happens after first breakdown
        # (already covered by strict breakdown rule above, so not needed here)

        # Freshness check: ensure no bullish pump happens before inside body formed
        pre_inside_respected = False
        for i in range(n + 1, len(closes)):
            body_low_i = min(opens[i], closes[i])
            # if we find an inside body before any pump, stop freshness loop
            if body_low_i >= gap_bottom and body_low_i <= gap_top:
                break
            # pump before inside -> skip (pre-inside pump)
            if lows[i] <= mid and closes[i] > gap_top and closes[i] > opens[i]:
                pre_inside_respected = True
                break
        if pre_inside_respected:
            continue

        # find an "inside" candle based on the BOTTOM of the body being within the zone
        # (this allows examples like open=115, close=105 to be considered inside because body_low=105 ∈ [100,110])
        inside_idx: Optional[int] = None
        for i in range(n + 1, len(closes)):
            body_low = min(opens[i], closes[i])
            # require the bottom of the body to lie within the FVG zone
            if body_low >= gap_bottom and body_low <= gap_top:
                inside_idx = i
                break
        if inside_idx is None:
            continue

        # If there were breakouts before this inside, ensure they don't exceed allowed count
        breakouts_before_inside = [b for b in breakout_idxs if b < inside_idx]
        if len(breakouts_before_inside) > allow_prior_breakouts:
            continue

        # Fail after inside: if price later dropped below gap_bottom -> skip
        failed_after_inside = False
        for k in range(inside_idx, len(closes)):
            if (lows[k] < gap_bottom) if TREAT_WICK_AS_BREAKDOWN else (closes[k] < gap_bottom):
                failed_after_inside = True
                break
        if failed_after_inside:
            continue

        # respected after inside: allow any candle to touch mid first, 
        # then later any bullish candle to breakout above gap_top
        respected = False
        respect_idx: Optional[int] = None
        mid_touched = False

        for i in range(inside_idx + 1, len(closes)):
            # Step 1: check if mid level touched
            if lows[i] <= mid:
                mid_touched = True

            # Step 2: if mid was touched (earlier or this candle), check breakout
            if mid_touched and closes[i] > gap_top and closes[i] > opens[i]:
                respected = True
                respect_idx = i
                break

        # Now consider breakouts that happened AFTER inside
        breakouts_after_inside = [b for b in breakout_idxs if b >= inside_idx]
        # If there were breakouts after inside but no respect was found - skip (we don't accept random post-inside breakouts)
        if breakouts_after_inside and respect_idx is None:
            continue
        # If there are breakouts after inside and the first after-inside breakout is not the respect candle -> skip
        if breakouts_after_inside and respect_idx is not None:
            if breakouts_after_inside[0] != respect_idx:
                continue

        # double pump avoid after respect: now use FVG zone size % as threshold
        if respected and respect_idx is not None:
            double_after = False

            # zone size percentage (relative to gap_bottom)
            zone_pct = (gap_top - gap_bottom) / gap_bottom

            for j in range(respect_idx + 1, len(closes)):
                pump_pct = (closes[j] - gap_top) / gap_top
                if pump_pct >= zone_pct:   # যদি FVG zone size% এর বেশি pump করে
                    double_after = True
                    break

            if double_after:
                continue

        # Fail after respect: if any later low goes below gap_bottom -> invalidate respect
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
            if highs[j] >= entry * 1.05:
                win = True
                break
            if lows[j] < sl:
                win = False
                break
        results.append(win)

    win_rate = (sum(results) / len(results) * 100) if results else 0.0

    best_tp: Optional[float] = None
    best_rate = 0.0
    for tp in np.arange(0.01, 0.12, 0.01):
        tmp_results: List[bool] = []
        for idx, gb, gt in zones:
            entry = closes[idx]
            sl = gb
            win = False
            for j in range(idx + 1, min(n, idx + 8)):
                if highs[j] >= entry * (1 + tp):
                    win = True
                    break
                if lows[j] < sl:
                    win = False
                    break
            tmp_results.append(win)
        rate = (sum(tmp_results) / len(tmp_results) * 100) if tmp_results else 0.0
        if rate > best_rate:
            best_tp = float(tp)
            best_rate = float(rate)

    return {
        "win_rate_5pct": win_rate,
        "best_tp_pct": round(best_tp * 100, 2) if best_tp is not None else None,
        "best_tp_rate": round(best_rate, 2),
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

                # if current candle's low (or close, depending on config) is below gap_bottom, skip
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
    win_txt = ""
    win_data = c.get("win_rate_analysis") or {}
    wr = win_data.get("win_rate_5pct", None)
    tp = win_data.get("best_tp_pct", None)
    tprate = win_data.get("best_tp_rate", None)

    if wr is not None:
        win_txt = win_txt + f"\n🏆 *FVG Win Rate*: `{wr:.2f}%`"
    if tp is not None and tprate is not None:
        win_txt = win_txt + f"\n🎯 *Best TP*: `{tp:.2f}%` | Success Rate: `{tprate:.2f}%`"

    signals: List[str] = []
    if c.get("rsi_div"):
        signals.append("📈 RSI Divergence")
    if c.get("rs_val", 0) > 1:
        signals.append("💪 Relative Strength↑")
    if c.get("cvd_val", 0) > 0:
        signals.append("📊 CVD↑")
    if c.get("disp_candle"):
        signals.append("🔥 Displacement")

    if signals:
        extra_str = " | ".join(signals)
    else:
        extra_str = "—"

    avg_vol = c.get("avg_volume", 0)
    try:
        avg_vol_str = f"{int(avg_vol):,}"
    except Exception:
        avg_vol_str = str(avg_vol)

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
        f"\n🔖 *Status*: `{c.get('status','')}`"
        f"\n🚦 Signals: {extra_str}"
        f"{win_txt}"
        f"\n━━━━━━━━━━━━━━"
        f"\n✨ Trade wisely!"
    )
    return msg


# =========================================
# ------- Custom Command System -----------
# =========================================

def parse_tf_command(text: str) -> Optional[Tuple[str, str]]:
    # Regex case-insensitive, যাতে user 1m / 1M / 1h / 1d / 1w লিখলেও match হয়
    m = re.match(r"^([0-9]+m|[0-9]+h|[0-9]+d|[0-9]+w|[0-9]+M)\s+FVG\s+COIN\s*LIST$", 
                 text.strip(), re.IGNORECASE)
    if not m:
        return None

    tf_label = m.group(1)  # match capture (যেমন 1m, 1M, 4h ইত্যাদি)

    # smart mapping: 1M মানে Month, ছোট m মানে Minute
    if tf_label.endswith("M") and tf_label.isupper():
        tf = tf_label   # যেমন 1M → মাস
    else:
        tf = tf_label.lower()  # যেমন 1m → মিনিট, 1h → ঘন্টা, 1d → দিন, 1w → সপ্তাহ

    return tf_label.upper(), tf


# =========================================
# --------- Telegram Handler --------------
# =========================================

async def fvg_coinlist_handler(update: TGUpdate, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    tf_parse = parse_tf_command(text)
    if not tf_parse:
        await update.message.reply_text(
            "⚠️ Invalid command format!\n📝 Example: `4H FVG Coin list` or `15m FVG Coin list`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    tf_label, tf = tf_parse

    await update.message.reply_text(
        f"⏳ Scanning Binance ({tf_label}) FVG coins...",
        parse_mode=ParseMode.MARKDOWN,
    )

    coins = await get_fvg_coins_async(tf)

    if not coins:
        await update.message.reply_text(
            f"😔 No coins matched the criteria for {tf_label}. Try again later.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    def coin_sort_key(c: Dict[str, Any]) -> Tuple[int, float]:
        wr = (c.get("win_rate_analysis") or {}).get("win_rate_5pct", 0.0)
        respected = 1 if c.get("fvg_status") == "RESPECTED" else 0
        return respected, float(wr or 0.0)

    coins.sort(key=coin_sort_key, reverse=True)

    for c in coins:
        msg = format_coin_message(c, tf_label)
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Chart", url=tradingview_link(c.get("symbol", "")))]])
        try:
            await update.message.reply_text(
                msg + f"\n🔲 FVG Zone: `{c.get('fvg_zone_text','')}`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
            )
        except Exception:
            await update.message.reply_text(
                msg + f"\n🔲 FVG Zone: `{c.get('fvg_zone_text','')}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        await asyncio.sleep(0.3)
