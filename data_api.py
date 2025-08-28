# -*- CryptoICT AI -*-

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config_and_utils import (
    logger,
    BINANCE_BASE,
    MAX_CONCURRENCY,
    REQUEST_TIMEOUT,
    supabase,
    binance_connector_client,
    RateLimitError,
)

# Globals managed by main.post_init()
aiohttp_session: Optional[aiohttp.ClientSession] = None
binance_sem: Optional[asyncio.Semaphore] = None

log = logging.getLogger("data_api")

# ===================== Supabase async wrappers =====================
async def _to_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)

async def get_column(table: str, column: str = "coin") -> List[str]:
    try:
        def query():
            res = supabase.table(table).select(column).execute()
            return res.data or []
        rows = await _to_thread(query)
        return [r[column].upper() for r in rows if r.get(column)]
    except Exception:
        log.exception("Supabase get_column error")
        return []

async def add_coin(table: str, coin: str) -> None:
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up}).execute()
        await _to_thread(insert)
        log.info("Added %s to %s", coin_up, table)
    except Exception:
        log.exception("Supabase add_coin error")

async def add_coin_with_date(table: str, coin: str) -> None:
    from datetime import datetime, timezone
    coin_up = coin.upper()
    existing = await get_column(table)
    if coin_up in existing:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        def insert():
            return supabase.table(table).insert({"coin": coin_up, "timestamp": timestamp}).execute()
        await _to_thread(insert)
        log.info("Added %s to %s with timestamp", coin_up, table)
    except Exception:
        log.exception("Supabase add_coin_with_date error")

async def remove_coin_from_table(table: str, coin: str) -> bool:
    coin_up = coin.upper()
    try:
        def delete_fn():
            resp = supabase.table(table).select("id, coin").eq("coin", coin_up).execute()
            rows = resp.data or []
            for r in rows:
                supabase.table(table).delete().eq("id", r["id"]).execute()
            return bool(rows)
        return await _to_thread(delete_fn)
    except Exception:
        log.exception("Supabase remove_coin_from_table error")
        return False

async def get_removed_map() -> Dict[str, str]:
    try:
        def query():
            res = supabase.table("removed").select("coin, timestamp").execute()
            return res.data or []
        rows = await _to_thread(query)
        return {r["coin"].upper(): r.get("timestamp") for r in rows if r.get("coin")}
    except Exception:
        log.exception("Supabase get_removed_map error")
        return {}

async def log_to_supabase(symbol: str, reasons_text: str, opinion_text: Optional[str]) -> None:
    from datetime import datetime, timezone
    coin = symbol.upper()
    ts_iso = datetime.now(timezone.utc).isoformat()
    payload = {"coin": coin, "timestamp": ts_iso, "reasons": reasons_text}
    if opinion_text is not None:
        payload["opinion"] = opinion_text
    try:
        def insert():
            return supabase.table("signals").insert(payload).execute()
        await _to_thread(insert)
        log.info("Logged to supabase: %s | %s", coin, reasons_text)
    except Exception:
        log.exception("Supabase log error for %s", coin)

async def fetch_signals_since(since_iso: str) -> List[Dict[str, Any]]:
    try:
        def query():
            return supabase.table("signals").select("*").gte("timestamp", since_iso).execute().data or []
        return await _to_thread(query)
    except Exception:
        log.exception("Supabase fetch_signals_since error")
        return []

# ===================== Binance REST =====================
async def binance_request(path: str, params: Optional[dict] = None) -> Tuple[Any, Optional[str]]:
    global aiohttp_session
    if aiohttp_session is None:
        raise RuntimeError("HTTP session is not initialized")
    url = f"{BINANCE_BASE}{path}"
    params = params or {}
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

async def get_exchange_info(filtering_on=True):
    data, used = await binance_request("/api/v3/exchangeInfo")
    if used:
        log.debug("ExchangeInfo weight: %s", used)
    if not filtering_on:
        return data
    symbols = [
        s for s in data.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]
    data["symbols"] = symbols
    return data

async def get_klines(symbol: str, interval: str, limit: int = 150):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data, used = await binance_request("/api/v3/klines", params=params)
    return data

async def get_ticker_24h(symbol: str) -> Optional[float]:
    try:
        data, _ = await binance_request("/api/v3/ticker/24hr", {"symbol": symbol})
        pct = float(data.get("priceChangePercent", 0.0))
        return pct
    except Exception:
        log.exception("24h ticker error for %s", symbol)
        return None

async def get_top_gainers(n=10, filtering_on=True) -> List[Tuple[str, float]]:
    exinfo = await get_exchange_info(filtering_on=filtering_on)
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and (filtering_on is False or not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"]))
    ]
    global binance_sem
    if binance_sem is None:
        binance_sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def get_pct(symbol):
        try:
            if binance_sem is None:
                raise RuntimeError("binance_sem not initialized")
            sem = binance_sem
            async with sem:
                pct = await get_ticker_24h(symbol)
                # small throttle between requests to be polite
                await asyncio.sleep(0.25)
            return (symbol, pct)
        except Exception:
            return (symbol, None)

    tasks = [get_pct(sym) for sym in symbols]
    all_results = await asyncio.gather(*tasks)
    filtered = [(sym, pct) for sym, pct in all_results if pct is not None]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]

# ===================== Parse & Intervals =====================
def parse_ohlcv(candles: list):
    if not candles:
        return [], [], [], [], [], []
    opens = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[3]) for c in candles]
    closes= [float(c[4]) for c in candles]
    volumes=[float(c[5]) for c in candles]
    times = [int(c[0]) for c in candles]
    return opens, highs, lows, closes, volumes, times

INTERVALS_CORE = ["15m", "1h", "4h"]
INTERVALS_FVG  = ["1h", "4h", "1d"]

async def fetch_intervals(symbol: str, limits: Dict[str, int]):
    async def _one(iv: str, lim: int):
        try:
            data = await get_klines(symbol, iv, lim)
            return iv, data if isinstance(data, list) else []
        except Exception as e:
            log.warning("Klines error %s %s: %s", symbol, iv, e)
            return iv, []
    tfs = set(INTERVALS_CORE + INTERVALS_FVG + ["1d"])
    tasks = [_one(iv, limits.get(iv, 240)) for iv in tfs]
    res = await asyncio.gather(*tasks)
    return {iv: data if isinstance(data, list) else [] for iv, data in res}

# ===================== Historical Win Rate =====================
def get_binance_klines_official(symbol: str, interval: str, start_time: int, end_time: int, max_per_req=1000) -> List[list]:
    all_klines = []
    cur_start = start_time
    while cur_start < end_time:
        klines = binance_connector_client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=cur_start,
            endTime=end_time,
            limit=max_per_req,
        )
        if not klines:
            break
        all_klines.extend(klines)
        last_time = klines[-1][0]
        if last_time == cur_start:
            break
        cur_start = last_time + 1
        if len(klines) < max_per_req:
            break
    return all_klines

def calc_hist_win_rate(symbol: str, interval: str, pump_pct=5.0, years: int = 5) -> Tuple[float, int]:
    """
    Calculates historical FVG win rate for the given symbol, interval, and TP%.
    Win = price pumps 'pump_pct'% above entry within 6 bars, SL = FVG bottom.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    ms_per_bar = {'15m': 15*60*1000, '1h': 60*60*1000, '4h': 4*60*60*1000, '1d': 24*60*60*1000}[interval]
    bars = int((years * 365 * 24 * 60 * 60 * 1000) / ms_per_bar)
    end_time = int(now.timestamp() * 1000)
    start_time = end_time - int(1.05 * bars * ms_per_bar)
    candles = get_binance_klines_official(symbol, interval, start_time, end_time)
    if len(candles) < 100:
        return 0.0, 0
    opens, highs, lows, closes, volumes, times = parse_ohlcv(candles)
    wins, total = 0, 0
    lookahead = 6
    for i in range(len(closes)-lookahead-1):
        entry = closes[i]
        target = entry * (1 + pump_pct/100)
        sl = lows[i]  # SL as FVG bottom
        # Check if target hit before SL
        win = False
        for j in range(i+1, i+1+lookahead):
            if highs[j] >= target:  # TP hit
                win = True
                break
            if lows[j] < sl:  # SL hit
                win = False
                break
        if win:
            wins += 1
        total += 1
    win_rate = (wins/total*100) if total > 0 else 0.0
    return win_rate, total

async def get_hist_win_rates(symbol: str) -> Dict[str, float]:
    loop = asyncio.get_event_loop()
    out = {}
    for tf in ['15m','1h','4h']:
        wr, _ = await loop.run_in_executor(None, lambda: calc_hist_win_rate(symbol, tf))
        out[tf] = wr
    return out

# ===================== Market Cap =====================
async def get_market_cap(symbol: str) -> Optional[float]:
    """
    Get market cap for a given coin symbol using CoinGecko (fallback).
    Returns value in USD if available.
    """
    global aiohttp_session
    if aiohttp_session is None:
        raise RuntimeError("HTTP session is not initialized")

    try:
        # Map Binance symbol (e.g. BTCUSDT -> bitcoin)
        symbol_lower = symbol.replace("USDT", "").lower()

        url = f"https://api.coingecko.com/api/v3/coins/{symbol_lower}"
        # Use session timeout already configured in ClientSession
        async with aiohttp_session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            market_cap = data.get("market_data", {}).get("market_cap", {}).get("usd")
            if market_cap is not None:
                return float(market_cap)
    except Exception as e:
        log.warning("get_market_cap error for %s: %s", symbol, e)
        return None

    return None
