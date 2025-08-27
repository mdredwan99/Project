import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Sequence
from datetime import datetime

from config_and_utils import logger, cooldown_ok
from data_api import (
    fetch_intervals, parse_ohlcv, get_klines,
    get_hist_win_rates
)

# =============== Global Strictness ===============
# True => শেষ (incomplete) ক্যান্ডেল বাদ দিয়ে শুধুই closed bar এ লজিক চালাবে
STRICT_CLOSE = True

# ===================== Numeric helpers & Indicators =====================
ArrayLike = Union[Sequence[float], np.ndarray]


def np_safe(arr: Optional[ArrayLike]) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    return np.array(arr, dtype=float)


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


# ---- Wilder RSI & ATR (stabler for trading logic) ----
def rsi_series(closes: ArrayLike, length: int = 14) -> np.ndarray:
    c = np_safe(closes)
    if c.size < length + 1:
        return np.full(c.size, 50.0)

    delta = np.diff(c)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    avg_up = np.mean(up[:length])
    avg_down = np.mean(down[:length])
    rs = (avg_up / avg_down) if avg_down > 0 else np.inf
    rsi_vals = [100 - 100 / (1 + rs)]

    for i in range(length, up.size):
        avg_up = (avg_up * (length - 1) + up[i]) / length
        avg_down = (avg_down * (length - 1) + down[i]) / length
        rs = (avg_up / avg_down) if avg_down > 0 else np.inf
        rsi_vals.append(100 - 100 / (1 + rs))

    pad = np.full(c.size - len(rsi_vals), 50.0)
    return np.concatenate([pad, np.array(rsi_vals, dtype=float)])


def atr_series(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    length: int = 14
) -> np.ndarray:
    high_arr = np_safe(highs)
    low_arr = np_safe(lows)
    close_arr = np_safe(closes)

    if close_arr.size < length + 1:
        return np.full(close_arr.size, 0.0)

    prev_close = np.concatenate(([close_arr[0]], close_arr[:-1]))
    tr = np.maximum(
        high_arr - low_arr,
        np.maximum(np.abs(high_arr - prev_close), np.abs(low_arr - prev_close))
    )

    atr = np.zeros_like(tr)
    first_atr = np.mean(tr[:length])
    if length - 1 > 0:
        atr[:length - 1] = first_atr
    atr[length - 1] = first_atr

    for t in range(length, tr.size):
        atr[t] = (atr[t - 1] * (length - 1) + tr[t]) / length
    return atr


def ema_series(values: ArrayLike, length: int) -> np.ndarray:
    values_np = np_safe(values)
    if values_np.size == 0:
        return values_np
    alpha = 2 / (length + 1)
    out = np.empty_like(values_np)
    out[0] = values_np[0]
    for i in range(1, values_np.size):
        out[i] = alpha * values_np[i] + (1 - alpha) * out[i - 1]
    return out


def sma_series(values: ArrayLike, length: int) -> np.ndarray:
    v = np_safe(values)
    if v.size < length:
        return np.full(v.size, safe_mean(v) if v.size else 0.0)
    kernel = np.ones(length) / length
    out = np.convolve(v, kernel, mode='valid')
    pad = np.full(v.size - out.size, out[0] if out.size else 0.0)
    return np.concatenate([pad, out])


def cvd_proxy(closes: ArrayLike, volumes: ArrayLike) -> np.ndarray:
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


def rel_volume(volumes: ArrayLike, lookback: int = 20) -> float:
    v = np_safe(volumes)
    if v.size < lookback + 1:
        return 1.0
    idx_last = -2 if (STRICT_CLOSE and v.size >= 2) else -1
    last = v[idx_last]
    avg = safe_mean(v[-lookback - 1:idx_last])
    return float(last / avg) if avg > 0 else 1.0


def sell_side_liquidity_sweep_bullish(
    highs: ArrayLike,
    lows: ArrayLike,
    opens: ArrayLike,
    closes: ArrayLike,
    lookback: int = 20
) -> bool:
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if lows_np.size < lookback + 2:
        return False
    idx = -2 if (STRICT_CLOSE and lows_np.size >= 2) else -1
    prior_low = float(np.min(lows_np[-(lookback + 1):-1]))
    sweep = (lows_np[idx] < prior_low) and (closes_np[idx] > prior_low) and (closes_np[idx] > opens_np[idx])
    return bool(sweep)


def displacement_bullish(
    highs: ArrayLike,
    lows: ArrayLike,
    opens: ArrayLike,
    closes: ArrayLike,
    atr: ArrayLike,
    body_ratio: float = 0.6,
    atr_mult: float = 1.2
) -> bool:
    high_arr = np_safe(highs)
    low_arr = np_safe(lows)
    open_arr = np_safe(opens)
    close_arr = np_safe(closes)
    atr_arr = np_safe(atr)

    if high_arr.size < 2 or atr_arr.size != high_arr.size:
        return False

    idx = -2 if (STRICT_CLOSE and high_arr.size >= 2) else -1
    rng = float(high_arr[idx] - low_arr[idx])
    body = float(close_arr[idx] - open_arr[idx])
    if rng <= 0:
        return False
    return bool(body > 0 and rng > atr_mult * float(atr_arr[idx]) and (body / rng) >= body_ratio)


def bullish_rsi_divergence(
    closes: ArrayLike,
    lows: ArrayLike,
    rsi: ArrayLike,
    sw_lookback: int = 3,
    lookback: int = 50
) -> bool:
    c = np_safe(closes)
    l_arr = np_safe(lows)
    r = np_safe(rsi)
    if c.size < lookback or l_arr.size != c.size or r.size != c.size:
        return False

    start = max(0, c.size - lookback)
    swings: List[int] = []
    for i in range(start + sw_lookback, c.size - sw_lookback):
        if l_arr[i] == np.min(l_arr[i - sw_lookback: i + sw_lookback + 1]):
            swings.append(i)
    if len(swings) < 2:
        return False

    i1, i2 = swings[-2], swings[-1]
    price_ll = l_arr[i2] < l_arr[i1]
    rsi_hl = r[i2] > r[i1]
    return bool(price_ll and rsi_hl)


def whale_entry(volumes: ArrayLike, closes: ArrayLike, factor: float = 3.0) -> bool:
    volumes_np = np_safe(volumes)
    closes_np = np_safe(closes)
    if volumes_np.size < 20 or closes_np.size < 2:
        return False
    idx = -2 if (STRICT_CLOSE and volumes_np.size >= 2) else -1
    last = volumes_np[idx]
    mean = float(np.mean(volumes_np[-21:idx])) if volumes_np.size >= 22 else float(np.mean(volumes_np))
    return bool((last > mean * factor) and (closes_np[idx] > closes_np[idx - 1]))


def cvd_imbalance_up(cvd: np.ndarray, bars: int = 5, mult: float = 1.6) -> bool:
    if cvd.size < bars + 1:
        return False
    idx = -2 if (STRICT_CLOSE and cvd.size >= 2) else -1
    slope = cvd[idx] - cvd[idx - bars]
    ref_seg = np.diff(cvd[:idx - bars]) if (idx - bars) > 10 else np.diff(cvd[:idx]) if idx > 2 else np.diff(cvd)
    ref_std = safe_std(ref_seg)
    if ref_std == 0:
        return bool(slope > 0)
    return bool(slope > mult * ref_std)


def volatility_metric(closes: ArrayLike, win: int = 30) -> float:
    closes_np = np_safe(closes)
    if closes_np.size < win:
        return 0.0
    end = -2 if (STRICT_CLOSE and closes_np.size >= 2) else -1
    seg = closes_np[end - win + 1:end + 1]
    mu = float(np.mean(seg))
    return float(np.std(seg) / mu) if mu else 0.0


# --- Bands & Squeeze ---
def bbands(values: ArrayLike, length: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np_safe(values)
    ma = sma_series(values, length)
    std = np.zeros_like(v)
    if v.size >= length:
        for i in range(length - 1, v.size):
            std[i] = np.std(v[i - length + 1:i + 1])
    upper = ma + mult * std
    lower = ma - mult * std
    width = (upper - lower) / ma.clip(min=1e-9)
    return lower, ma, upper, width


def keltner_channels(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    length: int = 20,
    mult: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema = ema_series(closes, length)
    atr = atr_series(highs, lows, closes, 20)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return lower, ema, upper


def is_bb_inside_kc(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5
) -> bool:
    bb_l, _, bb_u, _ = bbands(closes, length, bb_mult)
    kc_l, _, kc_u = keltner_channels(highs, lows, closes, length, kc_mult)
    if len(bb_l) == 0:
        return False
    idx = -2 if (STRICT_CLOSE and len(np_safe(closes)) >= 2) else -1
    return bool(bb_u[idx] < kc_u[idx] and bb_l[idx] > kc_l[idx])


# --- FVG core ---
def find_bullish_fvg_indices(highs: ArrayLike, lows: ArrayLike) -> List[int]:
    high_arr = np_safe(highs)
    low_arr = np_safe(lows)
    out: List[int] = []
    for n in range(2, high_arr.size):
        if low_arr[n] > high_arr[n - 2]:
            out.append(n)
    return out


def last_fvg_zone(highs: ArrayLike, lows: ArrayLike) -> Optional[Tuple[float, float, int]]:
    idxs = find_bullish_fvg_indices(highs, lows)
    if not idxs:
        return None
    n = idxs[-1]
    low_arr = np_safe(lows)
    high_arr = np_safe(highs)
    gap_top = float(low_arr[n])        # top of gap (upper bound of FVG box)
    gap_bottom = float(high_arr[n - 2])  # bottom of gap (lower bound)
    return gap_top, gap_bottom, int(n)


def bullish_fvg_status(
    opens: ArrayLike,
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    invalid_mult: float = 1.0,
    recent_margin: int = 2
) -> Optional[Tuple[str, float, float, int, int]]:
    """
    Returns:
      None | ('RESPECTED'/'JUST_ENTERED'/'INSIDE', gap_bottom, gap_top, inside_idx, fvg_idx)
    Strict close + filled-avoid + 1x invalidation guarded.
    """
    o = np_safe(opens)
    h = np_safe(highs)
    l_arr = np_safe(lows)
    c = np_safe(closes)

    zone = last_fvg_zone(h.tolist(), l_arr.tolist())
    if not zone:
        return None

    gap_top, gap_bottom, idx_fvg = zone
    size = gap_top - gap_bottom

    end_idx = (c.size - 2) if (STRICT_CLOSE and c.size >= 2) else (c.size - 1)
    if end_idx <= idx_fvg + 2:
        return None

    # already filled once? avoid
    for j in range(idx_fvg + 1, end_idx + 1):
        if (l_arr[j] <= gap_bottom) and (h[j] >= gap_top):
            return None

    # 1x invalidation
    if c[end_idx] > (gap_top + invalid_mult * size):
        return None

    inside_idx: Optional[int] = None
    for i in range(idx_fvg + 1, end_idx + 1):
        body_low = float(min(o[i], c[i]))
        body_high = float(max(o[i], c[i]))
        if (gap_bottom <= body_low) and (body_high <= gap_top):
            inside_idx = i
            break
    if inside_idx is None:
        return None

    # RESPECTED: later a candle CLOSE > gap_top and bullish body
    for j in range(inside_idx + 1, end_idx + 1):
        if c[j] > gap_top and c[j] > o[j]:
            return "RESPECTED", gap_bottom, gap_top, inside_idx, idx_fvg

    # JUST_ENTERED vs INSIDE (based on recency)
    if gap_bottom <= c[end_idx] <= gap_top:
        if (end_idx - inside_idx) <= recent_margin:
            return "JUST_ENTERED", gap_bottom, gap_top, inside_idx, idx_fvg
        return "INSIDE", gap_bottom, gap_top, inside_idx, idx_fvg

    return None


def bullish_fvg_alert_logic(
    opens: ArrayLike,
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    volumes: ArrayLike,
    tf_label: str,
    invalid_mult: float = 1.0
) -> Optional[str]:
    """
    Returns text if FVG reclaim truly confirmed (strict close), else None.
    """
    o = np_safe(opens)
    h = np_safe(highs)
    l_arr = np_safe(lows)
    c = np_safe(closes)
    _ = np_safe(volumes)  # volumes not used in core confirm but kept for signature compatibility

    if c.size < 60:
        return None

    status = bullish_fvg_status(o, h, l_arr, c, invalid_mult=invalid_mult)
    if not status:
        return None
    stat, _, _, _, _ = status

    # require RESPECTED
    if stat != "RESPECTED":
        return None

    return f"Bullish FVG Confirmed ({tf_label})"


# ===================== Risk filters & HTF =====================
def htf_bullish_bias(closes_1h: ArrayLike, closes_4h: ArrayLike) -> bool:
    c1h = np_safe(closes_1h)
    if c1h.size < 60:
        return False
    ema1h = ema_series(c1h, 50)
    idx = -2 if (STRICT_CLOSE and c1h.size >= 2) else -1
    cond_1h = (c1h[idx] > ema1h[idx]) and (ema1h[idx] > ema1h[idx - 5])

    cond_4h = False
    c4h = np_safe(closes_4h)
    if c4h.size >= 60:
        ema4h = ema_series(c4h, 50)
        idx4 = -2 if (STRICT_CLOSE and c4h.size >= 2) else -1
        cond_4h = (c4h[idx4] > ema4h[idx4]) and (ema4h[idx4] > ema4h[idx4 - 5])
    return bool(cond_1h or cond_4h)


def atr_vol_gate(highs_15: ArrayLike, lows_15: ArrayLike, closes_15: ArrayLike) -> bool:
    atr15 = atr_series(highs_15, lows_15, closes_15, 14)
    atr_slice = atr15[-100:] if atr15.size >= 100 else atr15
    if atr_slice.size < 20:
        return False
    idx = -2 if (STRICT_CLOSE and atr15.size >= 2) else -1
    cur_atr = float(atr15[idx])
    p40 = percentile(atr_slice, 40)
    return bool(cur_atr >= p40 and cur_atr > 0)


# ===================== Helper: structure/MSS/ChoCH =====================
def swing_high_idx(highs: ArrayLike, lookback: int = 5) -> Optional[int]:
    h = np_safe(highs)
    if h.size < 2 * lookback + 1:
        return None
    i = h.size - 1 - lookback
    window = h[i - lookback:i + lookback + 1]
    return i if h[i] == np.max(window) else None


def swing_low_idx(lows: ArrayLike, lookback: int = 5) -> Optional[int]:
    lows_np = np_safe(lows)
    if lows_np.size < 2 * lookback + 1:
        return None
    i = lows_np.size - 1 - lookback
    window = lows_np[i - lookback:i + lookback + 1]
    return i if lows_np[i] == np.min(window) else None


def broke_above_previous_swing_high(
    closes: ArrayLike,
    highs: ArrayLike,
    lookback: int = 5,
    bars_ahead: int = 60
) -> bool:
    """
    Returns True if the latest (closed) price is above any prominent swing-high inside last `bars_ahead` bars.
    """
    h = np_safe(highs)
    c = np_safe(closes)
    if h.size < lookback * 2 + 2 or c.size < lookback * 2 + 2:
        return False
    check_idx = -2 if (STRICT_CLOSE and c.size >= 2) else -1
    start = max(lookback + 1, h.size - bars_ahead)
    for i in range(start, h.size - lookback - 1):
        if h[i] == np.max(h[i - lookback: i + lookback + 1]):
            if c[check_idx] > h[i]:
                return True
    return False


# ===================== Profiles =====================
def profile_htf_sweep_mss_fvg(
    o1h: ArrayLike, h1h: ArrayLike, l1h: ArrayLike, c1h: ArrayLike, v1h: ArrayLike,
    o4h: ArrayLike, h4h: ArrayLike, l4h: ArrayLike, c4h: ArrayLike, v4h: ArrayLike,
    o15: ArrayLike, h15: ArrayLike, l15: ArrayLike, c15: ArrayLike, v15: ArrayLike
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    sweep_1h = sell_side_liquidity_sweep_bullish(h1h, l1h, o1h, c1h, lookback=20)
    c4h_np = np_safe(c4h)
    if c4h_np.size > 0:
        sweep_4h = sell_side_liquidity_sweep_bullish(h4h, l4h, o4h, c4h, lookback=20)
    else:
        sweep_4h = False
    if sweep_1h or sweep_4h:
        reasons.append("HTF Sell-side Sweep")

    atr15 = atr_series(h15, l15, c15, 14)
    disp_15 = displacement_bullish(h15, l15, o15, c15, atr15, 0.55, 1.15)
    choch_1h = broke_above_previous_swing_high(c1h, h1h, lookback=5, bars_ahead=80)
    if disp_15 or choch_1h:
        reasons.append("MSS/ChoCH + Displacement")

    fvg_15 = bullish_fvg_alert_logic(o15, h15, l15, c15, v15, "15M")
    fvg_1h = bullish_fvg_alert_logic(o1h, h1h, l1h, c1h, v1h, "1H")
    if fvg_15 or fvg_1h:
        reasons.append("FVG Reclaim")

    cvd15 = cvd_proxy(c15, v15)
    cvd_ok = cvd_imbalance_up(cvd15, bars=5, mult=1.5)
    relv_ok = rel_volume(v15, 20) >= 1.5
    if cvd_ok and relv_ok:
        reasons.append("CVD Ramp + RelVol↑")

    ok = len(reasons) >= 3
    return ok, reasons


def profile_bullish_fvg_htf(
    o1h: ArrayLike, h1h: ArrayLike, l1h: ArrayLike, c1h: ArrayLike, v1h: ArrayLike,
    o4h: ArrayLike, h4h: ArrayLike, l4h: ArrayLike, c4h: ArrayLike, v4h: ArrayLike,
    o1d: ArrayLike, h1d: ArrayLike, l1d: ArrayLike, c1d: ArrayLike, v1d: ArrayLike
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    for tf_name, (o, h, lows, c, v) in [
        ("1D", (o1d, h1d, l1d, c1d, v1d)),
        ("4H", (o4h, h4h, l4h, c4h, v4h))
    ]:
        c_np = np_safe(c)
        if c_np.size == 0:
            continue
        alert = bullish_fvg_alert_logic(o, h, lows, c, v, tf_name)
        if alert:
            reasons.append(f"{alert}")
    ok = len(reasons) >= 1
    return ok, reasons


def profile_squeeze_expansion(
    o1h: ArrayLike, h1h: ArrayLike, l1h: ArrayLike, c1h: ArrayLike, v1h: ArrayLike
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    squeeze = is_bb_inside_kc(h1h, l1h, c1h, length=20, bb_mult=2.0, kc_mult=1.5)
    _, _, _, bb_width = bbands(c1h, 20, 2.0)

    c1h_np = np_safe(c1h)
    width_p20_ok = False
    if bb_width.size > 30:
        p20 = percentile(bb_width[-30:], 20)
        idx = -2 if (STRICT_CLOSE and c1h_np.size >= 2) else -1
        width_p20_ok = bb_width[idx] <= p20

    if squeeze and width_p20_ok:
        bb_l, _, bb_u, _ = bbands(c1h, 20, 2.0)
        idx = -2 if (STRICT_CLOSE and c1h_np.size >= 2) else -1
        relv = rel_volume(v1h, 20)
        if c1h_np.size > 0 and bb_u.size > 0 and (c1h_np[idx] > bb_u[idx]) and relv >= 1.5:
            cvd1h = cvd_proxy(c1h, v1h)
            if cvd_imbalance_up(np_safe(cvd1h), bars=3, mult=1.2):
                reasons.append("Squeeze→Expansion (1H)")

    ok = len(reasons) >= 1
    return ok, reasons


def profile_rs_breakout_vs_btc(
    o15: ArrayLike, h15: ArrayLike, l15: ArrayLike, c15: ArrayLike,
    c1h_coin: ArrayLike, c1h_btc: ArrayLike, h1h: ArrayLike
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    coin = np_safe(c1h_coin)
    btc = np_safe(c1h_btc)
    if coin.size < 25 or btc.size < 25:
        return False, reasons
    rs = coin / np.clip(btc, 1e-9, None)
    rs_hh = bool(rs[-1] >= np.max(rs[-20:]))
    if rs_hh:
        bos_1h = broke_above_previous_swing_high(c1h_coin, h1h, 5, bars_ahead=80)
        disp_15 = False
        h15_np = np_safe(h15)
        c15_np = np_safe(c15)
        l15_np = np_safe(l15)
        if h15_np.size == c15_np.size and h15_np.size > 0:
            atr_15 = atr_series(h15_np, l15_np, c15_np, 14)
            disp_15 = displacement_bullish(h15_np, l15_np, o15, c15_np, atr_15, 0.5, 1.1)
        if bos_1h or disp_15:
            reasons.append("RS Breakout vs BTC (1H)")
    ok = len(reasons) >= 1
    return ok, reasons


def profile_weekly_orb(
    o1h: ArrayLike, h1h: ArrayLike, l1h: ArrayLike, c1h: ArrayLike, v1h: ArrayLike, now_utc: datetime
) -> Tuple[bool, List[str]]:
    # Placeholder (intentionally minimal)
    return False, []


# ===================== 15m REFINEMENT =====================
def ltf_refine_15m(
    o15: ArrayLike, h15: ArrayLike, l15: ArrayLike, c15: ArrayLike, v15: ArrayLike
) -> Tuple[bool, List[str]]:
    """
    Require at least 2 of 4:
      1) 15m BOS (broke above a recent swing-high)
      2) 15m Displacement (ATR-based big bullish candle)
      3) 15m Bullish FVG Reclaim (strict confirmed)
      4) CVD Ramp + Relative Volume >= 1.4
    """
    reasons: List[str] = []

    if broke_above_previous_swing_high(c15, h15, lookback=4, bars_ahead=70):
        reasons.append("15m BOS")

    atr15 = atr_series(h15, l15, c15, 14)
    if displacement_bullish(h15, l15, o15, c15, atr15, body_ratio=0.55, atr_mult=1.10):
        reasons.append("15m Displacement")

    if bullish_fvg_alert_logic(o15, h15, l15, c15, v15, "15M"):
        reasons.append("15m FVG Reclaim")

    cvd15 = cvd_proxy(c15, v15)
    if cvd_imbalance_up(cvd15, bars=5, mult=1.4) and rel_volume(v15, 20) >= 1.4:
        reasons.append("15m CVD Ramp + RelVol↑")

    ok = len(reasons) >= 2
    return ok, reasons


# ===================== Opinion helper =====================
def opinion_from_hist_win_rates(hw_rates: Dict[str, float]) -> Tuple[str, float]:
    w = {"15m": 0.4, "1h": 0.35, "4h": 0.25}
    num = sum(hw_rates.get(tf, 0.0) * w[tf] for tf in w)
    den = sum(w.values())
    agg = num / den if den else 0.0
    if agg >= 70:
        verdict = "🔥 Strong"
    elif agg >= 55:
        verdict = "✅ Moderate"
    elif agg > 0:
        verdict = "🟡 Weak"
    else:
        verdict = "⚪ No edge"
    return verdict, agg


# ===================== Volatility list (1H) =====================
async def volatility_coin_list(get_exchange_info_func, get_klines_func, top_n: int = 20) -> List[Tuple[str, float]]:
    exinfo = await get_exchange_info_func()
    symbols = [
        s["symbol"] for s in exinfo.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    ]

    async def vol_for(sym: str) -> Tuple[str, float]:
        try:
            k = await get_klines_func(sym, "1h", 180)
            _, _, _, c, _, _ = parse_ohlcv(k)
            v = volatility_metric(c, win=60)
            return sym, v
        except Exception:
            return sym, 0.0

    import asyncio
    res = await asyncio.gather(*[vol_for(s) for s in symbols])
    res.sort(key=lambda x: x[1], reverse=True)
    return res[:max(1, top_n)]


# ===================== Detector (MTF + 15m refine) =====================
async def detect_signals(
    symbol: str,
    btc_1h_cache: Optional[Dict[str, List[float]]] = None
) -> Tuple[Dict[str, List[str]], Optional[str], Dict[str, float], List[str]]:
    limits = {"15m": 240, "1h": 240, "4h": 240, "1d": 240}
    data_map = await fetch_intervals(symbol, limits)
    if any(len(data_map.get(iv, [])) == 0 for iv in ["15m", "1h"]):
        return {}, None, {}, []

    # OHLCV
    o15, h15, l15, c15, v15, t15 = parse_ohlcv(data_map["15m"])
    o1h, h1h, l1h, c1h, v1h, t1h = parse_ohlcv(data_map["1h"])
    o4h, h4h, l4h, c4h, v4h, t4h = parse_ohlcv(data_map.get("4h", []))
    o1d, h1d, l1d, c1d, v1d, t1d = parse_ohlcv(data_map.get("1d", []))

    # Global gates (15m ATR gate + HTF bias)
    if not c1h or not c15:
        return {}, None, {}, []
    if not atr_vol_gate(h15, l15, c15):
        return {}, None, {}, []
    if not htf_bullish_bias(c1h, c4h):
        return {}, None, {}, []

    # Historical WR (edge gate)
    hw_rates = await get_hist_win_rates(symbol)
    verdict, agg = opinion_from_hist_win_rates(hw_rates)
    if agg < 1.0:
        return {}, None, hw_rates, []

    # HTF profiles
    profiles_triggered: List[str] = []
    reasons_by_tf: Dict[str, List[str]] = {"15m": [], "1h": [], "4h": [], "1d": []}

    ok1, r1 = profile_htf_sweep_mss_fvg(
        o1h, h1h, l1h, c1h, v1h,
        o4h, h4h, l4h, c4h, v4h,
        o15, h15, l15, c15, v15
    )
    if ok1:
        profiles_triggered.append("HTF Sweep→MSS→FVG+CVD")
        reasons_by_tf["1h"].extend(r1)

    ok2, r2 = profile_bullish_fvg_htf(
        o1h, h1h, l1h, c1h, v1h,
        o4h, h4h, l4h, c4h, v4h,
        o1d, h1d, l1d, c1d, v1d
    )
    if ok2:
        profiles_triggered.append("HTF Bullish FVG Reclaim")
        for item in r2:
            if "1D" in item:
                reasons_by_tf["1d"].append(item)
            elif "4H" in item:
                reasons_by_tf["4h"].append(item)
            else:
                reasons_by_tf["1h"].append(item)

    ok3, r3 = profile_squeeze_expansion(o1h, h1h, l1h, c1h, v1h)
    if ok3:
        profiles_triggered.append("Squeeze→Expansion (1H)")
        reasons_by_tf["1h"].extend(r3)

    # RS vs BTC
    c1h_btc: List[float] = []
    if btc_1h_cache and "BTCUSDT" in btc_1h_cache:
        c1h_btc = btc_1h_cache["BTCUSDT"]
    else:
        btc_1h = await get_klines("BTCUSDT", "1h", 240)
        _, _, _, c_btc, _, _ = parse_ohlcv(btc_1h)
        c1h_btc = c_btc
        if btc_1h_cache is not None:
            btc_1h_cache["BTCUSDT"] = c1h_btc

    ok4, r4 = profile_rs_breakout_vs_btc(o15, h15, l15, c15, c1h, c1h_btc, h1h)
    if ok4:
        profiles_triggered.append("RS Breakout vs BTC")
        reasons_by_tf["1h"].extend(r4)

    # ---------------- 15m Refinement (MANDATORY) ----------------
    refine_ok, refine_reasons = ltf_refine_15m(o15, h15, l15, c15, v15)
    if not refine_ok:
        return {}, None, hw_rates, []
    reasons_by_tf["15m"].extend(refine_reasons)

    # Extra 15m enrichment (optional tags)
    if cvd_imbalance_up(cvd_proxy(c15, v15), bars=5, mult=1.6):
        reasons_by_tf["15m"].append("CVD Imbalance Up")
    if whale_entry(v15, c15, factor=3.0):
        reasons_by_tf["15m"].append("Whale Entry")
    if sell_side_liquidity_sweep_bullish(h15, l15, o15, c15, 20):
        reasons_by_tf["15m"].append("SSL Sweep")
    if displacement_bullish(h15, l15, o15, c15, atr_series(h15, l15, c15, 14), 0.6, 1.2):
        reasons_by_tf["15m"].append("Smart Money Entry (Displacement)")
    if bullish_rsi_divergence(c15, l15, rsi_series(c15, 14), 3, 20):
        reasons_by_tf["15m"].append("RSI Bullish Div")
    if volatility_metric(c15, 30) > 0.5:
        reasons_by_tf["15m"].append("High Volatility")

    # HTF FVG confirm scan again (1h/4h/1d) – adds confidence tags
    for tf, candles in [
        ("1h", data_map.get("1h", [])),
        ("4h", data_map.get("4h", [])),
        ("1d", data_map.get("1d", []))
    ]:
        if not candles:
            continue
        o, h, lows, c, v, _ = parse_ohlcv(candles)
        alert = bullish_fvg_alert_logic(o, h, lows, c, v, tf.upper())
        if alert:
            reasons_by_tf[tf].append("Bullish FVG Confirmed")

    # Final gate (conservative)
    gate = (
        (len(reasons_by_tf["15m"]) >= 3) or
        (len(reasons_by_tf["1h"]) >= 2) or
        any("Bullish FVG Confirmed" in x for x in reasons_by_tf["1h"] + reasons_by_tf["4h"] + reasons_by_tf["1d"])
    )
    if not gate or not cooldown_ok(symbol):
        return {}, None, {}, []

    final_text = (
        f"{opinion_from_hist_win_rates(hw_rates)[0]} "
        f"(Hist-Win Rate: {round(opinion_from_hist_win_rates(hw_rates)[1]):.0f}%). "
        f"15m: {round(hw_rates.get('15m', 0)):.0f}% | "
        f"1h: {round(hw_rates.get('1h', 0)):.0f}% | "
        f"4h: {round(hw_rates.get('4h', 0)):.0f}%"
    )
    return reasons_by_tf, final_text, hw_rates, profiles_triggered
