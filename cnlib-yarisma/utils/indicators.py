"""
Teknik indicator'lar — pandas-ta yerine manuel implementation.

Tüm fonksiyonlar lookahead-bias'siz (sadece geçmiş veriyi kullanır),
giriş olarak pandas Series (Close/High/Low) alır, Series döner.

Wilder's smoothing (EMA with alpha=1/period) RSI ve ATR'de kullanılır —
standart tanım.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).

    0 -> aşırı satım, 100 -> aşırı alım.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD = EMA(fast) - EMA(slow), Signal = EMA(MACD, signal_period).

    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    close: pd.Series,
    period: int = 20,
    stddev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns: (middle, upper, lower, percent_b)
    percent_b = (close - lower) / (upper - lower), 0-1 genelde.
    """
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + stddev * std
    lower = mid - stddev * std
    band_width = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / band_width
    return mid, upper, lower, pct_b


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """True Range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index — trend gücü (sinyal YÖN vermez, güç).

    ADX > 25 genelde güçlü trend anlamına gelir.
    Returns: (adx, plus_di, minus_di)
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = true_range(high, low, close)
    # Wilder smoothing
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm_s / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_, plus_di, minus_di
