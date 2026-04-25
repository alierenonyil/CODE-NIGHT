"""
Feature engineering — ML stratejisi için.

Tüm feature'lar sadece geçmiş veriyi kullanır (lookahead yok).
Rolling hesaplar ilk N candle'da NaN döner → training sırasında drop.

Fonksiyon: build_features(df) → DataFrame
  - Giriş: OHLCV DataFrame
  - Çıkış: feature DataFrame (aynı index, ilk ~50 satır NaN)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rsi, macd, bollinger, adx, atr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame'den feature seti üretir.

    Feature listesi (~20 feature):
      - Past returns: r1, r5, r10, r20
      - Price / SMA ratios: p_over_sma5, 10, 20, 50
      - RSI, MACD hist, Bollinger pct_b
      - ADX, +DI, -DI
      - ATR pct, range pct, close position in range
      - Volume ratio
      - Return volatility (10, 20 gün std)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feats = pd.DataFrame(index=df.index)

    # 1) Past returns
    feats["r1"] = close.pct_change(1)
    feats["r5"] = close.pct_change(5)
    feats["r10"] = close.pct_change(10)
    feats["r20"] = close.pct_change(20)

    # 2) Price / SMA ratios (1.0 = eşit, >1 trend yukarı)
    for period in (5, 10, 20, 50):
        sma = close.rolling(period, min_periods=period).mean()
        feats[f"p_over_sma{period}"] = (close / sma) - 1.0

    # 3) RSI
    feats["rsi14"] = rsi(close, 14)

    # 4) MACD histogram (normalize close'a göre)
    _, _, hist = macd(close, 12, 26, 9)
    feats["macd_hist_norm"] = hist / close

    # 5) Bollinger pct_b (20/2)
    _, _, _, pct_b = bollinger(close, 20, 2.0)
    feats["bb_pct_b"] = pct_b

    # 6) ADX
    adx_, pdi, mdi = adx(high, low, close, 14)
    feats["adx14"] = adx_
    feats["plus_di"] = pdi
    feats["minus_di"] = mdi
    feats["di_spread"] = pdi - mdi

    # 7) ATR pct
    atr_ = atr(high, low, close, 14)
    feats["atr_pct"] = atr_ / close

    # 8) Günlük range pct
    feats["range_pct"] = (high - low) / close

    # 9) Close position in range (0=low, 1=high)
    daily_range = (high - low).replace(0, np.nan)
    feats["close_pos"] = (close - low) / daily_range

    # 10) Volume ratio (son volume / 20-günlük ortalama)
    vol_ma = volume.rolling(20, min_periods=20).mean()
    feats["volume_ratio"] = volume / vol_ma

    # 11) Return volatility (rolling std)
    r1 = close.pct_change()
    feats["vol_10"] = r1.rolling(10, min_periods=10).std()
    feats["vol_20"] = r1.rolling(20, min_periods=20).std()

    return feats


def build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Binary target: `horizon` gün sonra kapanış bugünün kapanışından yüksek mi?

    NOT: Son `horizon` candle'ın target'ı NaN olur — training'den drop etmek
    gerekir. `build_training_set` bunu otomatik yapar.
    """
    close = df["Close"]
    future = close.shift(-horizon)
    return (future > close).astype("float32").where(future.notna())


def build_training_set(
    df: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Feature + target birleştirip NaN satırlarını atar.

    Returns: (X, y) — ML için hazır.
    """
    X = build_features(df)
    y = build_target(df, horizon)

    # Hem feature hem target'ı geçerli olan satırları al
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask].astype(int)
