"""
Adım 2 — Teknik Ensemble Stratejisi.

Mantık:
  5 indicator'dan bireysel sinyaller (+1 / 0 / -1) topla, skor hesapla:
    1) SMA20 vs Close       → momentum temeli
    2) RSI(14)              → momentum teyidi
    3) MACD(12,26,9) hist   → trend değişim
    4) Bollinger pct_b      → band pozisyonu
    5) ADX(14) gate         → düşük trend gücünde skoru yarıla

  Skor ≥ 3 → long, aksi halde flat (long-only — short yok).

Sizing:
  - Aktif coin sayısı k → her aktif coin: min(0.33, 0.90/k)
  - 3 coin hepsi aktif: 0.30 + 0.30 + 0.30 = 0.90  (%10 cash buffer)
  - 1 coin aktif: 0.33 tek başına
  - Leverage = 1 (liquidation riski yok)

Warmup:
  ADX için min 26 candle (period * 2) gerekli → ilk 30 candle flat.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS
from utils.indicators import rsi, macd, bollinger, adx


class TechnicalEnsemble(BaseStrategy):
    """5 indicator konsensüs — long-only, leverage 1."""

    WARMUP = 30                 # indicator'lar dolması için minimum
    SMA_PERIOD = 20
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.0
    ADX_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    RSI_UPPER = 55              # RSI > 55 bullish, < 45 bearish, arada nötr
    RSI_LOWER = 45
    BB_UPPER = 0.6              # pct_b > 0.6 üst banda yakın
    BB_LOWER = 0.4
    ADX_WEAK = 20               # ADX < 20 → skoru yarıla

    LONG_THRESHOLD = 3          # 5 üzerinden 3 konsensüs → long
    MAX_TOTAL_ALLOC = 0.90      # %10 cash buffer
    MAX_PER_COIN = 1.0 / 3.0    # 0.333

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Pass 1: her coin için ensemble skoru hesapla → sinyal
        signals: dict[str, int] = {}
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.WARMUP:
                signals[coin] = 0
                continue
            signals[coin] = self._ensemble_signal(df)

        # Pass 2: aktif coin sayısına göre pay hesapla
        active_coins = [c for c, s in signals.items() if s == 1]
        k = len(active_coins)
        if k == 0:
            per_alloc = 0.0
        else:
            per_alloc = min(self.MAX_PER_COIN, self.MAX_TOTAL_ALLOC / k)

        # Decisions
        decisions: list[dict] = []
        for coin in COINS:
            if signals[coin] == 1:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": per_alloc,
                    "leverage":   1,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })
        return decisions

    def _ensemble_signal(self, df: pd.DataFrame) -> int:
        """5 indicator'dan skor çıkar, threshold karşılanırsa 1 döner."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        last_close = float(close.iloc[-1])

        # 1) SMA20
        sma20 = float(close.iloc[-self.SMA_PERIOD:].mean())
        sma_sig = 1 if last_close > sma20 else -1 if last_close < sma20 else 0

        # 2) RSI(14)
        rsi_s = rsi(close, self.RSI_PERIOD)
        rsi_last = rsi_s.iloc[-1]
        if pd.isna(rsi_last):
            return 0
        if rsi_last > self.RSI_UPPER:
            rsi_sig = 1
        elif rsi_last < self.RSI_LOWER:
            rsi_sig = -1
        else:
            rsi_sig = 0

        # 3) MACD histogram
        _, _, hist = macd(close, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL)
        hist_last = hist.iloc[-1]
        if pd.isna(hist_last):
            return 0
        macd_sig = 1 if hist_last > 0 else -1 if hist_last < 0 else 0

        # 4) Bollinger pct_b
        _, _, _, pct_b = bollinger(close, self.BB_PERIOD, self.BB_STD)
        pctb_last = pct_b.iloc[-1]
        if pd.isna(pctb_last):
            return 0
        if pctb_last > self.BB_UPPER:
            bb_sig = 1
        elif pctb_last < self.BB_LOWER:
            bb_sig = -1
        else:
            bb_sig = 0

        # 5) ADX gate
        adx_s, _, _ = adx(high, low, close, self.ADX_PERIOD)
        adx_last = adx_s.iloc[-1]
        if pd.isna(adx_last):
            return 0

        # Ham skor — 4 directional + 1 ADX-as-extra (güçlü trendse bonus)
        score = sma_sig + rsi_sig + macd_sig + bb_sig
        # ADX trend gücü: >25 ise ekstra +1 long/-1 short, zayıfsa skoru yarıla
        if adx_last >= 25:
            # Yön belirsiz olduğu için ADX kendi başına yön vermez,
            # sadece mevcut skorun önemini artırır (yine de +/-1 eşdeğeri)
            score = score * 1  # ADX güçlü → skor aynen geçerli
        elif adx_last < self.ADX_WEAK:
            score = score / 2.0  # zayıf trend → güvensizlik

        # Long-only: skor threshold'u aşıyorsa long, aksi halde flat
        if score >= self.LONG_THRESHOLD:
            return 1
        return 0
