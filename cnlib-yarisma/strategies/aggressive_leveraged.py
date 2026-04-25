"""
Aggressive Leveraged — Katrilyon hedefi için 10x leverage stratejisi.

Felsefe:
  Rakipler 10^15+ multiplier yapıyor. Sadece compound + yüksek leverage
  bunu başarabilir. Tek liquidation = oyun biter, o yüzden çok selective.

Tier sistemi (leverage tier'ları):
  - TIER 10x: ÇOK güçlü sinyal (tüm filtreler yeşil + düşük vol + ADX>35)
             stop_loss = entry * 0.95  (liquidation %90'da olur, 5%'ten önce çıkıyoruz)
  - TIER 5x:  Güçlü sinyal (ana filtreler yeşil, vol makul)
             stop_loss = entry * 0.88  (liq %80'de)
  - TIER 3x:  Orta sinyal
             stop_loss = entry * 0.82
  - FLAT:     Diğer her durumda

Filtreler:
  1. Primary: Close > SMA20         (trend yukarı)
  2. Slope:   SMA10 > SMA30         (momentum confirmed)
  3. Strength: ADX > threshold      (güçlü trend)
  4. Volatility: ATR_pct < threshold (düşük vol → liquidation riski az)
  5. RSI range: 40 < RSI < 80       (aşırı alım değil ama momentum+)

Allocation:
  Aktif tier varsa 0.33 per coin (3 coin × 0.33 = 0.99 total).
  Stop-loss tetiğinde pozisyon kapanır, yeniden girmek için
  tüm filtrelerin tekrar yeşil olması + en az 1 candle beklemek.

Leverage ve stop-loss birlikte:
  Leverage 10x'te liquidation price = entry * 0.90 (long)
  Stop-loss = entry * 0.95 → SL liquidation'dan ÖNCE tetikler
  Böylece tüm sermaye kaybı yerine sadece ~%5 kayıp.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS
from utils.indicators import rsi, adx, atr


class AggressiveLeveraged(BaseStrategy):
    """10x/5x/3x leverage + stop-loss + super-selective entry."""

    WARMUP = 50
    SMA_FAST = 10
    SMA_MAIN = 20
    SMA_SLOW = 30

    # Tier thresholds
    ADX_TIER10 = 35
    ADX_TIER5  = 25
    ADX_TIER3  = 18

    ATR_PCT_TIER10 = 0.035     # günlük vol < 3.5% stable
    ATR_PCT_TIER5  = 0.055
    ATR_PCT_TIER3  = 0.08

    RSI_MIN = 45
    RSI_MAX = 80

    # Stop-loss (entry price'tan aşağı yüzde)
    SL_TIER10 = 0.95           # -5%
    SL_TIER5  = 0.88           # -12%
    SL_TIER3  = 0.82           # -18%

    ALLOC_PER_COIN = 0.33      # toplam 3 × 0.33 = 0.99 (validator OK)

    def predict(self, data: dict[str, Any]) -> list[dict]:
        decisions: list[dict] = []

        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.WARMUP:
                decisions.append(self._flat(coin))
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]

            last = float(close.iloc[-1])
            sma_main = float(close.iloc[-self.SMA_MAIN:].mean())

            # Primary: trend yukarı mı?
            if last <= sma_main:
                decisions.append(self._flat(coin))
                continue

            # Slope: fast SMA slow SMA üstünde mi?
            sma_fast = float(close.iloc[-self.SMA_FAST:].mean())
            sma_slow = float(close.iloc[-self.SMA_SLOW:].mean())
            if sma_fast <= sma_slow:
                decisions.append(self._flat(coin))
                continue

            # Indicators
            rsi_last = rsi(close, 14).iloc[-1]
            adx_last = adx(high, low, close, 14)[0].iloc[-1]
            atr_last = atr(high, low, close, 14).iloc[-1]

            if pd.isna(rsi_last) or pd.isna(adx_last) or pd.isna(atr_last):
                decisions.append(self._flat(coin))
                continue

            atr_pct = float(atr_last / last)

            # RSI range — aşırı olmamalı
            if not (self.RSI_MIN < rsi_last < self.RSI_MAX):
                decisions.append(self._flat(coin))
                continue

            # Tier seçimi
            if adx_last >= self.ADX_TIER10 and atr_pct < self.ATR_PCT_TIER10:
                leverage = 10
                sl = last * self.SL_TIER10
            elif adx_last >= self.ADX_TIER5 and atr_pct < self.ATR_PCT_TIER5:
                leverage = 5
                sl = last * self.SL_TIER5
            elif adx_last >= self.ADX_TIER3 and atr_pct < self.ATR_PCT_TIER3:
                leverage = 3
                sl = last * self.SL_TIER3
            else:
                decisions.append(self._flat(coin))
                continue

            decisions.append({
                "coin":       coin,
                "signal":     1,
                "allocation": self.ALLOC_PER_COIN,
                "leverage":   leverage,
                "stop_loss":  sl,
            })

        return decisions

    @staticmethod
    def _flat(coin: str) -> dict:
        return {
            "coin":       coin,
            "signal":     0,
            "allocation": 0.0,
            "leverage":   1,
        }
