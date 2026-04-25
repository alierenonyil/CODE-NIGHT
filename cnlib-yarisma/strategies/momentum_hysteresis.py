"""
Momentum Hysteresis — Baseline'ı geçmek için 2. deneme.

Sorun tanısı:
  Baseline (Close > SMA20) 253 işlem açıyor; muhtemelen birçoğu whipsaw:
  fiyat SMA20 civarında dalgalanınca aç-kapa aç-kapa → küçük kayıplar
  birikiyor. Trend yakalandığında kazanç büyük ama whipsaw kemiriyor.

Çözüm — double-band (hysteresis):
  - Giriş: Close > SMA20 * 1.005 (yukarı kırma net olmalı)
  - Çıkış: Close < SMA20 * 0.97  (3% düşüşe kadar pozisyonu koru)
  - Bu aralıkta bir pozisyon varsa bırak, yoksa flat

Ayrıca:
  - Long-only + leverage 1
  - Alloc 0.33 per coin (baseline ile aynı), 3 coin toplam 0.99 cap
  - Failed_open fix: k aktif coin sayısına göre allocation düşürerek 0.96 total.

State:
  self.open_long[coin] = True/False
  (leverage 1 ve TP/SL yok, intrabar liquidation mümkün değil → senkron)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class MomentumHysteresis(BaseStrategy):
    """Close > SMA20 + hysteresis + dinamik alloc (leverage 1, long-only)."""

    SMA_PERIOD = 20
    ENTRY_BAND = 1.005      # close > SMA * 1.005 → giriş
    EXIT_BAND = 0.97        # close < SMA * 0.97 → çıkış
    MAX_PER_COIN = 0.33
    TOTAL_ALLOC_CAP = 0.96

    def __init__(self) -> None:
        super().__init__()
        self.open_long: dict[str, bool] = {c: False for c in COINS}

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Pass 1: sinyal belirle (hysteresis kuralıyla)
        signals: dict[str, int] = {}
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.SMA_PERIOD:
                signals[coin] = 0
                continue

            last = float(df["Close"].iloc[-1])
            sma = float(df["Close"].iloc[-self.SMA_PERIOD:].mean())

            if self.open_long[coin]:
                # Pozisyon açık — sadece EXIT_BAND altına düşerse kapat
                if last < sma * self.EXIT_BAND:
                    self.open_long[coin] = False
                    signals[coin] = 0
                else:
                    signals[coin] = 1
            else:
                # Pozisyon kapalı — ENTRY_BAND üstüne kırarsa aç
                if last > sma * self.ENTRY_BAND:
                    self.open_long[coin] = True
                    signals[coin] = 1
                else:
                    signals[coin] = 0

        # Pass 2: aktif coin sayısına göre alloc (cash buffer)
        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        if k == 0:
            per_alloc = 0.0
        else:
            per_alloc = min(self.MAX_PER_COIN, self.TOTAL_ALLOC_CAP / k)

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
