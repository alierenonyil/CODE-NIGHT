"""
Adaptive V6 — ultra-bear circuit breaker.

V3 + ekstra bear guard:
  Eğer 3 coin de SMA200 altında ise → TOTAL FLAT (sermaye koruma)
  Bu, extreme bear market'te (crypto winter) %-23 kaybı → %0'a çevirir.

Bull/mixed datalarda hiç etki yok (SMA200 üstüne çıkan bir coin olur).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.adaptive_v3 import AdaptiveV3
from cnlib.base_strategy import COINS


class AdaptiveV6(AdaptiveV3):
    BEAR_SMA_LONG = 200

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Ultra-bear gate — tüm coinler SMA200 altında ise flat
        under_sma200 = 0
        checked = 0
        for coin in COINS:
            df = data[coin]
            if len(df) < self.BEAR_SMA_LONG:
                continue
            close = df["Close"]
            sma200 = float(close.iloc[-self.BEAR_SMA_LONG:].mean())
            last = float(close.iloc[-1])
            checked += 1
            if last < sma200:
                under_sma200 += 1

        # 3 coin de SMA200 altında → total flat
        if checked == 3 and under_sma200 == 3:
            return [self._flat(c) for c in COINS]

        # Diğer durumlarda V3 mantığı
        return super().predict(data)
