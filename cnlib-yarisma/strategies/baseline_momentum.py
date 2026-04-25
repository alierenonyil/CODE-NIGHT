"""
Adım 1 — Baseline Momentum Stratejisi.

Mantık:
  - 20 günlük basit momentum: son kapanış > son 20 günün SMA → long
  - Aksi halde flat (signal=0)
  - Leverage her zaman 1 (liquidation riski yok)
  - 3 coin için aynı mantık, her biri portföyün 1/3'ü
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class BaselineMomentum(BaseStrategy):
    """20-günlük momentum baseline — ilk işaret, framework doğrulaması."""

    # Rolling window ve coin başı allocation
    LOOKBACK = 20
    ALLOC_PER_COIN = 1.0 / 3.0  # 3 coin = 0.333...

    def predict(self, data: dict[str, Any]) -> list[dict]:
        decisions: list[dict] = []

        for coin in COINS:
            df: pd.DataFrame = data[coin]

            # İlk N candle: yeterli history yok → flat
            if len(df) < self.LOOKBACK:
                decisions.append(self._flat(coin))
                continue

            closes = df["Close"]
            son_fiyat = float(closes.iloc[-1])
            sma20 = float(closes.iloc[-self.LOOKBACK:].mean())

            if son_fiyat > sma20:
                # Momentum yukarı → long
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": self.ALLOC_PER_COIN,
                    "leverage":   1,
                })
            else:
                # Momentum yok → flat
                decisions.append(self._flat(coin))

        return decisions

    @staticmethod
    def _flat(coin: str) -> dict:
        """signal=0 ise allocation=0 zorunlu (validator kuralı)."""
        return {
            "coin":       coin,
            "signal":     0,
            "allocation": 0.0,
            "leverage":   1,
        }
