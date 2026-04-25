"""
Leveraged Baseline — Baseline mantığı + yüksek leverage, stop-loss YOK.

Kritik bulgu (data analizi):
  - 3x leverage: 0 liquidation riski (hiç -33% intrabar düşüş yok)
  - 5x leverage: sadece 1 gün -20% altı (kapcoin)
  - 10x leverage: 127 gün -10% altı → çok riskli
  - 2x leverage: 0 risk

Stratejimiz:
  Baseline'ın `Close > SMA20` sinyali + kontrollü leverage.
  Stop-loss YOK çünkü liquidation zaten imkansız (3x'te) veya çok nadir (5x'te).
  Stop-loss koysak trendleri erken kestirip baseline'dan düşük kalıyor
  (aggressive_leveraged denemesi bunu gösterdi).

Factory:
  LeveragedBaseline3x, 5x, 10x — parametrik.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class LeveragedBaseline(BaseStrategy):
    """Baseline + configurable leverage."""

    LEVERAGE = 3              # override in subclasses
    SMA_PERIOD = 20
    MAX_PER_COIN = 0.33
    TOTAL_ALLOC_CAP = 0.96    # %4 cash buffer

    def predict(self, data: dict[str, Any]) -> list[dict]:
        signals: dict[str, int] = {}
        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < self.SMA_PERIOD:
                signals[coin] = 0
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            sma = float(close.iloc[-self.SMA_PERIOD:].mean())
            signals[coin] = 1 if last > sma else 0

        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_ALLOC_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            if signals[coin] == 1:
                decisions.append({
                    "coin":       coin,
                    "signal":     1,
                    "allocation": per_alloc,
                    "leverage":   self.LEVERAGE,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })
        return decisions


class LeveragedBaseline2x(LeveragedBaseline):
    LEVERAGE = 2


class LeveragedBaseline3x(LeveragedBaseline):
    LEVERAGE = 3


class LeveragedBaseline5x(LeveragedBaseline):
    LEVERAGE = 5


class LeveragedBaseline10x(LeveragedBaseline):
    LEVERAGE = 10
