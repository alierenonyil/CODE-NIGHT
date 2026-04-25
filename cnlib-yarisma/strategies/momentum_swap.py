"""
Momentum Swap — autocorrelation lag-1 = 0.65 exploitation.

Data analizi:
  Real datada autocorr lag-1 = 0.65 (aşırı yüksek).
  Bu demektir ki: son candle return pozitifse, yarın da %65 pozitif.
  Son candle return negatifse, yarın da %65 negatif.

Strateji:
  - Last return > threshold  → yarın yükselecek → LONG  5x
  - Last return < -threshold → yarın düşecek  → SHORT 5x
  - |last return| ≤ threshold → belirsiz      → FLAT

Teorik expectancy (autocorr 0.65 ile):
  Doğru: +%10 (5x × %2)
  Yanlış: -%5 (5x × %1)
  E[daily] = 0.65 × 10 + 0.35 × (-5) = +4.75%
  (1.0475)^1570 = 10^31 teorik — sekstilyona (10^21) yaklaşır
  ...ama gerçek data volatility ile bu çok daha düşük olur.

Leverage 5x güvenli: liquidation eşiği %20, data'da sadece 1 aykırı olay.

Short izni dökümanda kesin onaylı (signal=-1).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class MomentumSwap(BaseStrategy):
    MIN_MAGNITUDE = 0.005        # |son return| > 0.5% — düşük gürültü filtresi
    LEVERAGE = 5
    MAX_PER_COIN = 0.32
    TOTAL_CAP = 0.96

    def predict(self, data: dict[str, Any]) -> list[dict]:
        signals: dict[str, int] = {}

        for coin in COINS:
            df: pd.DataFrame = data[coin]
            if len(df) < 2:
                signals[coin] = 0
                continue

            close = df["Close"]
            last_return = float(close.iloc[-1] / close.iloc[-2] - 1)

            if abs(last_return) < self.MIN_MAGNITUDE:
                signals[coin] = 0
            elif last_return > 0:
                signals[coin] = 1       # long (yarın da artacak)
            else:
                signals[coin] = -1      # short (yarın da düşecek)

        # Allocation
        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin":       coin,
                    "signal":     s,
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


class MomentumSwap10x(MomentumSwap):
    LEVERAGE = 10
    MAX_PER_COIN = 0.32


class MomentumSwap3x(MomentumSwap):
    LEVERAGE = 3


class MomentumSwap2x(MomentumSwap):
    LEVERAGE = 2
