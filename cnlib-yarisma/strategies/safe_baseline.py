"""
Safe Baseline — Base2x + bear protection.

Hedef: Mümkün olduğunca çok dataset'te pozitif kazanç + 0 wipeout.

Mantık:
  1. Bear gate: 3 coin'in TÜMÜ SMA50 altında VE
     son 30 günde -%10+ düştüyse → tamamen flat (sermaye koruma)
  2. Aksi halde: Close > SMA20 → long 2x leverage
  3. Per-coin: o coin SMA50 altındaysa flat
  4. Allocation 0.32 per coin, total cap 0.96

Liquidation eşiği 2x'te %50 düşüş — pratik olarak imkansız bir günde.
Bear gate sayesinde extended bear'de sermayeyi flat tutar (kayıp yok).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class SafeBaseline(BaseStrategy):
    SMA_SHORT = 20
    SMA_LONG = 50
    BEAR_DRIFT_DAYS = 30
    BEAR_DRIFT_THRESHOLD = -0.10
    LEVERAGE = 2
    MAX_PER_COIN = 0.32
    TOTAL_CAP = 0.96
    EXTREME_THRESHOLD = 0.15

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # 1. Bear gate: 3 coin'in tümü ciddi bear durumda mı?
        bear_count = 0
        for coin in COINS:
            df = data[coin]
            if len(df) < max(self.SMA_LONG, self.BEAR_DRIFT_DAYS + 1):
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            sma50 = float(close.iloc[-self.SMA_LONG:].mean())
            drift = (close.iloc[-1] / close.iloc[-self.BEAR_DRIFT_DAYS] - 1)
            if last < sma50 and drift < self.BEAR_DRIFT_THRESHOLD:
                bear_count += 1

        # Tüm coinler bear → flat
        if bear_count == 3:
            return [self._flat(c) for c in COINS]

        # 2. Per-coin signal
        signals = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < self.SMA_LONG:
                signals[coin] = 0
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.iloc[-self.SMA_SHORT:].mean())
            sma50 = float(close.iloc[-self.SMA_LONG:].mean())

            # Per-coin bear: kendisi SMA50 altıysa flat
            if last <= sma50 or last <= sma20:
                signals[coin] = 0
                continue

            # Extreme single-day
            ext = close.pct_change().iloc[-5:].abs().max()
            if pd.notna(ext) and ext > self.EXTREME_THRESHOLD:
                signals[coin] = 0
                continue

            signals[coin] = 1

        # 3. Allocation
        active = [c for c, s in signals.items() if s == 1]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions = []
        for coin in COINS:
            if signals[coin] == 1:
                decisions.append({
                    "coin": coin, "signal": 1,
                    "allocation": per_alloc, "leverage": self.LEVERAGE,
                })
            else:
                decisions.append(self._flat(coin))
        return decisions

    @staticmethod
    def _flat(coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}


class SafeBaseline3x(SafeBaseline):
    LEVERAGE = 3


class SafeBaseline1x(SafeBaseline):
    LEVERAGE = 1
