"""
Swap V7 — Cross-coin direction confirmation.

Mantık:
  3 coin aynı yönde hareket ettiyse (tüm r1 +, veya tüm r1 -) →
  piyasa genelinde directional consensus → yüksek güven → FULL 10x.

  Sadece 2/3 aynı yönde → orta güven → 5x.
  1/3 → karışık → 3x veya flat.

Bu cross-correlation'ı exploit ediyor:
  Eğer 3 coin yüksek korelasyonlu ise (bizim train datada olabilir),
  aynı yönde hareket = gürültü değil gerçek trend → leverage at.

Plus V3 damper + extreme filter.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from cnlib.base_strategy import BaseStrategy, COINS


class SwapV7(BaseStrategy):
    MIN_MAGNITUDE = 0.0055
    DAMPER_THRESHOLD = -0.075
    DAMPER_CANDLES = 4
    EXTREME_THRESHOLD = 0.12
    EXTREME_WINDOW = 5

    # Cross-coin consensus tier
    LEV_3_CONSENSUS = 10    # 3 coin aynı yönde
    LEV_2_CONSENSUS = 5     # 2 coin aynı yönde
    LEV_1_OR_0 = 3          # karışık

    MAX_PER_COIN = 0.333
    TOTAL_CAP = 0.999

    def __init__(self):
        super().__init__()
        self.damper_remaining = 0

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Broad market damper
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.DAMPER_THRESHOLD:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1

        # Her coin'in son return'ü ve yön
        per_coin_r1 = {}
        per_coin_ext = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < self.EXTREME_WINDOW + 2:
                per_coin_r1[coin] = 0.0
                per_coin_ext[coin] = True  # extreme gibi davran
                continue
            close = df["Close"]
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            ext = close.pct_change().iloc[-self.EXTREME_WINDOW:].abs().max()
            per_coin_r1[coin] = r1
            per_coin_ext[coin] = pd.notna(ext) and ext > self.EXTREME_THRESHOLD

        # Cross-coin consensus
        positives = sum(1 for r in per_coin_r1.values() if r > self.MIN_MAGNITUDE)
        negatives = sum(1 for r in per_coin_r1.values() if r < -self.MIN_MAGNITUDE)
        consensus = max(positives, negatives)  # 0-3

        # Base leverage by consensus
        if consensus >= 3:
            base_lev = self.LEV_3_CONSENSUS
        elif consensus >= 2:
            base_lev = self.LEV_2_CONSENSUS
        else:
            base_lev = self.LEV_1_OR_0

        if damper_active:
            base_lev = min(base_lev, 5)  # damper cap

        # Per-coin signal
        signals = {}
        for coin in COINS:
            r1 = per_coin_r1[coin]
            if abs(r1) < self.MIN_MAGNITUDE or per_coin_ext[coin]:
                signals[coin] = 0
            elif r1 > 0:
                signals[coin] = 1
            else:
                signals[coin] = -1

        active = [c for c, s in signals.items() if s != 0]
        k = len(active)
        per_alloc = min(self.MAX_PER_COIN, self.TOTAL_CAP / k) if k > 0 else 0.0

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            if s != 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": per_alloc, "leverage": base_lev,
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions


class SwapV7Concentrated(SwapV7):
    """V7 + MaxConc allocation — top magnitude coin'e %80."""
    MAX_PER_COIN = 0.8

    def predict(self, data: dict[str, Any]) -> list[dict]:
        # Broad market damper
        recent_drops = []
        for coin in COINS:
            df = data[coin]
            if len(df) >= 2:
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
                recent_drops.append(r)
        if len(recent_drops) == 3 and sum(recent_drops) / 3 < self.DAMPER_THRESHOLD:
            self.damper_remaining = self.DAMPER_CANDLES
        damper_active = self.damper_remaining > 0
        if damper_active:
            self.damper_remaining -= 1

        # Her coin'in son return'ü
        per_coin_r1 = {}
        per_coin_ext = {}
        for coin in COINS:
            df = data[coin]
            if len(df) < self.EXTREME_WINDOW + 2:
                per_coin_r1[coin] = 0.0
                per_coin_ext[coin] = True
                continue
            close = df["Close"]
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
            ext = close.pct_change().iloc[-self.EXTREME_WINDOW:].abs().max()
            per_coin_r1[coin] = r1
            per_coin_ext[coin] = pd.notna(ext) and ext > self.EXTREME_THRESHOLD

        # Cross-coin consensus
        positives = sum(1 for r in per_coin_r1.values() if r > self.MIN_MAGNITUDE)
        negatives = sum(1 for r in per_coin_r1.values() if r < -self.MIN_MAGNITUDE)
        consensus = max(positives, negatives)

        if consensus >= 3:
            base_lev = self.LEV_3_CONSENSUS
        elif consensus >= 2:
            base_lev = self.LEV_2_CONSENSUS
        else:
            base_lev = self.LEV_1_OR_0

        if damper_active:
            base_lev = min(base_lev, 5)

        # Magnitude-weighted alloc
        signals = {}
        weights = {}
        for coin in COINS:
            r1 = per_coin_r1[coin]
            if abs(r1) < self.MIN_MAGNITUDE or per_coin_ext[coin]:
                signals[coin] = 0
                weights[coin] = 0.0
            else:
                signals[coin] = 1 if r1 > 0 else -1
                weights[coin] = abs(r1)

        total_w = sum(weights.values())
        allocations = {}
        if total_w > 0:
            raw = {c: (weights[c] / total_w) * self.TOTAL_CAP
                   for c in COINS if weights[c] > 0}
            capped = {c: min(a, self.MAX_PER_COIN) for c, a in raw.items()}
            total_c = sum(capped.values())
            if total_c > self.TOTAL_CAP:
                scale = self.TOTAL_CAP / total_c
                capped = {c: a * scale for c, a in capped.items()}
            allocations = capped

        decisions: list[dict] = []
        for coin in COINS:
            s = signals[coin]
            a = allocations.get(coin, 0.0)
            if s != 0 and a > 0:
                decisions.append({
                    "coin": coin, "signal": s,
                    "allocation": a, "leverage": base_lev,
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                  "allocation": 0.0, "leverage": 1})
        return decisions
